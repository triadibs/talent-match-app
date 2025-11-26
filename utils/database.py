# database.py
# -*- coding: utf-8 -*-
"""
Database Manager for Talent Match System
Using psycopg3 with Supabase Pooler

Updated:
- Use TEMP TABLEs for isolation
- Persist job_vacancy_id through pipeline steps
- Return results only for requested vacancy
- get_summary_results filters by vacancy
"""

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional


class DatabaseManager:
    """Manages database connections and queries"""

    def __init__(self):
        """Initialize database connection using Streamlit secrets"""
        try:
            self.conn_params = {
                "host": st.secrets["database"]["host"],
                "port": st.secrets["database"]["port"],
                "dbname": st.secrets["database"]["dbname"],
                "user": st.secrets["database"]["user"],
                "password": st.secrets["database"]["password"],
            }
            self._test_connection()
        except Exception as e:
            st.error(f"Database configuration error: {e}")
            raise

    def _test_connection(self):
        """Test database connection"""
        try:
            conn = self.get_connection()
            conn.close()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to database: {e}")

    def get_connection(self):
        """Return psycopg3 connection"""
        # row_factory=dict_row returns dict-like rows which pandas can consume
        return psycopg.connect(**self.conn_params, row_factory=dict_row)

    # ---------------------------------------------
    # GENERIC QUERY FUNCTION
    # ---------------------------------------------
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> pd.DataFrame:
        """Execute SQL query and return results (returns empty DataFrame if no rows)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)

                if fetch:
                    rows = cur.fetchall()
                    return pd.DataFrame(rows) if rows else pd.DataFrame()
                else:
                    conn.commit()
                    return pd.DataFrame()

    # ---------------------------------------------
    # HOME PAGE QUERIES
    # ---------------------------------------------
    def get_total_employees(self) -> int:
        query = "SELECT COUNT(*) as count FROM employees"
        result = self.execute_query(query)
        return int(result["count"].iloc[0]) if not result.empty else 0

    def get_high_performers_count(self) -> int:
        query = """
        SELECT COUNT(DISTINCT employee_id) as count
        FROM performance_yearly
        WHERE rating = 5
        AND year = (SELECT MAX(year) FROM performance_yearly)
        """
        result = self.execute_query(query)
        return int(result["count"].iloc[0]) if not result.empty else 0

    def get_total_vacancies(self) -> int:
        query = "SELECT COUNT(*) as count FROM talent_benchmarks"
        result = self.execute_query(query)
        return int(result["count"].iloc[0]) if not result.empty else 0

    def get_recent_vacancies(self, limit: int = 5) -> pd.DataFrame:
        query = """
        SELECT
            job_vacancy_id,
            role_name,
            job_level,
            role_purpose,
            created_at,
            array_length(selected_talent_ids, 1) AS benchmark_count
        FROM talent_benchmarks
        ORDER BY created_at DESC
        LIMIT %s
        """
        return self.execute_query(query, params=(limit,))

    # ---------------------------------------------
    # HIGH PERFORMER LIST
    # ---------------------------------------------
    def get_high_performers(self) -> pd.DataFrame:
        query = """
        SELECT DISTINCT
            e.employee_id,
            e.fullname,
            pos.name AS position,
            dir.name AS directorate,
            gr.name AS grade
        FROM employees e
        JOIN performance_yearly p ON e.employee_id = p.employee_id
        LEFT JOIN dim_positions pos ON e.position_id = pos.position_id
        LEFT JOIN dim_directorates dir ON e.directorate_id = dir.directorate_id
        LEFT JOIN dim_grades gr ON e.grade_id = gr.grade_id
        WHERE p.rating = 5
        AND p.year = (SELECT MAX(year) FROM performance_yearly)
        ORDER BY e.fullname
        """
        return self.execute_query(query)

    # ---------------------------------------------
    # INSERT VACANCY
    # ---------------------------------------------
    def insert_vacancy(
        self,
        role_name: str,
        job_level: str,
        role_purpose: str,
        selected_talent_ids: List[str],
        weights_config: Optional[Dict] = None,
    ) -> int:
        """Insert vacancy using psycopg3 Jsonb"""

        query = """
        INSERT INTO talent_benchmarks
        (role_name, job_level, role_purpose, selected_talent_ids, weights_config)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING job_vacancy_id
        """

        json_data = Jsonb(weights_config) if weights_config else None

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        role_name,
                        job_level,
                        role_purpose,
                        selected_talent_ids,
                        json_data,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                return row["job_vacancy_id"]

    # ---------------------------------------------
    # RUN MATCHING PIPELINE (revised: temp tables + job_vacancy_id preserved)
    # ---------------------------------------------
    def run_matching_query(self, job_vacancy_id: int) -> pd.DataFrame:
        """
        Revised matching pipeline using TEMP TABLEs and preserving job_vacancy_id.
        Returns detailed per-TV rows joined with tgv/tgv_match_rate/final_match_rate
        for the given job_vacancy_id only.
        """
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cur:
                    # Use TEMP TABLES to avoid persistent schema pollution and to isolate runs per session.
                    # All temp tables will exist for the duration of this session (the connection).
                    # Note: we use one connection and one transaction for the pipeline run.
                    cur.execute("BEGIN;")

                    # ===== 1) DROP TEMP TABLES IF EXIST (defensive, though TEMP tables are session-scoped) =====
                    cur.execute(
                        """
                        DROP TABLE IF EXISTS temp.tb_vacancy;
                        DROP TABLE IF EXISTS temp.tb_latest_year;
                        DROP TABLE IF EXISTS temp.tb_selected_talents;
                        DROP TABLE IF EXISTS temp.tb_profiles_psych_norm;
                        DROP TABLE IF EXISTS temp.tb_tv_scores_long;
                        DROP TABLE IF EXISTS temp.tb_baseline_per_tv;
                        DROP TABLE IF EXISTS temp.tb_employees_with_tv;
                        DROP TABLE IF EXISTS temp.tb_tv_match_calc;
                        DROP TABLE IF EXISTS temp.tb_tv_match_with_weights;
                        DROP TABLE IF EXISTS temp.tb_tgv_aggregation;
                        DROP TABLE IF EXISTS temp.tb_tgv_with_weights;
                        DROP TABLE IF EXISTS temp.tb_final_aggregation;
                        """,
                    )

                    # ===== 2) CREATE VACANCY TEMP TABLE =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_vacancy AS
                        SELECT job_vacancy_id, role_name, job_level, role_purpose, selected_talent_ids, weights_config
                        FROM talent_benchmarks
                        WHERE job_vacancy_id = %s
                        """,
                        (job_vacancy_id,),
                    )

                    # Ensure vacancy exists
                    cur.execute("SELECT COUNT(*) AS cnt FROM temp.tb_vacancy;")
                    cnt_v = cur.fetchone()["cnt"]
                    if cnt_v == 0:
                        conn.rollback()
                        raise ValueError(f"Job vacancy {job_vacancy_id} not found in talent_benchmarks.")

                    # ===== 3) LATEST YEAR =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_latest_year AS
                        SELECT MAX(year) AS max_year
                        FROM competencies_yearly
                        """
                    )

                    # ===== 4) SELECTED TALENTS (include job_vacancy_id) =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_selected_talents AS
                        SELECT v.job_vacancy_id,
                               unnest(v.selected_talent_ids) AS employee_id,
                               v.weights_config
                        FROM temp.tb_vacancy v
                        """
                    )

                    # ===== 5) PROFILES PSYCH NORM (copy) =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_profiles_psych_norm AS
                        SELECT
                          employee_id,
                          iq,
                          gtq,
                          faxtor,
                          pauli,
                          tiki
                        FROM profiles_psych
                        """
                    )

                    # ===== 6) TV SCORES LONG (UNION ALL) =====
                    # Note: We keep job_vacancy_id only where relevant (for baseline calc we join selected talents)
                    cur.execute(
                        r"""
                        CREATE TEMP TABLE temp.tb_tv_scores_long AS
                        SELECT employee_id,
                               'cognitive_ability'::text AS tgv_name,
                               'iq'::text AS tv_name,
                               (iq::text)::numeric AS tv_score,
                               'higher_better'::text AS scoring_direction
                        FROM temp.tb_profiles_psych_norm
                        WHERE iq IS NOT NULL
                          AND trim(coalesce(iq::text,'')) <> ''
                          AND (iq::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

                        UNION ALL
                        SELECT employee_id, 'cognitive_ability', 'gtq', (gtq::text)::numeric, 'higher_better'
                        FROM temp.tb_profiles_psych_norm
                        WHERE gtq IS NOT NULL AND trim(coalesce(gtq::text,'')) <> ''
                          AND (gtq::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

                        UNION ALL
                        SELECT employee_id, 'attention_processing', 'faxtor', (faxtor::text)::numeric, 'higher_better'
                        FROM temp.tb_profiles_psych_norm
                        WHERE faxtor IS NOT NULL AND trim(coalesce(faxtor::text,'')) <> ''
                          AND (faxtor::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

                        UNION ALL
                        SELECT employee_id, 'attention_processing', 'pauli', (pauli::text)::numeric, 'higher_better'
                        FROM temp.tb_profiles_psych_norm
                        WHERE pauli IS NOT NULL AND trim(coalesce(pauli::text,'')) <> ''
                          AND (pauli::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

                        UNION ALL
                        SELECT employee_id, 'attention_processing', 'tiki', (tiki::text)::numeric, 'higher_better'
                        FROM temp.tb_profiles_psych_norm
                        WHERE tiki IS NOT NULL AND trim(coalesce(tiki::text,'')) <> ''
                          AND (tiki::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

                        UNION ALL
                        SELECT c.employee_id, 'leadership_competencies', c.pillar_code::text AS tv_name, (c.score::text)::numeric, 'higher_better'
                        FROM competencies_yearly c, temp.tb_latest_year ly
                        WHERE c.year = ly.max_year
                          AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
                          AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

                        UNION ALL
                        SELECT ps.employee_id, 'work_preferences', ('papi_' || lower(ps.scale_code))::text AS tv_name,
                               (ps.score::text)::numeric,
                               CASE WHEN ps.scale_code IN ('Z','K') THEN 'lower_better' ELSE 'higher_better' END
                        FROM papi_scores ps
                        WHERE ps.score IS NOT NULL AND trim(coalesce(ps.score::text, '')) <> ''
                          AND (ps.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

                        UNION ALL
                        SELECT e.employee_id, 'experience', 'years_of_service_months',
                               (e.years_of_service_months::text)::numeric, 'higher_better'
                        FROM employees e
                        WHERE e.years_of_service_months IS NOT NULL
                          AND trim(coalesce(e.years_of_service_months::text, '')) <> ''
                          AND (e.years_of_service_months::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
                        """
                    )

                    # ===== 7) BASELINE PER TV (mean & stddev from selected benchmarks) =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_baseline_per_tv AS
                        SELECT
                          st.job_vacancy_id,
                          tv.tgv_name,
                          tv.tv_name,
                          tv.scoring_direction,
                          AVG(tv.tv_score) AS baseline_mean,
                          STDDEV_POP(tv.tv_score) AS baseline_stddev,
                          COUNT(DISTINCT tv.employee_id) AS benchmark_count
                        FROM temp.tb_selected_talents st
                        JOIN temp.tb_tv_scores_long tv ON tv.employee_id = st.employee_id
                        GROUP BY st.job_vacancy_id, tv.tgv_name, tv.tv_name, tv.scoring_direction
                        """
                    )

                    # ===== 8) EMPLOYEES WITH TV (include job_vacancy_id via CROSS JOIN tb_vacancy) =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_employees_with_tv AS
                        SELECT
                          e.employee_id,
                          e.fullname,
                          dir.name AS directorate,
                          pos.name AS role,
                          gr.name AS grade,
                          tv.tgv_name,
                          tv.tv_name,
                          tv.tv_score AS user_score,
                          tv.scoring_direction,
                          b.baseline_mean,
                          b.baseline_stddev,
                          v.weights_config,
                          v.job_vacancy_id
                        FROM employees e
                        LEFT JOIN dim_directorates dir ON e.directorate_id = dir.directorate_id
                        LEFT JOIN dim_positions pos ON e.position_id = pos.position_id
                        LEFT JOIN dim_grades gr ON e.grade_id = gr.grade_id
                        CROSS JOIN temp.tb_vacancy v
                        LEFT JOIN temp.tb_tv_scores_long tv ON tv.employee_id = e.employee_id
                        LEFT JOIN temp.tb_baseline_per_tv b
                          ON b.job_vacancy_id = v.job_vacancy_id
                          AND b.tgv_name = tv.tgv_name
                          AND b.tv_name = tv.tv_name
                        WHERE tv.tv_score IS NOT NULL
                        """
                    )

                    # ===== 9) TV MATCH CALC (z-score logistic, fallback to ratio) =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_tv_match_calc AS
                        SELECT
                          e.*,
                          CASE
                            WHEN e.baseline_mean IS NULL THEN NULL
                            WHEN e.scoring_direction = 'boolean' THEN
                              CASE WHEN e.user_score = e.baseline_mean THEN 100.0 ELSE 0.0 END
                            WHEN e.baseline_stddev IS NOT NULL AND e.baseline_stddev <> 0 THEN
                              (
                                100.0 * (1.0 / (1.0 + exp( - ( (e.user_score - e.baseline_mean) / NULLIF(e.baseline_stddev, 0) ) ) )))
                            WHEN e.baseline_mean IS NOT NULL AND (e.baseline_stddev IS NULL OR e.baseline_stddev = 0) THEN
                              CASE
                                WHEN e.scoring_direction = 'lower_better' THEN
                                  CASE WHEN e.user_score IS NULL OR e.baseline_mean = 0 THEN NULL
                                  ELSE LEAST(GREATEST(((2.0 * e.baseline_mean - e.user_score) / NULLIF(e.baseline_mean,0)) * 100.0, 0.0), 100.0)
                                  END
                                ELSE
                                  CASE WHEN e.user_score IS NULL OR e.baseline_mean = 0 THEN NULL
                                  ELSE LEAST(GREATEST((e.user_score / NULLIF(e.baseline_mean,0)) * 100.0, 0.0), 100.0)
                                  END
                              END
                            ELSE
                              NULL
                          END AS tv_match_rate
                        FROM temp.tb_employees_with_tv e
                        """
                    )

                    # ===== 10) TV MATCH WITH WEIGHTS =====
                    cur.execute(
                        r"""
                        CREATE TEMP TABLE temp.tb_tv_match_with_weights AS
                        SELECT
                          t.*,
                          CASE
                            WHEN t.weights_config IS NULL THEN 1.0
                            WHEN (t.weights_config #>> ARRAY['TV_weights', t.tgv_name, t.tv_name]) ~ '^\s*[-+]?\d+(\.\d+)?\s*$'
                              THEN (t.weights_config #>> ARRAY['TV_weights', t.tgv_name, t.tv_name])::numeric
                            ELSE 1.0
                          END AS tv_weight
                        FROM temp.tb_tv_match_calc t
                        """
                    )

                    # ===== 11) TGV AGGREGATION (preserve job_vacancy_id) =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_tgv_aggregation AS
                        SELECT
                          v.job_vacancy_id,
                          employee_id,
                          fullname,
                          directorate,
                          role,
                          grade,
                          tgv_name,
                          CASE
                            WHEN SUM(tv_weight) FILTER (WHERE tv_match_rate IS NOT NULL) = 0 THEN NULL
                            ELSE SUM(tv_match_rate * tv_weight) FILTER (WHERE tv_match_rate IS NOT NULL)
                                 / NULLIF(SUM(tv_weight) FILTER (WHERE tv_match_rate IS NOT NULL), 0)
                          END AS tgv_match_rate,
                          COUNT(*) FILTER (WHERE tv_match_rate IS NOT NULL) AS tv_count_with_baseline,
                          MIN(weights_config::text)::jsonb AS weights_config
                        FROM temp.tb_tv_match_with_weights tm
                        JOIN temp.tb_vacancy v ON v.job_vacancy_id = tm.job_vacancy_id
                        GROUP BY v.job_vacancy_id, employee_id, fullname, directorate, role, grade, tgv_name
                        """
                    )

                    # ===== 12) TGV WITH WEIGHTS =====
                    cur.execute(
                        r"""
                        CREATE TEMP TABLE temp.tb_tgv_with_weights AS
                        SELECT
                          t.*,
                          CASE
                            WHEN t.weights_config IS NULL THEN
                              CASE t.tgv_name
                                WHEN 'interpersonal_skills' THEN 0.612
                                WHEN 'leadership_competencies' THEN 0.314
                                WHEN 'execution_competencies' THEN 0.041
                                WHEN 'attention_processing' THEN 0.016
                                WHEN 'cognitive_ability' THEN 0.009
                                WHEN 'experience' THEN 0.005
                                WHEN 'work_preferences' THEN 0.003
                                ELSE 1.0 / 7.0
                              END
                            WHEN (t.weights_config #>> ARRAY['TGV_weights', t.tgv_name]) ~ '^\s*[-+]?\d+(\.\d+)?\s*$'
                              THEN (t.weights_config #>> ARRAY['TGV_weights', t.tgv_name])::numeric
                            ELSE
                              CASE t.tgv_name
                                WHEN 'interpersonal_skills' THEN 0.612
                                WHEN 'leadership_competencies' THEN 0.314
                                WHEN 'execution_competencies' THEN 0.041
                                WHEN 'attention_processing' THEN 0.016
                                WHEN 'cognitive_ability' THEN 0.009
                                WHEN 'experience' THEN 0.005
                                WHEN 'work_preferences' THEN 0.003
                                ELSE 1.0 / 7.0
                              END
                          END AS tgv_weight
                        FROM temp.tb_tgv_aggregation t
                        """
                    )

                    # ===== 13) FINAL AGGREGATION (preserve job_vacancy_id) =====
                    cur.execute(
                        """
                        CREATE TEMP TABLE temp.tb_final_aggregation AS
                        SELECT
                          job_vacancy_id,
                          employee_id,
                          fullname,
                          directorate,
                          role,
                          grade,
                          CASE
                            WHEN SUM(tgv_weight) FILTER (WHERE tgv_match_rate IS NOT NULL) = 0 THEN NULL
                            ELSE SUM(tgv_match_rate * tgv_weight) FILTER (WHERE tgv_match_rate IS NOT NULL)
                                 / NULLIF(SUM(tgv_weight) FILTER (WHERE tgv_match_rate IS NOT NULL), 0)
                          END AS final_match_rate,
                          SUM(CASE WHEN tgv_match_rate IS NOT NULL THEN 1 ELSE 0 END) AS tgv_count_with_baseline
                        FROM temp.tb_tgv_with_weights
                        GROUP BY job_vacancy_id, employee_id, fullname, directorate, role, grade
                        """
                    )

                    # commit all DDL/DML above
                    conn.commit()

                    # ===== 14) FINAL SELECT - RETURN DETAILED RESULTS FOR REQUESTED VACANCY =====
                    final_sql = """
                    SELECT
                      tm.job_vacancy_id,
                      tm.employee_id,
                      tm.fullname,
                      tm.directorate,
                      tm.role,
                      tm.grade,
                      tm.tgv_name,
                      tm.tv_name,
                      ROUND(tm.baseline_mean::numeric, 2) AS baseline_score,
                      ROUND(tm.user_score::numeric, 2) AS user_score,
                      ROUND(tm.tv_match_rate::numeric, 2) AS tv_match_rate,
                      ROUND(tgv.tgv_match_rate::numeric, 4) AS tgv_match_rate,
                      ROUND(fa.final_match_rate::numeric, 4) AS final_match_rate
                    FROM temp.tb_tv_match_calc tm
                    LEFT JOIN temp.tb_tgv_with_weights tgv
                      ON tgv.employee_id = tm.employee_id
                     AND tgv.tgv_name = tm.tgv_name
                     AND tgv.job_vacancy_id = tm.job_vacancy_id
                    LEFT JOIN temp.tb_final_aggregation fa
                      ON fa.employee_id = tm.employee_id
                     AND fa.job_vacancy_id = tm.job_vacancy_id
                    WHERE tm.baseline_mean IS NOT NULL
                      AND tm.job_vacancy_id = %s
                    ORDER BY fa.final_match_rate DESC NULLS LAST,
                             tm.employee_id, tm.tgv_name, tm.tv_name
                    """

                    df = pd.read_sql_query(final_sql, conn, params=(job_vacancy_id,))

                    # sanitize numeric columns
                    numeric_cols = ['baseline_score', 'user_score', 'tv_match_rate', 'tgv_match_rate', 'final_match_rate']
                    for c in numeric_cols:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors='coerce')

                    # rename final_match_rate -> final_match_rate_percentage for backward compatibility
                    if 'final_match_rate' in df.columns and 'final_match_rate_percentage' not in df.columns:
                        df = df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})

                    # upscale if values in 0..1
                    if 'final_match_rate_percentage' in df.columns:
                        maxv = df['final_match_rate_percentage'].max(skipna=True)
                        if pd.notna(maxv) and maxv <= 1.0:
                            df['final_match_rate_percentage'] = df['final_match_rate_percentage'] * 100.0

                    return df

            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                st.error(f"❌ SQL Pipeline Error: {str(e)}")
                raise

    # ---------------------------------------------
    # SUMMARY RESULTS
    # ---------------------------------------------
    def get_summary_results(self, job_vacancy_id: int, limit: int = 50) -> pd.DataFrame:
        query = """
        SELECT
          fa.employee_id,
          fa.fullname,
          fa.directorate,
          fa.role,
          fa.grade,
          ROUND(fa.final_match_rate::numeric, 2) AS final_match_rate_percentage,
          fa.tgv_count_with_baseline
        FROM temp.tb_final_aggregation fa
        WHERE fa.job_vacancy_id = %s
          AND fa.final_match_rate IS NOT NULL
        ORDER BY fa.final_match_rate DESC
        LIMIT %s
        """
        # Note: temp.tb_final_aggregation exists only within a session where run_matching_query was executed.
        # If you want persistent summary across sessions, query the permanent table or store snapshots.
        return self.execute_query(query, params=(job_vacancy_id, limit))

    def get_vacancy_info(self, job_vacancy_id: int) -> Dict:
        query = "SELECT * FROM talent_benchmarks WHERE job_vacancy_id = %s"
        result = self.execute_query(query, params=(job_vacancy_id,))
        return result.iloc[0].to_dict() if not result.empty else {}
