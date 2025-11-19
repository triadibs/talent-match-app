"""
Database Manager for Talent Match System
Using psycopg3 with Supabase Pooler
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
                'host': st.secrets["database"]["host"],
                'port': st.secrets["database"]["port"],
                'dbname': st.secrets["database"]["dbname"],
                'user': st.secrets["database"]["user"],
                'password': st.secrets["database"]["password"]
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
        return psycopg.connect(**self.conn_params, row_factory=dict_row)

    # ---------------------------------------------
    # GENERIC QUERY FUNCTION
    # ---------------------------------------------
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> pd.DataFrame:
        """Execute SQL query and return results (returns empty DataFrame if no rows)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Handle params None explicitly to avoid passing None to execute
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
        return int(result['count'].iloc[0]) if not result.empty else 0

    def get_high_performers_count(self) -> int:
        query = """
        SELECT COUNT(DISTINCT employee_id) as count
        FROM performance_yearly
        WHERE rating = 5 
        AND year = (SELECT MAX(year) FROM performance_yearly)
        """
        result = self.execute_query(query)
        return int(result['count'].iloc[0]) if not result.empty else 0

    def get_total_vacancies(self) -> int:
        query = "SELECT COUNT(*) as count FROM talent_benchmarks"
        result = self.execute_query(query)
        return int(result['count'].iloc[0]) if not result.empty else 0

    def get_recent_vacancies(self, limit: int = 5) -> pd.DataFrame:
        query = f"""
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
        # parameterize limit
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
        weights_config: Optional[Dict] = None
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
                        json_data
                    )
                )
                row = cur.fetchone()
                conn.commit()
                return row["job_vacancy_id"]

    # ---------------------------------------------
    # RUN MATCHING PIPELINE
    # ---------------------------------------------
        def run_matching_query(self, job_vacancy_id: int) -> pd.DataFrame:
        """
        Run COMPLETE matching pipeline: create all temp tables, then return detailed results.
        This executes the FULL SQL workflow from your Step 2.
        """
        
        with self.get_connection() as conn:
            cur = conn.cursor()

        
        try:
            # ===== 1) DROP OLD TABLES =====
            cur.execute("""
            DROP TABLE IF EXISTS tb_vacancy CASCADE;
            DROP TABLE IF EXISTS tb_latest_year CASCADE;
            DROP TABLE IF EXISTS tb_selected_talents CASCADE;
            DROP TABLE IF EXISTS tb_profiles_psych_norm CASCADE;
            DROP TABLE IF EXISTS tb_tv_scores_long CASCADE;
            DROP TABLE IF EXISTS tb_baseline_per_tv CASCADE;
            DROP TABLE IF EXISTS tb_employees_with_tv CASCADE;
            DROP TABLE IF EXISTS tb_tv_match_calc CASCADE;
            DROP TABLE IF EXISTS tb_tv_match_with_weights CASCADE;
            DROP TABLE IF EXISTS tb_tgv_aggregation CASCADE;
            DROP TABLE IF EXISTS tb_tgv_with_weights CASCADE;
            DROP TABLE IF EXISTS tb_final_aggregation CASCADE;
            """)
            
            # ===== 2) CREATE VACANCY TABLE =====
            cur.execute(f"""
            CREATE TABLE tb_vacancy AS
            SELECT
              j.job_vacancy_id,
              j.role_name,
              j.job_level,
              j.role_purpose,
              j.selected_talent_ids,
              j.weights_config
            FROM talent_benchmarks j
            WHERE j.job_vacancy_id = {job_vacancy_id}
            """)
            
            # ===== 3) LATEST YEAR =====
            cur.execute("""
            CREATE TABLE tb_latest_year AS
            SELECT MAX(year) AS max_year
            FROM competencies_yearly
            """)
            
            # ===== 4) SELECTED TALENTS =====
            cur.execute("""
            CREATE TABLE tb_selected_talents AS
            SELECT v.job_vacancy_id,
                   unnest(v.selected_talent_ids) AS employee_id,
                   v.weights_config
            FROM tb_vacancy v
            """)
            
            # ===== 5) PROFILES PSYCH NORM =====
            cur.execute("""
            CREATE TABLE tb_profiles_psych_norm AS
            SELECT
              employee_id,
              iq,
              gtq,
              faxtor, pauli, tiki
            FROM profiles_psych
            """)
            
            # ===== 6) TV SCORES LONG (COMPLETE UNION ALL) =====
            cur.execute("""
            CREATE TABLE tb_tv_scores_long AS
            SELECT employee_id,
                   'cognitive_ability'::text AS tgv_name,
                   'iq'::text AS tv_name,
                   (iq::text)::numeric AS tv_score,
                   'higher_better'::text AS scoring_direction
            FROM tb_profiles_psych_norm
            WHERE iq IS NOT NULL
              AND trim(coalesce(iq::text,'')) <> ''
              AND (iq::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT employee_id, 'cognitive_ability', 'gtq', (gtq::text)::numeric, 'higher_better'
            FROM tb_profiles_psych_norm
            WHERE gtq IS NOT NULL AND trim(coalesce(gtq::text,'')) <> '' 
              AND (gtq::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT employee_id, 'attention_processing', 'faxtor', (faxtor::text)::numeric, 'higher_better'
            FROM tb_profiles_psych_norm
            WHERE faxtor IS NOT NULL AND trim(coalesce(faxtor::text,'')) <> ''
              AND (faxtor::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT employee_id, 'attention_processing', 'pauli', (pauli::text)::numeric, 'higher_better'
            FROM tb_profiles_psych_norm
            WHERE pauli IS NOT NULL AND trim(coalesce(pauli::text,'')) <> ''
              AND (pauli::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT employee_id, 'attention_processing', 'tiki', (tiki::text)::numeric, 'higher_better'
            FROM tb_profiles_psych_norm
            WHERE tiki IS NOT NULL AND trim(coalesce(tiki::text,'')) <> ''
              AND (tiki::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'leadership_competencies', 'gdr', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'GDR' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'leadership_competencies', 'cex', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'CEX' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'leadership_competencies', 'ids', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'IDS' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'execution_competencies', 'qdd', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'QDD' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'execution_competencies', 'sto', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'STO' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'execution_competencies', 'ftc', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'FTC' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'interpersonal_skills', 'sea', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'SEA' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'interpersonal_skills', 'lie', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'LIE' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT c.employee_id, 'interpersonal_skills', 'csi', (c.score::text)::numeric, 'higher_better'
            FROM competencies_yearly c, tb_latest_year ly
            WHERE c.pillar_code = 'CSI' AND c.year = ly.max_year
              AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
              AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
            
            UNION ALL
            SELECT ps.employee_id, 'work_preferences', ('papi_' || lower(ps.scale_code))::text,
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
            """)
            
            # ===== 7) BASELINE PER TV =====
            cur.execute("""
            CREATE TABLE tb_baseline_per_tv AS
            SELECT
              st.job_vacancy_id,
              tv.tgv_name,
              tv.tv_name,
              tv.scoring_direction,
              PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY tv.tv_score) AS baseline_score,
              COUNT(DISTINCT tv.employee_id) AS benchmark_count
            FROM tb_selected_talents st
            JOIN tb_tv_scores_long tv ON tv.employee_id = st.employee_id
            GROUP BY st.job_vacancy_id, tv.tgv_name, tv.tv_name, tv.scoring_direction
            """)
            
            # ===== 8) EMPLOYEES WITH TV =====
            cur.execute("""
            CREATE TABLE tb_employees_with_tv AS
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
              b.baseline_score,
              v.weights_config,
              v.job_vacancy_id
            FROM employees e
            LEFT JOIN dim_directorates dir ON e.directorate_id = dir.directorate_id
            LEFT JOIN dim_positions pos ON e.position_id = pos.position_id
            LEFT JOIN dim_grades gr ON e.grade_id = gr.grade_id
            CROSS JOIN tb_vacancy v
            LEFT JOIN tb_tv_scores_long tv ON tv.employee_id = e.employee_id
            LEFT JOIN tb_baseline_per_tv b
              ON b.job_vacancy_id = v.job_vacancy_id
              AND b.tgv_name = tv.tgv_name
              AND b.tv_name = tv.tv_name
            WHERE tv.tv_score IS NOT NULL
            """)
            
            # ===== 9) TV MATCH CALC =====
            cur.execute("""
            CREATE TABLE tb_tv_match_calc AS
            SELECT
              e.*,
              CASE
                WHEN e.baseline_score IS NULL THEN NULL
                WHEN e.scoring_direction = 'boolean' THEN
                  CASE WHEN e.user_score = e.baseline_score THEN 100.0 ELSE 0.0 END
                WHEN e.scoring_direction = 'lower_better' THEN
                  CASE
                    WHEN e.baseline_score = 0 OR e.user_score IS NULL THEN NULL
                    ELSE LEAST(GREATEST(((2.0 * e.baseline_score - e.user_score) / NULLIF(e.baseline_score,0)) * 100.0, 0.0), 100.0)
                  END
                ELSE
                  CASE
                    WHEN e.baseline_score = 0 OR e.user_score IS NULL THEN NULL
                    ELSE LEAST(GREATEST((e.user_score / NULLIF(e.baseline_score,0)) * 100.0, 0.0), 100.0)
                  END
              END AS tv_match_rate
            FROM tb_employees_with_tv e
            """)
            
            # ===== 10) TV MATCH WITH WEIGHTS =====
            cur.execute("""
            CREATE TABLE tb_tv_match_with_weights AS
            SELECT
              t.*,
              CASE
                WHEN t.weights_config IS NULL THEN 1.0
                WHEN (t.weights_config #>> ARRAY['TV_weights', t.tgv_name, t.tv_name]) ~ '^\s*[-+]?\d+(\.\d+)?\s*$'
                  THEN (t.weights_config #>> ARRAY['TV_weights', t.tgv_name, t.tv_name])::numeric
                ELSE 1.0
              END AS tv_weight
            FROM tb_tv_match_calc t
            """)
            
            # ===== 11) TGV AGGREGATION =====
            cur.execute("""
            CREATE TABLE tb_tgv_aggregation AS
            SELECT
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
            FROM tb_tv_match_with_weights
            GROUP BY employee_id, fullname, directorate, role, grade, tgv_name
            """)
            
            # ===== 12) TGV WITH WEIGHTS =====
            cur.execute("""
            CREATE TABLE tb_tgv_with_weights AS
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
            FROM tb_tgv_aggregation t
            """)
            
            # ===== 13) FINAL AGGREGATION =====
            cur.execute("""
            CREATE TABLE tb_final_aggregation AS
            SELECT
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
            FROM tb_tgv_with_weights
            GROUP BY employee_id, fullname, directorate, role, grade
            """)
            
            conn.commit()
            
            # ===== 14) FINAL SELECT - RETURN DETAILED RESULTS =====
            final_sql = """
            SELECT
              tm.employee_id,
              tm.fullname,
              tm.directorate,
              tm.role,
              tm.grade,
              tm.tgv_name,
              tm.tv_name,
              ROUND(tm.baseline_score::numeric, 2) AS baseline_score,
              ROUND(tm.user_score::numeric, 2) AS user_score,
              ROUND(tm.tv_match_rate::numeric, 2) AS tv_match_rate,
              ROUND(tgv.tgv_match_rate::numeric, 4) AS tgv_match_rate,
              ROUND(fa.final_match_rate::numeric, 4) AS final_match_rate
            FROM tb_tv_match_calc tm
            LEFT JOIN tb_tgv_with_weights tgv
              ON tgv.employee_id = tm.employee_id 
             AND tgv.tgv_name = tm.tgv_name
            LEFT JOIN tb_final_aggregation fa 
              ON fa.employee_id = tm.employee_id
            WHERE tm.baseline_score IS NOT NULL
            ORDER BY fa.final_match_rate DESC NULLS LAST, 
                     tm.employee_id, tm.tgv_name, tm.tv_name
            """
            
            df = pd.read_sql_query(final_sql, conn)
            return df
            
        except Exception as e:
            conn.rollback()
            st.error(f"❌ SQL Pipeline Error: {str(e)}")
            raise e

    # ---------------------------------------------
    # SUMMARY RESULTS
    # ---------------------------------------------
    def get_summary_results(self, job_vacancy_id: int, limit: int = 50) -> pd.DataFrame:
        query = f"""
        SELECT
          fa.employee_id,
          fa.fullname,
          fa.directorate,
          fa.role,
          fa.grade,
          ROUND(fa.final_match_rate::numeric, 2) AS final_match_rate_percentage,
          fa.tgv_count_with_baseline
        FROM tb_final_aggregation fa
        WHERE fa.final_match_rate IS NOT NULL
        ORDER BY fa.final_match_rate DESC
        LIMIT %s
        """
        return self.execute_query(query, params=(limit,))

    def get_vacancy_info(self, job_vacancy_id: int) -> Dict:
        query = "SELECT * FROM talent_benchmarks WHERE job_vacancy_id = %s"
        result = self.execute_query(query, params=(job_vacancy_id,))
        return result.iloc[0].to_dict() if not result.empty else {}
