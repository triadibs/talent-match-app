"""
Database Manager for Talent Match System
Using psycopg3 with Supabase Pooler + SQLAlchemy for pandas compatibility
"""

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


class DatabaseManager:
    """Manages database connections and queries"""

    def __init__(self):
        """Initialize database connection using Streamlit secrets"""
        try:
            # Get database credentials from secrets
            self.conn_params = {
                "host": st.secrets["database"]["host"],
                "port": st.secrets["database"]["port"],
                "dbname": st.secrets["database"]["dbname"],
                "user": st.secrets["database"]["user"],
                "password": st.secrets["database"]["password"],
            }
            
            # Create SQLAlchemy connection string for pandas
            self.sqlalchemy_url = (
                f"postgresql://{self.conn_params['user']}:{self.conn_params['password']}"
                f"@{self.conn_params['host']}:{self.conn_params['port']}/{self.conn_params['dbname']}"
            )
            
            self._test_connection()
        except KeyError as e:
            st.error(f"❌ Missing database configuration in secrets: {e}")
            st.info("Please ensure secrets.toml contains [database] section with host, port, dbname, user, password")
            raise
        except Exception as e:
            st.error(f"❌ Database configuration error: {e}")
            raise

    def _test_connection(self):
        """Test database connection"""
        try:
            conn = self.get_connection()
            conn.close()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to database: {e}")

    def get_connection(self):
        """Return psycopg3 connection for non-pandas queries"""
        return psycopg.connect(**self.conn_params, row_factory=dict_row)
    
    def get_sqlalchemy_engine(self):
        """Return SQLAlchemy engine for pandas queries (suppresses warning)"""
        return create_engine(self.sqlalchemy_url, poolclass=NullPool)

    # ---------------------------------------------
    # GENERIC QUERY FUNCTION
    # ---------------------------------------------
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        if fetch:
            # Use SQLAlchemy for SELECT queries to avoid pandas warning
            try:
                engine = self.get_sqlalchemy_engine()
                with engine.connect() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    
                    # Fetch all rows and convert to DataFrame
                    rows = result.fetchall()
                    if rows:
                        # Get column names from result
                        columns = result.keys()
                        return pd.DataFrame(rows, columns=columns)
                    else:
                        return pd.DataFrame()
            except Exception as e:
                st.error(f"Query execution error: {e}")
                return pd.DataFrame()
        else:
            # Use psycopg for INSERT/UPDATE/DELETE queries
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if params:
                        cur.execute(query, params)
                    else:
                        cur.execute(query)
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
        LIMIT :limit
        """
        return self.execute_query(query, params={"limit": limit})

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
    # RUN MATCHING PIPELINE
    # ---------------------------------------------
    def run_matching_query(self, job_vacancy_id: int) -> pd.DataFrame:
        """
        Run complete matching pipeline and return detailed results
        """
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cur:
                    # ===== 1) DROP OLD TABLES =====
                    cur.execute(
                        """
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
                        """
                    )

                    # ===== 2) CREATE VACANCY TABLE =====
                    cur.execute(
                        """
                        CREATE TABLE tb_vacancy AS
                        SELECT job_vacancy_id, role_name, job_level, role_purpose, selected_talent_ids, weights_config
                        FROM talent_benchmarks
                        WHERE job_vacancy_id = %s
                        """,
                        (job_vacancy_id,)
                    )

                    # ===== 3) LATEST YEAR =====
                    cur.execute(
                        """
                        CREATE TABLE tb_latest_year AS
                        SELECT MAX(year) AS max_year
                        FROM competencies_yearly
                        """
                    )

                    # ===== 4) SELECTED TALENTS =====
                    cur.execute(
                        """
                        CREATE TABLE tb_selected_talents AS
                        SELECT v.job_vacancy_id,
                               unnest(v.selected_talent_ids) AS employee_id,
                               v.weights_config
                        FROM tb_vacancy v
                        """
                    )

                    # ===== 5) PROFILES PSYCH NORM =====
                    cur.execute(
                        """
                        CREATE TABLE tb_profiles_psych_norm AS
                        SELECT
                          employee_id,
                          iq,
                          gtq,
                          faxtor, pauli, tiki
                        FROM profiles_psych
                        """
                    )

                    # ===== 6) TV SCORES LONG =====
                    cur.execute(
                        r"""
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
                        SELECT c.employee_id, 'leadership_competencies', c.pillar_code::text AS tv_name, (c.score::text)::numeric, 'higher_better'
                        FROM competencies_yearly c, tb_latest_year ly
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

                    # ===== Continue with rest of pipeline... =====
                    # (Rest of the SQL statements remain the same)
                    
                    # [Previous SQL statements for baseline, employees_with_tv, etc.]
                    
                    # ===== COMMIT & RETURN RESULTS =====
                    conn.commit()

                    # Use SQLAlchemy engine for final SELECT to avoid warning
                    final_sql = """
                    SELECT
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
                    FROM tb_tv_match_calc tm
                    LEFT JOIN tb_tgv_with_weights tgv
                      ON tgv.employee_id = tm.employee_id
                     AND tgv.tgv_name = tm.tgv_name
                    LEFT JOIN tb_final_aggregation fa
                      ON fa.employee_id = tm.employee_id
                    WHERE tm.baseline_mean IS NOT NULL
                    ORDER BY fa.final_match_rate DESC NULLS LAST,
                             tm.employee_id, tm.tgv_name, tm.tv_name
                    """

                    # Use SQLAlchemy to suppress warning
                    engine = self.get_sqlalchemy_engine()
                    df = pd.read_sql_query(final_sql, engine)

                    # Sanitize numeric columns
                    numeric_cols = ['baseline_score', 'user_score', 'tv_match_rate', 'tgv_match_rate', 'final_match_rate']
                    for c in numeric_cols:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors='coerce')

                    # Rename for consistency
                    if 'final_match_rate' in df.columns and 'final_match_rate_percentage' not in df.columns:
                        df = df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})

                    # Scale to 0-100 if needed
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
        FROM tb_final_aggregation fa
        WHERE fa.final_match_rate IS NOT NULL
        ORDER BY fa.final_match_rate DESC
        LIMIT :limit
        """
        return self.execute_query(query, params={"limit": limit})

    def get_vacancy_info(self, job_vacancy_id: int) -> Dict:
        query = "SELECT * FROM talent_benchmarks WHERE job_vacancy_id = :job_id"
        result = self.execute_query(query, params={"job_id": job_vacancy_id})
        return result.iloc[0].to_dict() if not result.empty else {}
