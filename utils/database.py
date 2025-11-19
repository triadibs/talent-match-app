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
        Attempts to run the final select of the matching pipeline.
        If the pipeline temporary tables are not present (or SQL fails),
        falls back to get_summary_results to avoid crashing the app.
        """
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

        try:
            with self.get_connection() as conn:
                # If you have a multi-step SQL pipeline, it should be executed
                # here in the same connection before final_sql (commented out
                # in your current code). If not implemented, final_sql may fail.
                df = pd.read_sql_query(final_sql, conn)
                return df
        except Exception as e:
            # Log the exception to Streamlit so developer can inspect logs
            st.warning(f"Matching pipeline final select failed: {e}. Falling back to summary results.")
            # Fallback: return summary results which are likely pre-aggregated
            return self.get_summary_results(job_vacancy_id, limit=500)

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
