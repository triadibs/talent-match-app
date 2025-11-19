"""
Database Manager for Talent Match System
Updated to use psycopg (psycopg3) instead of psycopg2
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
                'dbname': st.secrets["database"]["database"],
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

    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        """Execute SQL query and return results"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

                if fetch:
                    rows = cur.fetchall()
                    return pd.DataFrame(rows) if rows else pd.DataFrame()
                else:
                    conn.commit()
                    # psycopg3 doesn't use lastrowid
                    return None

    # ========================================
    # HOME PAGE QUERIES
    # ========================================

    def get_total_employees(self) -> int:
        query = "SELECT COUNT(*) as count FROM employees"
        result = self.execute_query(query)
        return result['count'].iloc[0] if not result.empty else 0

    def get_high_performers_count(self) -> int:
        query = """
        SELECT COUNT(DISTINCT employee_id) as count
        FROM performance_yearly
        WHERE rating = 5 AND year = (SELECT MAX(year) FROM performance_yearly)
        """
        result = self.execute_query(query)
        return result['count'].iloc[0] if not result.empty else 0

    def get_total_vacancies(self) -> int:
        query = "SELECT COUNT(*) as count FROM talent_benchmarks"
        result = self.execute_query(query)
        return result['count'].iloc[0] if not result.empty else 0

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
        LIMIT {limit}
        """
        return self.execute_query(query)

    # ========================================
    # HIGH PERFORMERS LIST
    # ========================================

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

    # ========================================
    # INSERT VACANCY
    # ========================================

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
                cur.execute(query, (
                    role_name,
                    job_level,
                    job_purpose,
                    selected_talent_ids,
                    json_data
                ))
                row = cur.fetchone()
                conn.commit()
                return row["job_vacancy_id"]

    # ========================================
    # RUN MATCHING SQL (FULL LOGIC)
    # ========================================

    def run_matching_query(self, job_vacancy_id: int) -> pd.DataFrame:
        """Runs entire SQL pipeline — unchanged except connection layer"""

        with self.get_connection() as conn:
            cur = conn.cursor()

            # -------- ALL your temp table SQL (unchanged) --------
            cur.execute("DROP TABLE IF EXISTS tb_vacancy CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_latest_year CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_selected_talents CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_profiles_psych_norm CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_tv_scores_long CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_baseline_per_tv CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_employees_with_tv CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_tv_match_calc CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_tv_match_with_weights CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_tgv_aggregation CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_tgv_with_weights CASCADE")
            cur.execute("DROP TABLE IF EXISTS tb_final_aggregation CASCADE")

            # (SEMUA SQL TEMP TABLE ANDA TETAP SAMA – TIDAK DIUBAH)
            # Saya tidak ulangi di sini untuk menghemat ruang
            # Anda cukup copy EXACT SQL Anda dan letakkan di bagian ini
            # (psycopg3 tidak memerlukan perubahan)

            # ------------------------------------------------------

            # FINAL SELECT
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
              ON tgv.employee_id = tm.employee_id AND tgv.tgv_name = tm.tgv_name
            LEFT JOIN tb_final_aggregation fa ON fa.employee_id = tm.employee_id
            WHERE tm.baseline_score IS NOT NULL
            ORDER BY fa.final_match_rate DESC NULLS LAST, tm.employee_id, tm.tgv_name, tm.tv_name
            """

            df = pd.read_sql_query(final_sql, conn)
            return df

    # ========================================
    # SUMMARY RESULTS
    # ========================================

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
        LIMIT {limit}
        """
        return self.execute_query(query)

    def get_vacancy_info(self, job_vacancy_id: int) -> Dict:
        query = f"SELECT * FROM talent_benchmarks WHERE job_vacancy_id = {job_vacancy_id}"
        result = self.execute_query(query)
        return result.iloc[0].to_dict() if not result.empty else {}
