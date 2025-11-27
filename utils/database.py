# -*- coding: utf-8 -*-
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
    # RUN MATCHING PIPELINE (FINAL FIXED VERSION)
    # ---------------------------------------------
    def run_matching_query(self, vacancy_id: int) -> pd.DataFrame:
        """
        Full SQL pipeline for Talent Match Step 2.
        Returns detailed TGV + TV rows for analytics.
        This version is robust to employee_id format differences between tables.
        """

        sql = """
WITH vacancy AS (
    SELECT 
        job_vacancy_id,
        role_name,
        job_level,
        role_purpose,
        selected_talent_ids,
        weights_config
    FROM talent_benchmarks
    WHERE job_vacancy_id = %(vacancy_id)s
),

selected_benchmarks AS (
    SELECT 
        v.job_vacancy_id,
        (jsonb_array_elements_text(v.selected_talent_ids))::text AS selected_id,
        -- normalized variant with EMP prefix if missing
        CASE
            WHEN (jsonb_array_elements_text(v.selected_talent_ids))::text LIKE 'EMP%' THEN (jsonb_array_elements_text(v.selected_talent_ids))::text
            ELSE CONCAT('EMP', (jsonb_array_elements_text(v.selected_talent_ids))::text)
        END AS selected_id_norm
    FROM vacancy v
),

tv_raw AS (
    SELECT
        -- raw employee id from employees (may be numeric or already prefixed)
        e.employee_id AS emp_id_raw,
        e.fullname,
        d.name AS directorate,
        p.name AS role,
        g.name AS grade,
        tv.employee_id AS tv_emp_id,       -- from tb_tv_scores_long (has EMP... format)
        tv.tgv_name,
        tv.tv_name,
        tv.tv_score,
        tv.scoring_direction
    FROM employees e
    LEFT JOIN dim_directorates d ON d.directorate_id = e.directorate_id
    LEFT JOIN dim_positions p ON p.position_id = e.position_id
    LEFT JOIN dim_grades g ON g.grade_id = e.grade_id
    -- robust join: match tv.employee_id to either e.employee_id (if already prefixed) OR 'EMP' || e.employee_id
    LEFT JOIN tb_tv_scores_long tv
      ON (
           tv.employee_id = e.employee_id
           OR tv.employee_id = CONCAT('EMP', e.employee_id::text)
         )
),

baseline AS (
    SELECT
        tv.tv_name,
        tv.tgv_name,
        AVG(tv.tv_score) AS baseline_score,
        STDDEV_POP(tv.tv_score) AS baseline_std
    FROM tv_raw tv
    JOIN selected_benchmarks sb
      ON (tv.tv_emp_id IS NOT NULL
          AND (tv.tv_emp_id = sb.selected_id OR tv.tv_emp_id = sb.selected_id_norm))
    GROUP BY tv.tv_name, tv.tgv_name
),

tv_match AS (
    SELECT
        -- use tv_emp_id when available (EMP-prefixed), otherwise build from emp_id_raw
        COALESCE(tv.tv_emp_id, CONCAT('EMP', tv.emp_id_raw::text)) AS employee_id,
        tv.fullname,
        tv.directorate,
        tv.role,
        tv.grade,
        tv.tgv_name,
        tv.tv_name,
        tv.tv_score AS user_score,
        b.baseline_score,
        b.baseline_std,
        tv.scoring_direction,
        CASE
            WHEN b.baseline_score IS NULL THEN NULL
            WHEN LOWER(tv.scoring_direction) LIKE '%lower%' OR tv.scoring_direction = 'lower-is-better'
                THEN LEAST(
                    ((2 * b.baseline_score - tv.tv_score) / NULLIF(b.baseline_score,0)) * 100,
                    100
                )
            ELSE LEAST(
                (tv.tv_score / NULLIF(b.baseline_score,0)) * 100,
                100
            )
        END AS tv_match_rate
    FROM tv_raw tv
    LEFT JOIN baseline b ON b.tv_name = tv.tv_name AND b.tgv_name = tv.tgv_name
),

tgv_match AS (
    SELECT
        employee_id,
        fullname,
        directorate,
        role,
        grade,
        tgv_name,
        AVG(tv_match_rate) AS tgv_match_rate
    FROM tv_match
    GROUP BY 
        employee_id, fullname, directorate, role, grade, tgv_name
),

final_match AS (
    SELECT
        employee_id,
        fullname,
        directorate,
        role,
        grade,
        AVG(tgv_match_rate) AS final_match_rate
    FROM tgv_match
    GROUP BY employee_id, fullname, directorate, role, grade
)

SELECT
    tv.employee_id,
    tv.fullname,
    tv.directorate,
    tv.role,
    tv.grade,
    tv.tgv_name,
    tv.tv_name,
    tv.user_score,
    tv.baseline_score,
    tv.tv_match_rate,
    tgv.tgv_match_rate,
    fm.final_match_rate
FROM tv_match tv
LEFT JOIN tgv_match tgv
    ON tgv.employee_id = tv.employee_id
    AND tgv.tgv_name = tv.tgv_name
LEFT JOIN final_match fm
    ON fm.employee_id = tv.employee_id

ORDER BY tv.employee_id, tv.tgv_name, tv.tv_name;
"""

        try:
            # Use read_sql to get a DataFrame. Pass connection object from get_connection()
            with self.get_connection() as conn:
                df = pd.read_sql(sql, conn, params={'vacancy_id': vacancy_id})
            # Normalize column names & numeric types for downstream code
            if 'final_match_rate' in df.columns and 'final_match_rate_percentage' not in df.columns:
                df = df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
            # Ensure percent scale (0..1 -> 0..100)
            if 'final_match_rate_percentage' in df.columns:
                df['final_match_rate_percentage'] = pd.to_numeric(df['final_match_rate_percentage'], errors='coerce')
                maxv = df['final_match_rate_percentage'].max(skipna=True)
                if pd.notna(maxv) and maxv <= 1.0:
                    df['final_match_rate_percentage'] = df['final_match_rate_percentage'] * 100.0
            # Coerce tv_match_rate and tgv_match_rate to numeric
            for c in ['tv_match_rate', 'tgv_match_rate', 'user_score', 'baseline_score']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
        except Exception as e:
            # print to logs (Streamlit will show exception in app logs)
            print("SQL ERROR in run_matching_query:", e)
            return pd.DataFrame()

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
        LIMIT %s
        """
        return self.execute_query(query, params=(limit,))

    def get_vacancy_info(self, job_vacancy_id: int) -> Dict:
        query = "SELECT * FROM talent_benchmarks WHERE job_vacancy_id = %s"
        result = self.execute_query(query, params=(job_vacancy_id,))
        return result.iloc[0].to_dict() if not result.empty else {}
