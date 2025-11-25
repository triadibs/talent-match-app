# app.py
# -*- coding: utf-8 -*-
"""
Talent Match Intelligence System - Step 3 (Revised, hardened)
Includes cleaning for placeholder/header-rows and robust handling when scores are missing.
Author: TRI ADI BASKORO (revised)
Date: 18 November 2025 (revised)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple

# Custom modules (must exist)
from utils.database import DatabaseManager
from utils.llm import generate_job_profile
from utils.visualizations import (
    plot_match_distribution,
    plot_top_candidates,
    plot_tgv_radar,
    plot_tv_heatmap,
    plot_strengths_gaps
)

# ----------------------------
# Helpers: cleaning + normalization
# ----------------------------
def _ensure_str_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Ensure listed columns are strings but preserve NaN as real NaN."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(object).where(~df[c].isna(), other=np.nan)
            df[c] = df[c].apply(lambda x: x if pd.isna(x) or isinstance(x, str) else str(x))
            # convert common literal tokens to NaN (defensive)
            df[c] = df[c].replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NaN': np.nan})
    return df

def _drop_placeholder_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where many fields are equal to their column names (common when header repeated),
    or where nearly all value cells are literal placeholders like 'employee_id', 'fullname', etc.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    # Replace literal tokens with NaN strings first to make comparison consistent
    df = df.replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NaN': np.nan})
    # build boolean mask: True if row looks like a header/placeholder row
    def _is_placeholder_row(row):
        # count how many cells equal to their column name (case-insensitive)
        match_count = 0
        total_checked = 0
        for col, val in row.items():
            # only check string-like values
            if pd.isna(val):
                continue
            total_checked += 1
            try:
                if str(val).strip().lower() == str(col).strip().lower():
                    match_count += 1
            except Exception:
                pass
        # if majority of checked cells equal their column names -> placeholder
        if total_checked == 0:
            return False
        return (match_count / total_checked) >= 0.6  # threshold: 60%
    mask = df.apply(_is_placeholder_row, axis=1)
    if mask.any():
        df = df.loc[~mask].reset_index(drop=True)
    return df

def _normalize_results_dfs(detailed: pd.DataFrame, summary: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize and clean the detailed and summary DataFrames:
      - drop placeholder rows
      - ensure key text columns are strings (employee_id, fullname, tgv_name, tv_name, role, grade, directorate)
      - coerce numeric final match column to percent (0..100)
      - return tuple (detailed_norm, summary_norm)
    """
    if detailed is None:
        detailed = pd.DataFrame()

    # Drop obvious placeholder/header rows (e.g., rows where values == column names)
    detailed = _drop_placeholder_rows(detailed)

    # Standard text columns to protect .str usage later
    text_cols = ['employee_id', 'fullname', 'tgv_name', 'tv_name', 'role', 'grade', 'directorate', 'scale_code']
    detailed = _ensure_str_cols(detailed, text_cols)

    # Normalize employee_id as string
    if 'employee_id' in detailed.columns:
        detailed['employee_id'] = detailed['employee_id'].astype(object).where(~detailed['employee_id'].isna(), other=np.nan)
        detailed['employee_id'] = detailed['employee_id'].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)

    # Normalize final match rate column names and scale
    if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
        detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})

    if 'final_match_rate_percentage' in detailed.columns:
        # coerce to numeric and convert 0..1 -> 0..100 if needed
        detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
        maxv = detailed['final_match_rate_percentage'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0:
            detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

    # Build summary from detailed if summary not provided
    if summary is None or (isinstance(summary, pd.DataFrame) and summary.empty):
        if not detailed.empty and 'employee_id' in detailed.columns:
            cols_for_summary = ['employee_id', 'fullname', 'directorate', 'role', 'grade', 'final_match_rate_percentage']
            present = [c for c in cols_for_summary if c in detailed.columns]
            if present:
                summary = detailed[present].groupby('employee_id', as_index=False).first()
            else:
                summary = pd.DataFrame()
        else:
            summary = pd.DataFrame()
    else:
        # if summary provided, also drop placeholder rows and normalize
        summary = _drop_placeholder_rows(summary)
        summary = _ensure_str_cols(summary, ['employee_id', 'fullname', 'role', 'grade', 'directorate'])
        if 'employee_id' in summary.columns:
            summary['employee_id'] = summary['employee_id'].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
        if 'final_match_rate_percentage' in summary.columns:
            summary['final_match_rate_percentage'] = pd.to_numeric(summary['final_match_rate_percentage'], errors='coerce')
            maxv = summary['final_match_rate_percentage'].max(skipna=True)
            if pd.notna(maxv) and maxv <= 1.0:
                summary['final_match_rate_percentage'] = summary['final_match_rate_percentage'] * 100.0

    # final defensive conversions for tgv/tv names
    detailed = _ensure_str_cols(detailed, ['tgv_name', 'tv_name'])
    return detailed.reset_index(drop=True), (summary.reset_index(drop=True) if isinstance(summary, pd.DataFrame) else pd.DataFrame())

# ----------------------------
# Init DB
# ----------------------------
@st.cache_resource
def init_database():
    return DatabaseManager()

db = init_database()

# ----------------------------
# Session defaults
# ----------------------------
if 'vacancy_created' not in st.session_state:
    st.session_state.vacancy_created = False
if 'job_vacancy_id' not in st.session_state:
    st.session_state.job_vacancy_id = None
if 'matching_results_summary' not in st.session_state:
    st.session_state.matching_results_summary = None
if 'matching_results_detailed' not in st.session_state:
    st.session_state.matching_results_detailed = None
if 'job_profile' not in st.session_state:
    st.session_state.job_profile = None

# ----------------------------
# Pages
# ----------------------------
def show_home_page():
    col1, col2, col3 = st.columns(3)
    try:
        total_employees = db.get_total_employees()
    except Exception:
        total_employees = 0
    try:
        high_performers = db.get_high_performers_count()
    except Exception:
        high_performers = 0
    try:
        total_vacancies = db.get_total_vacancies()
    except Exception:
        total_vacancies = 0

    with col1:
        st.metric("Total Employees", f"{total_employees:,}")
    with col2:
        st.metric("High Performers (Rating 5)", f"{high_performers:,}")
    with col3:
        st.metric("Active Vacancies", f"{total_vacancies:,}")

    st.markdown("---")
    st.markdown("## 🔍 How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1️⃣ Define Success")
        st.write("Select high-performing employees (rating 5) as benchmarks for the role.")
    with c2:
        st.markdown("### 2️⃣ AI Analysis")
        st.write("System analyzes 7 talent dimensions across 50+ variables to find patterns.")
    with c3:
        st.markdown("### 3️⃣ Match & Rank")
        st.write("All employees are scored against the benchmark profile and ranked.")
    st.markdown("---")

    st.markdown("## 📋 Recent Vacancies")
    try:
        recent_vacancies = db.get_recent_vacancies(limit=5)
    except Exception:
        recent_vacancies = pd.DataFrame()

    if isinstance(recent_vacancies, pd.DataFrame) and not recent_vacancies.empty:
        try:
            st.dataframe(recent_vacancies, width='stretch')
        except Exception:
            st.dataframe(recent_vacancies.head(10), width='stretch')
    else:
        st.info("No vacancies created yet. Go to 'Create Vacancy' to get started!")

def generate_fallback_profile(role_name: str, job_level: str, role_purpose: str) -> dict:
    return {
        "requirements": f"Experience in {role_name}, level: {job_level}. {role_purpose}",
        "description": f"{role_name} — {job_level}. {role_purpose}",
        "competencies": ["Analytical thinking", "Problem solving", "Communication", "Domain knowledge"]
    }

def show_create_vacancy_page():
    st.markdown("## ➕ Create New Job Vacancy")
    st.markdown("Define the role and select benchmark employees to find the best matches.")
    with st.form("vacancy_form"):
        st.markdown("### 📝 Job Details")
        col1, col2 = st.columns(2)
        with col1:
            role_name = st.text_input("Role Name *", placeholder="e.g., Data Analyst, Product Manager")
            job_level = st.selectbox("Job Level *", ["Entry", "Junior", "Middle", "Senior", "Lead", "Manager", "Director"], index=2)
        with col2:
            high_performers_df = db.get_high_performers()
            if not isinstance(high_performers_df, pd.DataFrame) or high_performers_df.empty:
                st.error("No high performers found!")
                selected_employees = []
            else:
                high_performers_df = _ensure_str_cols(high_performers_df, ['employee_id', 'fullname', 'position'])
                id_list = high_performers_df['employee_id'].astype(str).tolist()
                id_to_label = {str(row['employee_id']): f"{row['employee_id']} - {row['fullname']} ({row.get('position','')})" for _, row in high_performers_df.iterrows()}
                selected_ids = st.multiselect("Select Benchmark Employees *", options=id_list,
                                              format_func=lambda eid: id_to_label.get(str(eid), str(eid)),
                                              help="Pilih minimal 2 benchmark")
                selected_employees = [str(x) for x in selected_ids]
        role_purpose = st.text_area("Role Purpose *", placeholder="1-2 sentence summary...", height=100)
        submitted = st.form_submit_button("🚀 Create Vacancy & Run Matching", type="primary")

        if submitted:
            if not role_name or not role_purpose or len(selected_employees) < 2:
                st.error("❌ Please fill all required fields and select at least 2 benchmarks")
                st.stop()

            with st.spinner("🔄 Creating vacancy..."):
                try:
                    vacancy_id = db.insert_vacancy(
                        role_name=role_name,
                        job_level=job_level,
                        role_purpose=role_purpose,
                        selected_talent_ids=selected_employees,
                        weights_config=None
                    )
                    st.success(f"✅ Vacancy #{vacancy_id} created!")
                except Exception as e:
                    st.error(f"❌ Error creating vacancy: {e}")
                    st.stop()

            with st.spinner("🤖 Generating job profile with AI..."):
                try:
                    benchmarks_df = high_performers_df[high_performers_df['employee_id'].isin(selected_employees)]
                    job_profile = generate_job_profile(
                        role_name=role_name,
                        job_level=job_level,
                        role_purpose=role_purpose,
                        benchmark_employees=benchmarks_df
                    )
                except Exception as e:
                    st.warning(f"AI generation failed: {e}. Using fallback.")
                    job_profile = generate_fallback_profile(role_name, job_level, role_purpose)

            with st.spinner("📊 Computing talent matches (this may take 30-60 seconds)..."):
                try:
                    matching_results = db.run_matching_query(vacancy_id)
                    if not isinstance(matching_results, pd.DataFrame) or matching_results.empty:
                        st.error("❌ Matching returned no results. Check your data or pipeline logs.")
                        st.stop()
                    st.success(f"✅ Matched {len(matching_results)} records!")
                except Exception as e:
                    st.error(f"❌ Matching error: {e}")
                    st.exception(e)
                    st.stop()

            detailed = matching_results.copy()
            # normalize final_match_rate field
            if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
                detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
            if 'final_match_rate_percentage' in detailed.columns:
                detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
                maxv = detailed['final_match_rate_percentage'].max(skipna=True)
                if pd.notna(maxv) and maxv <= 1.0:
                    detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

            # Clean placeholder/header rows before building summary
            detailed = _drop_placeholder_rows(detailed)
            summary_df = pd.DataFrame()
            if not detailed.empty and 'employee_id' in detailed.columns:
                # build summary defensively
                cols_for_summary = ['employee_id', 'fullname', 'directorate', 'role', 'grade', 'final_match_rate_percentage']
                present = [c for c in cols_for_summary if c in detailed.columns]
                if present:
                    summary_df = detailed[present].groupby('employee_id').first().reset_index()
            # ensure numeric
            if 'final_match_rate_percentage' in summary_df.columns:
                summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df['final_match_rate_percentage'], errors='coerce')

            st.session_state.vacancy_created = True
            st.session_state.job_vacancy_id = vacancy_id
            st.session_state.matching_results_detailed = detailed
            st.session_state.matching_results_summary = summary_df
            st.session_state.job_profile = job_profile

            st.balloons()
            st.markdown("---")
            st.markdown("### 🎯 AI-Generated Job Profile")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**📋 Job Requirements:**")
                st.write(job_profile.get('requirements', 'N/A'))
                st.markdown("**📝 Job Description:**")
                st.write(job_profile.get('description', 'N/A'))
            with col2:
                st.markdown("**🎯 Key Competencies:**")
                competencies = job_profile.get('competencies', [])
                if isinstance(competencies, list):
                    for comp in competencies:
                        st.markdown(f"• {comp}")
                else:
                    st.write(competencies)
            st.markdown("---")
            st.info("👉 Go to **'View Results'** or **'Analytics'** to see detailed insights!")

def show_results_page():
    st.markdown("## 📊 Talent Matching Results")

    if not st.session_state.get('vacancy_created', False):
        st.warning("⚠️ No vacancy created yet. Create one or load existing.")
        existing = db.get_recent_vacancies(20)
        if not isinstance(existing, pd.DataFrame) or existing.empty:
            st.info("No vacancies found in DB (recent_vacancies is empty).")
            return

        selected = st.selectbox(
            "Select vacancy:",
            options=existing['job_vacancy_id'].tolist(),
            format_func=lambda x: f"ID {x}: {existing[existing['job_vacancy_id']==x]['role_name'].iloc[0]}"
        )

        if st.button("📥 Load Results"):
            with st.spinner("Loading..."):
                try:
                    vac_id = int(selected) if isinstance(selected, (str,)) and str(selected).isdigit() else selected
                    try:
                        summary_df = db.get_summary_results(vac_id, limit=5000)
                    except Exception:
                        summary_df = pd.DataFrame()

                    detailed = pd.DataFrame()
                    if summary_df is None or (isinstance(summary_df, pd.DataFrame) and summary_df.empty):
                        detailed = db.run_matching_query(vac_id)
                        if not isinstance(detailed, pd.DataFrame) or detailed.empty:
                            st.error("Matching returned no detailed rows. Pipeline may have failed.")
                            return
                        if 'final_match_rate' in detailed.columns:
                            detailed['final_match_rate'] = pd.to_numeric(detailed['final_match_rate'], errors='coerce')
                        if 'final_match_rate_percentage' not in detailed.columns and 'final_match_rate' in detailed.columns:
                            detailed['final_match_rate_percentage'] = detailed['final_match_rate']
                        tmp = detailed.copy()
                        if 'final_match_rate_percentage' not in tmp.columns:
                            for c in ['final_match_rate', 'tgv_match_rate', 'tv_match_rate']:
                                if c in tmp.columns and pd.api.types.is_numeric_dtype(tmp[c]):
                                    tmp['final_match_rate_percentage'] = tmp[c]
                                    break
                        if 'final_match_rate_percentage' in tmp.columns and tmp['final_match_rate_percentage'].max(skipna=True) <= 1.0:
                            tmp['final_match_rate_percentage'] = tmp['final_match_rate_percentage'] * 100.0

                        summary_df = tmp.groupby('employee_id').agg({
                            'fullname': 'first',
                            'directorate': 'first',
                            'role': 'first',
                            'grade': 'first',
                            'final_match_rate_percentage': 'first'
                        }).reset_index()
                    else:
                        try:
                            detailed = db.run_matching_query(vac_id)
                        except Exception:
                            detailed = pd.DataFrame()

                    # final cleaning
                    detailed = _drop_placeholder_rows(detailed)
                    if 'final_match_rate_percentage' in summary_df.columns:
                        summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df['final_match_rate_percentage'], errors='coerce')
                        maxv = summary_df['final_match_rate_percentage'].max(skipna=True)
                        if pd.notna(maxv) and maxv <= 1.0:
                            summary_df['final_match_rate_percentage'] = summary_df['final_match_rate_percentage'] * 100.0

                    st.session_state.job_vacancy_id = vac_id
                    st.session_state.matching_results_summary = summary_df.copy()
                    st.session_state.matching_results_detailed = detailed.copy() if isinstance(detailed, pd.DataFrame) else pd.DataFrame()
                    st.session_state.vacancy_created = True

                    st.success(f"Loaded vacancy {vac_id} — {len(summary_df)} summary rows, detailed rows: {len(st.session_state.matching_results_detailed)}")
                    return

                except Exception as e:
                    st.exception(e)

        return

    summary_df = st.session_state.matching_results_summary
    vacancy_id = st.session_state.job_vacancy_id

    if summary_df is None or (isinstance(summary_df, pd.DataFrame) and summary_df.empty):
        st.error("matching_results_summary is None or empty in session_state.")
        return

    df = summary_df.copy()
    if 'final_match_rate' in df.columns and 'final_match_rate_percentage' not in df.columns:
        df = df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
    df['final_match_rate_percentage'] = pd.to_numeric(df.get('final_match_rate_percentage'), errors='coerce')

    if df['final_match_rate_percentage'].dropna().empty:
        st.warning("All final match scores are missing (NaN). Showing raw detailed results for inspection.")
        try:
            raw = db.execute_query("SELECT employee_id, fullname, directorate, role, grade, final_match_rate FROM tb_final_aggregation ORDER BY final_match_rate DESC LIMIT 200", params=None, fetch=True)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                raw = raw.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
                raw['final_match_rate_percentage'] = pd.to_numeric(raw['final_match_rate_percentage'], errors='coerce')
                st.dataframe(raw.head(200), width='stretch')
            else:
                st.info("No rows in tb_final_aggregation to show.")
        except Exception as e:
            st.exception(e)
        return

    df_sorted = df.sort_values('final_match_rate_percentage', ascending=False)

    st.markdown("### 🏆 Top 20 Candidates")
    try:
        st.dataframe(df_sorted.head(20), width='stretch')
    except Exception:
        st.dataframe(df_sorted.head(20), width='stretch')

    csv = df_sorted.to_csv(index=False)
    st.download_button("📥 Download Results (CSV)", csv, f"talent_match_results_{vacancy_id}.csv", "text/csv")

    if st.button("🔄 Refresh detailed results from DB"):
        with st.spinner("Refreshing..."):
            try:
                detailed = db.run_matching_query(vacancy_id)
                if detailed is None or detailed.empty:
                    st.warning("No detailed results returned on refresh.")
                else:
                    if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
                        detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
                    if 'final_match_rate_percentage' in detailed.columns:
                        detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
                        maxv = detailed['final_match_rate_percentage'].max(skipna=True)
                        if pd.notna(maxv) and maxv <= 1.0:
                            detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0
                    detailed = _drop_placeholder_rows(detailed)
                    st.session_state.matching_results_detailed = detailed
                    summ = pd.DataFrame()
                    if not detailed.empty:
                        summ = detailed.groupby('employee_id').agg({
                            'fullname': 'first',
                            'directorate': 'first',
                            'role': 'first',
                            'grade': 'first',
                            'final_match_rate_percentage': 'first'
                        }).reset_index()
                        summ['final_match_rate_percentage'] = pd.to_numeric(summ['final_match_rate_percentage'], errors='coerce')
                    st.session_state.matching_results_summary = summ.sort_values('final_match_rate_percentage', ascending=False) if not summ.empty else summ
                    st.success("Refreshed detailed results.")
            except Exception as e:
                st.exception(e)

def show_analytics_page():
    st.markdown("## 📈 Analytics & Insights")

    detailed = st.session_state.get('matching_results_detailed', None)
    summary = st.session_state.get('matching_results_summary', None)
    vacancy_id = st.session_state.get('job_vacancy_id', None)

    if detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty):
        st.warning("⚠️ No detailed results available. Please load or refresh results from 'View Results'.")
        return

    # Normalize and clean dataframes defensively (this also removes placeholder rows)
    detailed, summary = _normalize_results_dfs(detailed, summary)

    # build summary if missing after normalization
    if summary is None or summary.empty:
        if not detailed.empty and 'employee_id' in detailed.columns:
            summary = detailed.groupby('employee_id').agg({
                'fullname': 'first',
                'directorate': 'first',
                'role': 'first',
                'grade': 'first',
                'final_match_rate_percentage': 'first'
            }).reset_index()
            summary['final_match_rate_percentage'] = pd.to_numeric(summary['final_match_rate_percentage'], errors='coerce')
            summary = summary.dropna(subset=['final_match_rate_percentage'])
            st.session_state.matching_results_summary = summary

    if summary is None or summary.empty:
        st.warning("⚠️ No valid matching summary found after cleaning. Your detailed rows may be placeholders or missing numeric scores.")
        # show a tiny sample to help debug
        st.write("Sample detailed rows (post-clean):")
        st.write(detailed.head(5))
        return

    st.markdown("### 🔍 Key Insights")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_match = summary['final_match_rate_percentage'].mean()
        st.metric("Avg Match Rate", f"{avg_match:.1f}%")
    with col2:
        top_match = summary['final_match_rate_percentage'].max()
        st.metric("Top Match", f"{top_match:.1f}%")
    with col3:
        candidates_above_70 = (summary['final_match_rate_percentage'] >= 70).sum()
        st.metric("Matches ≥70%", candidates_above_70)
    with col4:
        candidates_above_80 = (summary['final_match_rate_percentage'] >= 80).sum()
        st.metric("Matches ≥80%", candidates_above_80)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Match Score Distribution")
        try:
            fig_dist = plot_match_distribution(summary)
            st.plotly_chart(fig_dist, width='stretch')
        except Exception as e:
            st.error(f"Error creating distribution chart: {e}")
    with col2:
        st.markdown("#### 🏆 Top 10 Candidates")
        try:
            fig_top = plot_top_candidates(summary, top_n=10)
            st.plotly_chart(fig_top, width='stretch')
        except Exception as e:
            st.error(f"Error creating top candidates chart: {e}")

    st.markdown("---")
    st.markdown("### 📊 TGV-Level Analysis")

    # prepare tgv_summary for selection
    try:
        tgv_summary = detailed[['employee_id', 'fullname']].drop_duplicates().copy()
        if 'final_match_rate_percentage' in summary.columns:
            summary['employee_id'] = summary['employee_id'].astype(str)
            tgv_summary['employee_id'] = tgv_summary['employee_id'].astype(str)
            tgv_summary = tgv_summary.merge(summary[['employee_id', 'final_match_rate_percentage']], on='employee_id', how='left')
        tgv_summary['final_match_rate_percentage'] = pd.to_numeric(tgv_summary.get('final_match_rate_percentage'), errors='coerce').fillna(-1)
        tgv_summary = tgv_summary.sort_values('final_match_rate_percentage', ascending=False).head(200)
    except Exception as e:
        st.error(f"Cannot prepare TGV employee list: {e}")
        return

    employee_options = tgv_summary['employee_id'].astype(str).tolist()
    if not employee_options:
        st.warning("No employees found in results.")
        return

    def _format_employee(eid):
        eid_s = str(eid)
        row = tgv_summary.loc[tgv_summary['employee_id'] == eid_s]
        if row.empty:
            return eid_s
        fullname = row['fullname'].iloc[0] if pd.notna(row['fullname'].iloc[0]) else ''
        score = row['final_match_rate_percentage'].iloc[0]
        try:
            if pd.notna(score) and float(score) >= 0:
                return f"{eid_s} - {fullname} ({float(score):.1f}%)"
        except Exception:
            pass
        return f"{eid_s} - {fullname}"

    selected_employee = st.selectbox(
        "Select Employee for TGV Profile:",
        options=employee_options,
        format_func=_format_employee
    )
    selected_employee = str(selected_employee)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 TGV Radar Profile")
        try:
            # quick guard: check if there are any numeric tgv_match_rate for this employee
            emp_rows = detailed.loc[detailed['employee_id'].astype(str) == selected_employee]
            # remove placeholder-like rows in emp_rows too
            emp_rows = _drop_placeholder_rows(emp_rows)
            if emp_rows.empty:
                st.warning("No detailed rows for this employee after cleaning. The dataset may contain placeholder rows only.")
                st.write(emp_rows.head(5))
            else:
                # check if any numeric tgv_match_rate exists
                has_tgv_numeric = ('tgv_match_rate' in emp_rows.columns) and (pd.to_numeric(emp_rows['tgv_match_rate'], errors='coerce').dropna().size > 0)
                if not has_tgv_numeric:
                    st.warning("No numeric TGV match rate values for this employee. Radar cannot be created.")
                    st.write(emp_rows.head(5))
                else:
                    fig_radar = plot_tgv_radar(emp_rows, selected_employee)
                    st.plotly_chart(fig_radar, width='stretch')
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")
            st.write("Sample rows for this employee (post-clean):")
            st.write(emp_rows.head(5).to_dict(orient='records') if 'emp_rows' in locals() else "No emp_rows available")

    with col2:
        st.markdown("#### 🔥 TV Heatmap (Top TGVs)")
        try:
            emp_rows = detailed.loc[detailed['employee_id'].astype(str) == selected_employee]
            emp_rows = _drop_placeholder_rows(emp_rows)
            if emp_rows.empty:
                st.warning("No detailed rows for this employee after cleaning. Heatmap cannot be created.")
                st.write(emp_rows.head(5))
            else:
                has_tv_numeric = ('tv_match_rate' in emp_rows.columns) and (pd.to_numeric(emp_rows['tv_match_rate'], errors='coerce').dropna().size > 0)
                if not has_tv_numeric:
                    st.warning("No numeric TV match rate values for this employee. Heatmap cannot be created.")
                    st.write(emp_rows.head(5))
                else:
                    fig_heatmap = plot_tv_heatmap(emp_rows, selected_employee)
                    st.plotly_chart(fig_heatmap, width='stretch')
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
            st.write("Sample rows for this employee (post-clean):")
            st.write(emp_rows.head(5).to_dict(orient='records') if 'emp_rows' in locals() else "No emp_rows available")

    st.markdown("---")
    st.markdown("### ✅ Strengths & Gaps Analysis")
    try:
        emp_rows = detailed.loc[detailed['employee_id'].astype(str) == selected_employee]
        emp_rows = _drop_placeholder_rows(emp_rows)
        if emp_rows.empty:
            st.warning("No detailed rows for this employee after cleaning. Strengths & Gaps chart cannot be created.")
            st.write(emp_rows.head(5))
        else:
            has_tv_numeric = ('tv_match_rate' in emp_rows.columns) and (pd.to_numeric(emp_rows['tv_match_rate'], errors='coerce').dropna().size > 0)
            if not has_tv_numeric:
                st.warning("No numeric TV match rate values for this employee. Strengths & Gaps chart cannot be created.")
                st.write(emp_rows.head(5))
            else:
                fig_strengths_gaps = plot_strengths_gaps(emp_rows, selected_employee)
                st.plotly_chart(fig_strengths_gaps, width='stretch')
    except Exception as e:
        st.error(f"Error creating strengths/gaps chart: {e}")
        st.write("Sample rows for this employee (post-clean):")
        st.write(emp_rows.head(5).to_dict(orient='records') if 'emp_rows' in locals() else "No emp_rows available")

# ----------------------------
# Main
# ----------------------------
def main():
    st.set_page_config(page_title="Talent Match Intelligence", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
    st.markdown('<div style="font-size:2.2rem;font-weight:700;color:#1f77b4;margin-bottom:0.2rem;">🎯 Talent Match Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;color:#666;margin-bottom:1rem;">AI-Powered Talent Discovery & Succession Planning</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Company+X", width=200)
        st.markdown("---")
        page = st.radio("Select Module:", ["🏠 Home", "➕ Create Vacancy", "📊 View Results", "📈 Analytics"])
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This system uses AI and statistical modeling to match employees with job vacancies based on competencies, psychometric profiles, and behavioral patterns.")

    if page == "🏠 Home":
        show_home_page()
    elif page == "➕ Create Vacancy":
        show_create_vacancy_page()
    elif page == "📊 View Results":
        show_results_page()
    elif page == "📈 Analytics":
        show_analytics_page()

if __name__ == "__main__":
    main()
