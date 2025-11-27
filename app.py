# app.py
# -*- coding: utf-8 -*-
"""
Talent Match Intelligence System - Step 3 (fixed Analytics)
Author: generated patch (adapted from project files)
Date: 2025-11-27
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Optional, List

# local utils (must exist in project)
from utils.database import DatabaseManager
from utils.llm import generate_job_profile
from utils.visualizations import (
    plot_match_distribution,
    plot_top_candidates,
    plot_tgv_radar,
    plot_tv_heatmap,
    plot_strengths_gaps
)

# -------------------------
# Page config & CSS
# -------------------------
st.set_page_config(
    page_title="Talent Match Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight:700; color:#1f77b4; }
    .sub-header  { font-size: 1.0rem; color:#666; margin-bottom:1rem; }
    .metric-card { background:#f0f2f6; padding:0.8rem; border-radius:6px; border-left:4px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helper / fallback
# -------------------------
def generate_fallback_profile(role_name: str, job_level: str, role_purpose: str) -> dict:
    return {
        "requirements": f"Experience in {role_name}, level: {job_level}. {role_purpose}",
        "description": f"{role_name} — {job_level}. {role_purpose}",
        "competencies": [
            "Analytical thinking",
            "Problem solving",
            "Communication",
            "Domain knowledge"
        ]
    }

# -------------------------
# Session state defaults
# -------------------------
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

# -------------------------
# Database init
# -------------------------
@st.cache_resource
def init_database():
    return DatabaseManager()

db = init_database()

# -------------------------
# Pages
# -------------------------
def show_home_page():
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            total_employees = db.get_total_employees()
        except Exception:
            total_employees = 0
        st.metric("Total Employees", f"{int(total_employees):,}")

    with col2:
        try:
            high_performers = db.get_high_performers_count()
        except Exception:
            high_performers = 0
        st.metric("High Performers (Rating 5)", f"{int(high_performers):,}")

    with col3:
        try:
            total_vacancies = db.get_total_vacancies()
        except Exception:
            total_vacancies = 0
        st.metric("Active Vacancies", f"{int(total_vacancies):,}")

    st.markdown("---")
    st.markdown("## 🔍 How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1️⃣ Define Success")
        st.write("Select high-performing employees (rating 5) as benchmarks for the role.")
    with c2:
        st.markdown("### 2️⃣ AI Analysis")
        st.write("System analyzes multiple dimensions to derive a success profile.")
    with c3:
        st.markdown("### 3️⃣ Match & Rank")
        st.write("All employees are scored against benchmarks and ranked.")

    st.markdown("---")
    st.markdown("## 📋 Recent Vacancies")
    try:
        recent = db.get_recent_vacancies(limit=5)
    except Exception:
        recent = pd.DataFrame()
    if isinstance(recent, pd.DataFrame) and not recent.empty:
        try:
            st.dataframe(recent, width='stretch')
        except Exception:
            st.dataframe(recent)
    else:
        st.info("No vacancies yet. Create one in 'Create Vacancy'.")

def show_create_vacancy_page():
    st.markdown("## ➕ Create New Job Vacancy")
    st.markdown("Define role and choose benchmark employees.")

    with st.form("vac_form"):
        left, right = st.columns(2)
        with left:
            role_name = st.text_input("Role Name *", placeholder="e.g., Data Analyst")
            job_level = st.selectbox("Job Level *", ["Entry", "Junior", "Middle", "Senior", "Lead", "Manager", "Director"], index=2)
            role_purpose = st.text_area("Role Purpose *", placeholder="1-2 sentence summary...", height=90)
        with right:
            try:
                high_df = db.get_high_performers()
            except Exception:
                high_df = pd.DataFrame()
            if high_df is None or high_df.empty:
                st.error("No high performers found in DB.")
                selected_ids = []
            else:
                id_list = high_df['employee_id'].astype(str).tolist()
                id_to_label = {str(r['employee_id']): f"{r['employee_id']} - {r['fullname']} ({r.get('position','')})" for _, r in high_df.iterrows()}
                selected_ids = st.multiselect("Select Benchmark Employees *", options=id_list, format_func=lambda x: id_to_label.get(str(x), str(x)), help="Pilih minimal 2 benchmark")

        submitted = st.form_submit_button("🚀 Create Vacancy & Run Matching")
        if submitted:
            if not role_name or not role_purpose or len(selected_ids) < 2:
                st.error("Fill all fields and select at least 2 benchmarks.")
                st.stop()

            with st.spinner("Creating vacancy..."):
                try:
                    vacancy_id = db.insert_vacancy(role_name, job_level, role_purpose, selected_ids, weights_config=None)
                    st.success(f"Vacancy #{vacancy_id} created.")
                except Exception as e:
                    st.error(f"Error inserting vacancy: {e}")
                    st.stop()

            # Generate AI profile
            with st.spinner("Generating job profile..."):
                try:
                    benchmarks_df = high_df[high_df['employee_id'].astype(str).isin([str(x) for x in selected_ids])]
                    job_profile = generate_job_profile(role_name=role_name, job_level=job_level, role_purpose=role_purpose, benchmark_employees=benchmarks_df)
                except Exception:
                    job_profile = generate_fallback_profile(role_name, job_level, role_purpose)

            # Run matching SQL pipeline
            with st.spinner("Computing talent matches..."):
                try:
                    matching_results = db.run_matching_query(vacancy_id)
                    if matching_results is None or matching_results.empty:
                        st.error("Matching returned no results.")
                        st.stop()
                except Exception as e:
                    st.error(f"Matching error: {e}")
                    st.stop()

            # Normalize/prepare session state
            detailed = matching_results.copy()
            if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
                detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
            if 'final_match_rate_percentage' in detailed.columns:
                detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
                maxv = detailed['final_match_rate_percentage'].max(skipna=True)
                if pd.notna(maxv) and maxv <= 1.0:
                    detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

            summary_df = detailed.groupby('employee_id').agg({
                'fullname': 'first',
                'directorate': 'first',
                'role': 'first',
                'grade': 'first',
                'final_match_rate_percentage': 'first'
            }).reset_index()
            summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df['final_match_rate_percentage'], errors='coerce')

            st.session_state.vacancy_created = True
            st.session_state.job_vacancy_id = vacancy_id
            st.session_state.matching_results_detailed = detailed
            st.session_state.matching_results_summary = summary_df
            st.session_state.job_profile = job_profile

            st.balloons()
            st.markdown("### 🎯 AI-Generated Job Profile")
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown("**📋 Requirements:**")
                st.write(job_profile.get('requirements','N/A'))
                st.markdown("**📝 Description:**")
                st.write(job_profile.get('description','N/A'))
            with c2:
                st.markdown("**🎯 Competencies:**")
                comps = job_profile.get('competencies', [])
                if isinstance(comps, list):
                    for comp in comps:
                        st.markdown(f"• {comp}")
                else:
                    st.write(comps)
            st.info("👉 Go to 'View Results' or 'Analytics' for insights.")

def show_results_page():
    st.markdown("## 📊 Talent Matching Results")

    if not st.session_state.get('vacancy_created', False):
        st.warning("No vacancy created. Load an existing vacancy or create a new one.")
        existing = db.get_recent_vacancies(limit=20)
        if not isinstance(existing, pd.DataFrame) or existing.empty:
            st.info("No vacancies in DB.")
            return

        selected = st.selectbox("Select vacancy:", options=existing['job_vacancy_id'].tolist(), format_func=lambda x: f"ID {x}: {existing[existing['job_vacancy_id']==x]['role_name'].iloc[0]}")
        if st.button("📥 Load Results"):
            with st.spinner("Loading vacancy..."):
                try:
                    vac_id = int(selected) if isinstance(selected, (str,)) and str(selected).isdigit() else selected
                    # Try summary first, else run full pipeline
                    try:
                        summary_df = db.get_summary_results(vac_id, limit=5000)
                    except Exception:
                        summary_df = pd.DataFrame()

                    detailed = pd.DataFrame()
                    if summary_df is None or (isinstance(summary_df, pd.DataFrame) and summary_df.empty):
                        detailed = db.run_matching_query(vac_id)
                        if detailed is None or detailed.empty:
                            st.error("Matching returned no detailed rows.")
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
                            'fullname':'first','directorate':'first','role':'first','grade':'first','final_match_rate_percentage':'first'
                        }).reset_index()
                    else:
                        try:
                            detailed = db.run_matching_query(vac_id)
                        except Exception:
                            detailed = pd.DataFrame()

                    if 'final_match_rate_percentage' in summary_df.columns:
                        summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df['final_match_rate_percentage'], errors='coerce')
                        maxv = summary_df['final_match_rate_percentage'].max(skipna=True)
                        if pd.notna(maxv) and maxv <= 1.0:
                            summary_df['final_match_rate_percentage'] = summary_df['final_match_rate_percentage'] * 100.0

                    st.session_state.job_vacancy_id = vac_id
                    st.session_state.matching_results_summary = summary_df.copy()
                    st.session_state.matching_results_detailed = detailed.copy() if isinstance(detailed, pd.DataFrame) else pd.DataFrame()
                    st.session_state.vacancy_created = True
                    st.success(f"Loaded vacancy {vac_id} — {len(summary_df)} rows.")
                except Exception as e:
                    st.exception(e)
        return

    # show summary results from session
    summary = st.session_state.get('matching_results_summary', pd.DataFrame())
    vacancy_id = st.session_state.get('job_vacancy_id', None)

    if summary is None or summary.empty:
        st.error("No summary results available.")
        return

    df = summary.copy()
    if 'final_match_rate' in df.columns and 'final_match_rate_percentage' not in df.columns:
        df = df.rename(columns={'final_match_rate':'final_match_rate_percentage'})
    df['final_match_rate_percentage'] = pd.to_numeric(df.get('final_match_rate_percentage'), errors='coerce')
    df_sorted = df.sort_values('final_match_rate_percentage', ascending=False)

    st.markdown("### 🏆 Top 20 Candidates")
    try:
        st.dataframe(df_sorted.head(20), width='stretch')
    except Exception:
        st.dataframe(df_sorted.head(20))

    csv = df_sorted.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"talent_match_results_{vacancy_id}.csv", "text/csv")

    if st.button("🔄 Refresh detailed results from DB"):
        with st.spinner("Refreshing detailed results..."):
            try:
                detailed = db.run_matching_query(vacancy_id)
                if detailed is None or detailed.empty:
                    st.warning("No detailed results returned on refresh.")
                else:
                    if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
                        detailed = detailed.rename(columns={'final_match_rate':'final_match_rate_percentage'})
                    if 'final_match_rate_percentage' in detailed.columns:
                        detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
                        maxv = detailed['final_match_rate_percentage'].max(skipna=True)
                        if pd.notna(maxv) and maxv <= 1.0:
                            detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0
                    st.session_state.matching_results_detailed = detailed
                    summ = detailed.groupby('employee_id').agg({
                        'fullname':'first','directorate':'first','role':'first','grade':'first','final_match_rate_percentage':'first'
                    }).reset_index()
                    summ['final_match_rate_percentage'] = pd.to_numeric(summ['final_match_rate_percentage'], errors='coerce')
                    st.session_state.matching_results_summary = summ.sort_values('final_match_rate_percentage', ascending=False)
                    st.success("Refreshed detailed results.")
            except Exception as e:
                st.exception(e)

def show_analytics_page():
    st.markdown("## 📈 Analytics & Insights")

    detailed = st.session_state.get('matching_results_detailed', None)
    summary = st.session_state.get('matching_results_summary', None)
    vacancy_id = st.session_state.get('job_vacancy_id', None)

    # If detailed missing or doesn't appear to have TGV/TV columns, try to load from DB
    need_reload = False
    if detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty):
        need_reload = True
    else:
        # ensure detail contains expected columns for visualizations
        required_cols = {'employee_id', 'tgv_name', 'tv_name', 'tv_match_rate'}
        if not required_cols.intersection(set(detailed.columns)):
            need_reload = True

    if need_reload:
        st.info("Loading detailed results for analytics...")
        if vacancy_id is None:
            st.warning("No vacancy selected. Load vacancy in 'View Results' first.")
            return
        try:
            detailed = db.run_matching_query(vacancy_id)
            if detailed is None or detailed.empty:
                st.error("Detailed matching results could not be loaded from DB.")
                return
            st.session_state.matching_results_detailed = detailed
        except Exception as e:
            st.error(f"Failed to load detailed results: {e}")
            return

    # Normalize final match column
    if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
        detailed = detailed.rename(columns={'final_match_rate':'final_match_rate_percentage'})
    if 'final_match_rate_percentage' in detailed.columns:
        detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
        maxv = detailed['final_match_rate_percentage'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0:
            detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

    # Build summary if missing
    if summary is None or summary.empty:
        summary = detailed.groupby('employee_id').agg({
            'fullname':'first','directorate':'first','role':'first','grade':'first','final_match_rate_percentage':'first'
        }).reset_index()
        summary['final_match_rate_percentage'] = pd.to_numeric(summary['final_match_rate_percentage'], errors='coerce')
        st.session_state.matching_results_summary = summary

    if summary.empty:
        st.warning("No valid summary available for analytics.")
        return

    # Key metrics
    st.markdown("### 🔍 Key Insights")
    c1,c2,c3,c4 = st.columns(4)
    avg_match = summary['final_match_rate_percentage'].mean()
    top_match = summary['final_match_rate_percentage'].max()
    c1.metric("Avg Match Rate", f"{avg_match:.1f}%")
    c2.metric("Top Match", f"{top_match:.1f}%")
    c3.metric("Matches ≥70%", int((summary['final_match_rate_percentage'] >= 70).sum()))
    c4.metric("Matches ≥80%", int((summary['final_match_rate_percentage'] >= 80).sum()))
    st.markdown("---")

    # Visualizations: distribution & top candidates
    left, right = st.columns(2)
    with left:
        try:
            fig_dist = plot_match_distribution(summary)
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating distribution chart: {e}")
    with right:
        try:
            fig_top = plot_top_candidates(summary, top_n=10)
            st.plotly_chart(fig_top, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating top candidates chart: {e}")

    st.markdown("---")
    st.markdown("### 📊 TGV-Level Analysis")

    # Prepare tgv_summary for selectbox (use detailed)
    try:
        tgv_summary = detailed[['employee_id', 'fullname']].drop_duplicates().copy()
        if 'final_match_rate_percentage' in summary.columns:
            tgv_summary = tgv_summary.merge(summary[['employee_id','final_match_rate_percentage']], on='employee_id', how='left')
        tgv_summary['final_match_rate_percentage'] = pd.to_numeric(tgv_summary.get('final_match_rate_percentage'), errors='coerce').fillna(0)
        tgv_summary = tgv_summary.sort_values('final_match_rate_percentage', ascending=False).head(500)
    except Exception as e:
        st.error(f"Cannot prepare employee list for TGV analysis: {e}")
        return

    if tgv_summary.empty:
        st.warning("No employees found in detailed results.")
        return

    emp_options = tgv_summary['employee_id'].astype(str).tolist()
    def fmt_emp(eid):
        row = tgv_summary[tgv_summary['employee_id'].astype(str) == str(eid)]
        if row.empty:
            return str(eid)
        name = row['fullname'].iloc[0]
        pct = float(row['final_match_rate_percentage'].iloc[0] if pd.notna(row['final_match_rate_percentage'].iloc[0]) else 0.0)
        return f"{eid} - {name} ({pct:.1f}%)"

    selected_employee = st.selectbox("Select Employee for TGV Profile:", options=emp_options, format_func=fmt_emp, index=0)

    # Filter employee detail once and pass that to visualization functions
    emp_detail = detailed[detailed['employee_id'].astype(str) == str(selected_employee)].copy()
    # Ensure string columns exist
    if 'tgv_name' in emp_detail.columns:
        emp_detail['tgv_name'] = emp_detail['tgv_name'].astype(object).where(~emp_detail['tgv_name'].isna(), other=np.nan)
        emp_detail['tgv_name'] = emp_detail['tgv_name'].apply(lambda x: x if pd.isna(x) or isinstance(x, str) else str(x))
    if 'tv_name' in emp_detail.columns:
        emp_detail['tv_name'] = emp_detail['tv_name'].astype(object).where(~emp_detail['tv_name'].isna(), other=np.nan)
        emp_detail['tv_name'] = emp_detail['tv_name'].apply(lambda x: x if pd.isna(x) or isinstance(x, str) else str(x))
        
        st.subheader("DEBUG — detailed.columns")
        st.write(list(detailed.columns))
        
        st.subheader("DEBUG — detailed (10 baris)")
        st.write(detailed.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 TGV Radar Profile")
        try:
            fig_radar = plot_tgv_radar(emp_detail, selected_employee)
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")

    with col2:
        st.markdown("#### 🔥 TV Heatmap (Top TGVs)")
        try:
            fig_heatmap = plot_tv_heatmap(emp_detail, selected_employee)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

    st.markdown("---")
    st.markdown("### ✅ Strengths & Gaps Analysis")
    try:
        fig_sg = plot_strengths_gaps(emp_detail, selected_employee)
        st.plotly_chart(fig_sg, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating strengths/gaps chart: {e}")

# -------------------------
# Main (routing)
# -------------------------
def main():
    st.markdown('<div class="main-header">🎯 Talent Match Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Talent Discovery & Succession Planning</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Company+X", width=200)
        st.markdown("---")
        page = st.radio("Select Module:", ["🏠 Home", "➕ Create Vacancy", "📊 View Results", "📈 Analytics"])
        st.markdown("---")
        st.info("This system uses AI and statistical modeling to match employees to vacancies.")

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
