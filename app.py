# app.py
# -*- coding: utf-8 -*-
"""
Talent Match Intelligence System - Step 3
AI-Powered Talent Matching Dashboard

Author: TRI ADI BASKORO (revised)
Date: 18 November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

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

# Page config
st.set_page_config(
    page_title="Talent Match Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Simple fallback generator for job profile (used if LLM fails)
def generate_fallback_profile(role_name: str, job_level: str, role_purpose: str) -> dict:
    """Return a basic fallback job profile."""
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

# Initialize session state
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

# Initialize database connection (cached resource)
@st.cache_resource
def init_database():
    return DatabaseManager()

db = init_database()

# -----------------------------
# Helper utilities
# -----------------------------
def build_summary_from_detailed(detailed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detailed per-TV rows into one-row-per-employee summary."""
    if detailed is None or detailed.empty:
        return pd.DataFrame()
    df = detailed.copy()
    # Normalize final column names
    if 'final_match_rate' in df.columns and 'final_match_rate_percentage' not in df.columns:
        df = df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
    # Choose best numeric candidate
    if 'final_match_rate_percentage' not in df.columns:
        for c in ['final_match_rate', 'tgv_match_rate', 'tv_match_rate']:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df['final_match_rate_percentage'] = df[c]
                break
    # Ensure numeric and scale to 0..100 if necessary
    if 'final_match_rate_percentage' in df.columns:
        df['final_match_rate_percentage'] = pd.to_numeric(df['final_match_rate_percentage'], errors='coerce')
        maxv = df['final_match_rate_percentage'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0:
            df['final_match_rate_percentage'] = df['final_match_rate_percentage'] * 100.0
    # Group
    summary = df.groupby('employee_id').agg({
        'fullname': 'first',
        'directorate': 'first',
        'role': 'first',
        'grade': 'first',
        'final_match_rate_percentage': 'first'
    }).reset_index()
    # Ensure numeric column exists
    if 'final_match_rate_percentage' not in summary.columns:
        summary['final_match_rate_percentage'] = np.nan
    summary['final_match_rate_percentage'] = pd.to_numeric(summary['final_match_rate_percentage'], errors='coerce')
    return summary

def has_numeric_values(df: pd.DataFrame, employee_id: str, col: str) -> bool:
    """Return True if df contains at least one numeric non-null value in col for employee_id."""
    if df is None or df.empty or col not in df.columns:
        return False
    sub = df[df['employee_id'] == employee_id]
    if sub.empty:
        return False
    numeric = pd.to_numeric(sub[col], errors='coerce')
    return numeric.notna().any()

# =========================
# Page functions
# =========================

def show_home_page():
    """Home page with system overview"""
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            total_employees = db.get_total_employees()
        except Exception:
            total_employees = 0
        st.metric("Total Employees", f"{total_employees:,}")

    with col2:
        try:
            high_performers = db.get_high_performers_count()
        except Exception:
            high_performers = 0
        st.metric("High Performers (Rating 5)", f"{high_performers:,}")

    with col3:
        try:
            total_vacancies = db.get_total_vacancies()
        except Exception:
            total_vacancies = 0
        st.metric("Active Vacancies", total_vacancies)

    st.markdown("---")
    st.markdown("## 🔍 How It Works")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Define Success")
        st.write("Select high-performing employees (rating 5) as benchmarks for the role.")
    with col2:
        st.markdown("### 2️⃣ AI Analysis")
        st.write("System analyzes talent dimensions across variables to find patterns.")
    with col3:
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

def show_create_vacancy_page():
    """Page for creating new job vacancy"""
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
                id_list = high_performers_df['employee_id'].astype(str).tolist()
                id_to_label = {str(row['employee_id']): f"{row['employee_id']} - {row['fullname']}" for _, row in high_performers_df.iterrows()}
                selected_ids = st.multiselect("Select Benchmark Employees *", options=id_list, format_func=lambda eid: id_to_label.get(str(eid), str(eid)), help="Pilih minimal 2 benchmark")
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

            # Generate AI profile
            with st.spinner("🤖 Generating job profile with AI..."):
                try:
                    benchmarks_df = high_performers_df[high_performers_df['employee_id'].astype(str).isin(selected_employees)]
                    job_profile = generate_job_profile(role_name=role_name, job_level=job_level, role_purpose=role_purpose, benchmark_employees=benchmarks_df)
                except Exception as e:
                    st.warning(f"AI generation failed: {e}. Using fallback.")
                    job_profile = generate_fallback_profile(role_name, job_level, role_purpose)

            # Run matching pipeline (mandatory to populate baseline & results)
            with st.spinner("📊 Computing talent matches..."):
                try:
                    detailed = db.run_matching_query(vacancy_id)
                    if not isinstance(detailed, pd.DataFrame) or detailed.empty:
                        st.error("❌ Matching returned no results. Check your data or pipeline logs.")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Matching error: {e}")
                    st.exception(e)
                    st.stop()

            # Normalize result and store to session state
            detailed = detailed.copy()
            if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
                detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
            if 'final_match_rate_percentage' in detailed.columns:
                detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
                maxv = detailed['final_match_rate_percentage'].max(skipna=True)
                if pd.notna(maxv) and maxv <= 1.0:
                    detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

            summary_df = build_summary_from_detailed(detailed)

            st.session_state.vacancy_created = True
            st.session_state.job_vacancy_id = vacancy_id
            st.session_state.matching_results_detailed = detailed
            st.session_state.matching_results_summary = summary_df
            st.session_state.job_profile = job_profile

            # Display AI job profile
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
    """Results page — always ensure we have fresh detailed results by calling run_matching_query when needed"""
    st.markdown("## 📊 Talent Matching Results")

    # If no vacancy in session, provide option to load existing vacancy and run pipeline
    if not st.session_state.get('vacancy_created', False):
        st.warning("⚠️ No vacancy in session. Create one or load an existing vacancy to view results.")
        existing = db.get_recent_vacancies(20)
        if not isinstance(existing, pd.DataFrame) or existing.empty:
            st.info("No vacancies found in DB.")
            return

        selected = st.selectbox(
            "Select vacancy:",
            options=existing['job_vacancy_id'].tolist(),
            format_func=lambda x: f"ID {x}: {existing[existing['job_vacancy_id']==x]['role_name'].iloc[0]}"
        )
        if st.button("📥 Load & Run Matching for Selected Vacancy"):
            vac_id = int(selected)
            with st.spinner("Running matching pipeline for selected vacancy..."):
                try:
                    detailed = db.run_matching_query(vac_id)
                    if detailed is None or detailed.empty:
                        st.error("Matching returned no detailed rows. Pipeline may have failed.")
                        return
                    summary_df = build_summary_from_detailed(detailed)
                    st.session_state.job_vacancy_id = vac_id
                    st.session_state.matching_results_detailed = detailed
                    st.session_state.matching_results_summary = summary_df
                    st.session_state.vacancy_created = True
                    st.success(f"Loaded vacancy {vac_id} — {len(summary_df)} candidates")
                except Exception as e:
                    st.exception(e)
        return

    # If we have session results, show them
    summary_df = st.session_state.matching_results_summary
    detailed = st.session_state.matching_results_detailed
    vacancy_id = st.session_state.job_vacancy_id

    if summary_df is None or (isinstance(summary_df, pd.DataFrame) and summary_df.empty):
        st.error("matching_results_summary is empty. Try refreshing results.")
        return

    # Ensure numeric
    summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df.get('final_match_rate_percentage'), errors='coerce')
    # Sort
    df_sorted = summary_df.sort_values('final_match_rate_percentage', ascending=False)

    st.markdown("### 🏆 Top 20 Candidates")
    try:
        st.dataframe(df_sorted.head(20), width='stretch')
    except Exception:
        st.dataframe(df_sorted.head(20))

    # Download
    csv = df_sorted.to_csv(index=False)
    st.download_button("📥 Download Results (CSV)", csv, f"talent_match_results_{vacancy_id}.csv", "text/csv")

    # Refresh button: rerun pipeline and update session state
    if st.button("🔄 Refresh detailed results (re-run pipeline)"):
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
                    summary_df = build_summary_from_detailed(detailed)
                    st.session_state.matching_results_detailed = detailed
                    st.session_state.matching_results_summary = summary_df
                    st.success("Refreshed detailed results.")
            except Exception as e:
                st.exception(e)

def show_analytics_page():
    """Analytics & Visualizations"""
    st.markdown("## 📈 Analytics & Insights")

    detailed = st.session_state.get('matching_results_detailed', None)
    summary = st.session_state.get('matching_results_summary', None)
    vacancy_id = st.session_state.get('job_vacancy_id', None)

    if detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty):
        st.warning("⚠️ No detailed results available. Please load or refresh results from 'View Results'.")
        return

    # Normalize final column and scale
    if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
        detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
    if 'final_match_rate_percentage' in detailed.columns:
        detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
        maxv = detailed['final_match_rate_percentage'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0:
            detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

    # Build summary if missing
    if summary is None or summary.empty:
        summary = build_summary_from_detailed(detailed)
        st.session_state.matching_results_summary = summary

    if summary.empty:
        st.warning("⚠️ No valid matching summary found.")
        return

    # Key insights
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
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating distribution chart: {e}")
    with col2:
        st.markdown("#### 🏆 Top 10 Candidates")
        try:
            fig_top = plot_top_candidates(summary, top_n=10)
            st.plotly_chart(fig_top, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating top candidates chart: {e}")

    st.markdown("---")
    st.markdown("### 📊 TGV-Level Analysis")

    # Prepare employee selection for TGV analysis
    try:
        tgv_summary = detailed[['employee_id', 'fullname']].drop_duplicates().copy()
        if 'final_match_rate_percentage' in summary.columns:
            tgv_summary = tgv_summary.merge(summary[['employee_id', 'final_match_rate_percentage']], on='employee_id', how='left')
        tgv_summary = tgv_summary.sort_values('final_match_rate_percentage', ascending=False).head(200)
    except Exception:
        st.error("Cannot prepare TGV employee list.")
        return

    employee_options = tgv_summary['employee_id'].tolist()
    if not employee_options:
        st.warning("No employees found in results.")
        return

    def format_employee(eid):
        row = tgv_summary[tgv_summary['employee_id'] == eid]
        if row.empty:
            return str(eid)
        score = row['final_match_rate_percentage'].iloc[0]
        return f"{eid} - {row['fullname'].iloc[0]} ({score:.1f}%)" if pd.notna(score) else f"{eid} - {row['fullname'].iloc[0]}"

    selected_employee = st.selectbox("Select Employee for TGV Profile:", options=employee_options, format_func=format_employee)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 TGV Radar Profile")
        # check numeric tgv_match_rate exists
        if not has_numeric_values(detailed, selected_employee, 'tgv_match_rate'):
            st.warning("No numeric tgv_match_rate values found for this employee. This usually means baseline wasn't available for the vacancy or the pipeline hasn't been run for that vacancy.")
        else:
            try:
                fig_radar = plot_tgv_radar(detailed, selected_employee)
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating radar chart: {e}")

    with col2:
        st.markdown("#### 🔥 TV Heatmap (Top TGVs)")
        if not has_numeric_values(detailed, selected_employee, 'tv_match_rate'):
            st.warning("No numeric tv_match_rate values found for this employee. TV-level scores missing.")
        else:
            try:
                fig_heatmap = plot_tv_heatmap(detailed, selected_employee)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")

    st.markdown("---")
    st.markdown("### ✅ Strengths & Gaps Analysis")
    if not has_numeric_values(detailed, selected_employee, 'tv_match_rate'):
        st.warning("No numeric tv_match_rate values found for this employee. Strengths/Gaps chart requires TV-level numeric scores.")
    else:
        try:
            fig_strengths_gaps = plot_strengths_gaps(detailed, selected_employee)
            st.plotly_chart(fig_strengths_gaps, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating strengths/gaps chart: {e}")

# ================
# Main app
# ================
def main():
    st.markdown('<div class="main-header">🎯 Talent Match Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Talent Discovery & Succession Planning</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Company+X", width=200)
        st.markdown("---")
        st.markdown("### 📋 Navigation")
        page = st.radio("Select Module:", ["🏠 Home", "➕ Create Vacancy", "📊 View Results", "📈 Analytics"])
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This system uses AI and statistical modeling to match employees with job vacancies.")

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
