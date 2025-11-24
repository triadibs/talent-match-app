# -*- coding: utf-8 -*-
"""
Talent Match Intelligence System - Step 3
AI-Powered Talent Matching Dashboard

Author: TRI ADI BASKORO (revised)
Date: 18 November 2025

Notes:
- This file expects utils.database.DatabaseManager and utils.visualizations functions to exist.
- Replace/extend generate_fallback_profile as desired.
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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
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
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = None

# Initialize database connection
@st.cache_resource
def init_database():
    """Initialize database connection"""
    return DatabaseManager()

db = init_database()

# =========================
# Page functions (all defined before main)
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

    # How it works
    st.markdown("## 🔍 How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Define Success")
        st.write("Select high-performing employees (rating 5) as benchmarks for the role.")

    with col2:
        st.markdown("### 2️⃣ AI Analysis")
        st.write("System analyzes 7 talent dimensions across 50+ variables to find patterns.")

    with col3:
        st.markdown("### 3️⃣ Match & Rank")
        st.write("All employees are scored against the benchmark profile and ranked.")

    st.markdown("---")

    # Recent vacancies
    st.markdown("## 📋 Recent Vacancies")
    try:
        recent_vacancies = db.get_recent_vacancies(limit=5)
    except Exception:
        recent_vacancies = pd.DataFrame()

    if isinstance(recent_vacancies, pd.DataFrame) and not recent_vacancies.empty:
        # Use width='stretch' as replacement for use_container_width
        try:
            st.dataframe(
                recent_vacancies,
                column_config={
                    "job_vacancy_id": "ID",
                    "role_name": "Role",
                    "job_level": "Level",
                    "created_at": st.column_config.DatetimeColumn("Created", format="DD MMM YYYY"),
                    "benchmark_count": st.column_config.NumberColumn("Benchmarks", format="%d")
                },
                hide_index=True,
                width='stretch'
            )
        except Exception:
            # Fallback for Streamlit versions without column_config features
            st.dataframe(recent_vacancies.head(10), use_container_width=True)
    else:
        st.info("No vacancies created yet. Go to 'Create Vacancy' to get started!")

def show_create_vacancy_page():
    """Page for creating new job vacancy"""
    
    st.markdown("## ➕ Create New Job Vacancy")
    st.markdown("Define the role and select benchmark employees to find the best matches.")
    
    # Form
    with st.form("vacancy_form"):
        st.markdown("### 📝 Job Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            role_name = st.text_input(
                "Role Name *",
                placeholder="e.g., Data Analyst, Product Manager"
            )
            job_level = st.selectbox(
                "Job Level *",
                ["Entry", "Junior", "Middle", "Senior", "Lead", "Manager", "Director"],
                index=2
            )
        
        with col2:
            high_performers_df = db.get_high_performers()
            
            if not isinstance(high_performers_df, pd.DataFrame) or high_performers_df.empty:
                st.error("No high performers found!")
                selected_employees = []
            else:
                # safer options: use employee_id as option value, show label via format_func
                id_list = high_performers_df['employee_id'].astype(str).tolist()
                id_to_label = {
                    str(row['employee_id']): f"{row['employee_id']} - {row['fullname']} ({row.get('position','')})"
                    for _, row in high_performers_df.iterrows()
                }
                selected_ids = st.multiselect(
                    "Select Benchmark Employees *",
                    options=id_list,
                    format_func=lambda eid: id_to_label.get(str(eid), str(eid)),
                    help="Pilih minimal 2 benchmark"
                )
                selected_employees = [str(x) for x in selected_ids]
        
        role_purpose = st.text_area(
            "Role Purpose *",
            placeholder="1-2 sentence summary...",
            height=100
        )
        
        submitted = st.form_submit_button("🚀 Create Vacancy & Run Matching", type="primary")
        
        if submitted:
            if not role_name or not role_purpose or len(selected_employees) < 2:
                st.error("❌ Please fill all required fields and select at least 2 benchmarks")
                st.stop()
            
            # PROCESS
            with st.spinner("🔄 Creating vacancy..."):
                try:
                    # 1. Insert vacancy
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
            
            # 2. Generate AI profile
            with st.spinner("🤖 Generating job profile with AI..."):
                try:
                    # Prepare benchmark employees DF for LLM
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
            
            # 3. Run matching SQL
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
            
            # 4. Save to session state
            st.session_state.vacancy_created = True
            st.session_state.job_vacancy_id = vacancy_id
            st.session_state.matching_results = matching_results.copy() if isinstance(matching_results, pd.DataFrame) else matching_results
            st.session_state.job_profile = job_profile
            
            # 5. DISPLAY AI JOB PROFILE (THIS IS CRITICAL!)
            st.balloons()
            st.markdown("---")
            st.markdown("### 🎯 AI-Generated Job Profile")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**📋 Job Requirements:**")
                with st.container():
                    st.write(job_profile.get('requirements', 'N/A'))
                
                st.markdown("**📝 Job Description:**")
                with st.container():
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
    """Page showing matching results (stores both detailed + summary in session_state)."""
    st.markdown("## 📊 Talent Matching Results")

    # If no vacancy created in session, allow load existing
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
            with st.spinner("Loading detailed matching results..."):
                try:
                    vac_id = int(selected)
                    # ALWAYS fetch detailed results (per-TV) from the pipeline
                    detailed = db.run_matching_query(vac_id)

                    if detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty):
                        st.error("Matching returned no detailed rows. Pipeline may have failed.")
                        return

                    # Ensure numeric final_match_rate exists (some pipelines return final_match_rate)
                    if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
                        detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})

                    # Coerce numeric and scale if necessary (some SQL produce 0..1)
                    if 'final_match_rate_percentage' in detailed.columns:
                        detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
                        maxv = detailed['final_match_rate_percentage'].max(skipna=True)
                        if pd.notna(maxv) and maxv <= 1.0:
                            detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

                    # Create summary per employee (first row fields + aggregated final_match_rate)
                    summary_df = detailed.groupby('employee_id').agg({
                        'fullname': 'first',
                        'directorate': 'first',
                        'role': 'first',
                        'grade': 'first',
                        'final_match_rate_percentage': 'first'
                    }).reset_index()

                    # Clean summary numeric
                    summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df['final_match_rate_percentage'], errors='coerce')
                    summary_df = summary_df.dropna(subset=['final_match_rate_percentage'])
                    summary_df = summary_df.sort_values('final_match_rate_percentage', ascending=False)

                    # Save both in session_state
                    st.session_state.job_vacancy_id = vac_id
                    st.session_state.matching_results_detailed = detailed
                    st.session_state.matching_results_summary = summary_df
                    st.session_state.vacancy_created = True

                    st.success(f"Loaded vacancy {vac_id} — {len(summary_df)} candidates (detailed rows: {len(detailed)}).")
                    # fall through to display below

                except Exception as e:
                    st.exception(e)
                    return
        return

    # If we reach here, we already have a vacancy in session (either just created or previously loaded)
    vacancy_id = st.session_state.job_vacancy_id

    # Prefer the summary stored, otherwise build from detailed if available
    summary_df = st.session_state.get('matching_results_summary', None)
    detailed_df = st.session_state.get('matching_results_detailed', None)

    if summary_df is None and detailed_df is not None:
        summary_df = detailed_df.groupby('employee_id').agg({
            'fullname': 'first',
            'directorate': 'first',
            'role': 'first',
            'grade': 'first',
            'final_match_rate_percentage': 'first'
        }).reset_index()
        st.session_state.matching_results_summary = summary_df

    if summary_df is None or summary_df.empty:
        st.error("No results available to display. Try loading vacancy results again.")
        return

    # Display top 20 candidates (summary)
    st.markdown("### 🏆 Top 20 Candidates")

    try:
        st.dataframe(
            summary_df.head(20),
            column_config={
                "employee_id": "ID",
                "fullname": "Name",
                "directorate": "Directorate",
                "role": "Current Role",
                "grade": "Grade",
                "final_match_rate_percentage": st.column_config.ProgressColumn(
                    "Match Score",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                )
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
    except Exception:
        st.dataframe(summary_df.head(20), use_container_width=True, height=600)

    # Download
    csv = summary_df.to_csv(index=False)
    st.download_button(
        "📥 Download Results (CSV)",
        csv,
        f"talent_match_results_{vacancy_id}.csv",
        "text/csv"
    )

    # Provide a quick button to refresh detailed results if needed
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
                    st.session_state.matching_results_detailed = detailed
                    # also refresh summary
                    summ = detailed.groupby('employee_id').agg({
                        'fullname': 'first',
                        'directorate': 'first',
                        'role': 'first',
                        'grade': 'first',
                        'final_match_rate_percentage': 'first'
                    }).reset_index()
                    summ['final_match_rate_percentage'] = pd.to_numeric(summ['final_match_rate_percentage'], errors='coerce')
                    st.session_state.matching_results_summary = summ.sort_values('final_match_rate_percentage', ascending=False)
                    st.success("Refreshed detailed results.")
            except Exception as e:
                st.exception(e)

def show_analytics_page():
    """Page with visualizations and analytics using detailed results for charts."""
    st.markdown("## 📈 Analytics & Insights")

    # Require detailed results for visualizations
    detailed = st.session_state.get('matching_results_detailed', None)
    summary = st.session_state.get('matching_results_summary', None)
    vacancy_id = st.session_state.get('job_vacancy_id', None)

    if detailed is None or detailed.empty:
        st.warning("⚠️ No detailed results available. Please create a vacancy or load existing results (use 'View Results' and press Load Results).")
        return

    # Ensure final_match_rate numeric exists
    if 'final_match_rate' in detailed.columns and 'final_match_rate_percentage' not in detailed.columns:
        detailed = detailed.rename(columns={'final_match_rate': 'final_match_rate_percentage'})

    if 'final_match_rate_percentage' in detailed.columns:
        detailed['final_match_rate_percentage'] = pd.to_numeric(detailed['final_match_rate_percentage'], errors='coerce')
        maxv = detailed['final_match_rate_percentage'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0:
            detailed['final_match_rate_percentage'] = detailed['final_match_rate_percentage'] * 100.0

    # Prepare summary if missing
    if summary is None or summary.empty:
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

    # Use summary (per employee) for high-level KPIs
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

    # Visualizations: use detailed for distribution & TGV charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Match Score Distribution")
        try:
            # plot_match_distribution expects summary (per employee)
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

    # TGV Analysis - requires detailed with tgv_name and tgv_match_rate
    st.markdown("### 📊 TGV-Level Analysis")

    # build per-employee tgv summary for selection list (use tgv aggregation table if present)
    try:
        tgv_summary = detailed[['employee_id', 'fullname']].drop_duplicates().copy()
        # compute employee-level final percent from summary (already present)
        if 'final_match_rate_percentage' in summary.columns:
            tgv_summary = tgv_summary.merge(summary[['employee_id', 'final_match_rate_percentage']], on='employee_id', how='left')
        tgv_summary = tgv_summary.sort_values('final_match_rate_percentage', ascending=False).head(100)
    except Exception:
        st.error("Cannot prepare TGV employee list.")
        return

    employee_options = tgv_summary['employee_id'].tolist()
    if not employee_options:
        st.warning("No employees found in results.")
        return

    selected_employee = st.selectbox(
        "Select Employee for TGV Profile:",
        options=employee_options,
        format_func=lambda x: f"{x} - {tgv_summary[tgv_summary['employee_id']==x]['fullname'].iloc[0]} ({tgv_summary[tgv_summary['employee_id']==x]['final_match_rate_percentage'].iloc[0]:.1f}%)"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 TGV Radar Profile")
        try:
            # plot_tgv_radar expects detailed results df with tgv_name and tgv_match_rate
            fig_radar = plot_tgv_radar(detailed, selected_employee)
            st.plotly_chart(fig_radar, width='stretch')
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")

    with col2:
        st.markdown("#### 🔥 TV Heatmap (Top TGVs)")
        try:
            fig_heatmap = plot_tv_heatmap(detailed, selected_employee)
            st.plotly_chart(fig_heatmap, width='stretch')
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

    st.markdown("---")

    # Strengths & Gaps
    st.markdown("### ✅ Strengths & Gaps Analysis")
    try:
        fig_strengths_gaps = plot_strengths_gaps(detailed, selected_employee)
        st.plotly_chart(fig_strengths_gaps, width='stretch')
    except Exception as e:
        st.error(f"Error creating strengths/gaps chart: {e}")


# ================
# Main app (at bottom)
# ================
def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">🎯 Talent Match Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Talent Discovery & Succession Planning</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # replace deprecated use_column_width with explicit width
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Company+X", width=200)
        st.markdown("---")
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Select Module:",
            ["🏠 Home", "➕ Create Vacancy", "📊 View Results", "📈 Analytics"]
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This system uses AI and statistical modeling to match "
            "employees with job vacancies based on competencies, "
            "psychometric profiles, and behavioral patterns."
        )

    # Route to pages
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
