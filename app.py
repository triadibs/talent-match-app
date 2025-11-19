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

    if not recent_vacancies.empty:
        # Use width='stretch' as replacement for use_container_width
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
            
            if high_performers_df.empty:
                st.error("No high performers found!")
                selected_employees = []
            else:
                employee_options = high_performers_df.apply(
                    lambda x: f"{x['employee_id']} - {x['fullname']} ({x['position']})", 
                    axis=1
                ).tolist()
                
                selected_options = st.multiselect(
                    "Select Benchmark Employees *",
                    options=employee_options,
                    max_selections=5
                )
                selected_employees = [opt.split(' - ')[0] for opt in selected_options]
        
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
                    job_profile = generate_job_profile(
                        role_name=role_name,
                        job_level=job_level,
                        role_purpose=role_purpose,
                        benchmark_employees=high_performers_df[
                            high_performers_df['employee_id'].isin(selected_employees)
                        ]
                    )
                except Exception as e:
                    st.warning(f"AI generation failed: {e}. Using fallback.")
                    job_profile = generate_fallback_profile(role_name, job_level, role_purpose)
            
            # 3. Run matching SQL
            with st.spinner("📊 Computing talent matches (this may take 30-60 seconds)..."):
                try:
                    matching_results = db.run_matching_query(vacancy_id)
                    
                    if matching_results.empty:
                        st.error("❌ Matching returned no results. Check your data.")
                        st.stop()
                    
                    st.success(f"✅ Matched {len(matching_results)} records!")
                    
                except Exception as e:
                    st.error(f"❌ Matching error: {e}")
                    st.exception(e)
                    st.stop()
            
            # 4. Save to session state
            st.session_state.vacancy_created = True
            st.session_state.job_vacancy_id = vacancy_id
            st.session_state.matching_results = matching_results
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
    """Page showing matching results"""
    
    st.markdown("## 📊 Talent Matching Results")
    
    # Check session state
    if not st.session_state.get('vacancy_created', False):
        st.warning("⚠️ No vacancy created yet. Create one or load existing.")
        
        # Load existing vacancy
        existing = db.get_recent_vacancies(20)
        if not existing.empty:
            selected = st.selectbox("Select vacancy:", existing['job_vacancy_id'],
                                   format_func=lambda x: f"ID {x}: {existing[existing['job_vacancy_id']==x]['role_name'].iloc[0]}")
            
            if st.button("📥 Load Results"):
                with st.spinner("Loading..."):
                    st.session_state.job_vacancy_id = selected
                    st.session_state.matching_results = db.run_matching_query(selected)
                    st.session_state.vacancy_created = True
                    st.rerun()
        return
    
    # Get data
    results_df = st.session_state.matching_results
    vacancy_id = st.session_state.job_vacancy_id
    
    if results_df is None or results_df.empty:
        st.error("No results available.")
        return
    
    # Create summary
    summary_df = results_df.groupby('employee_id').agg({
        'fullname': 'first',
        'directorate': 'first',
        'role': 'first',
        'grade': 'first',
        'final_match_rate': 'first'
    }).reset_index()
    
    summary_df = summary_df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})
    summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df['final_match_rate_percentage'], errors='coerce')
    summary_df = summary_df.dropna(subset=['final_match_rate_percentage'])
    summary_df = summary_df.sort_values('final_match_rate_percentage', ascending=False)
    
    # Display
    st.markdown("### 🏆 Top 20 Candidates")
    
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
    
    # Download
    csv = summary_df.to_csv(index=False)
    st.download_button(
        "📥 Download Results (CSV)",
        csv,
        f"talent_match_results_{vacancy_id}.csv",
        "text/csv"
    )

def show_analytics_page():
    """Page with visualizations and analytics"""
    st.markdown("## 📈 Analytics & Insights")

    # Check if results available
    if not st.session_state.vacancy_created or st.session_state.matching_results is None:
        st.warning("⚠️ No results available. Please create a vacancy first.")
        return

    results_df = st.session_state.matching_results
    vacancy_id = st.session_state.job_vacancy_id

    # COMPUTE summary from detailed results
    # handle both column name variants safely
    if 'final_match_rate' not in results_df.columns and 'final_match_rate_percentage' in results_df.columns:
        results_df['final_match_rate'] = pd.to_numeric(results_df['final_match_rate_percentage'], errors='coerce')

    summary_df = results_df.groupby('employee_id').agg({
        'fullname': 'first',
        'directorate': 'first',
        'role': 'first',
        'grade': 'first',
        'final_match_rate': 'first'
    }).reset_index()

    summary_df = summary_df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})

    # Convert to numeric and drop NaN
    summary_df['final_match_rate_percentage'] = pd.to_numeric(
        summary_df['final_match_rate_percentage'],
        errors='coerce'
    )
    summary_df = summary_df.dropna(subset=['final_match_rate_percentage'])

    summary_df = summary_df.sort_values('final_match_rate_percentage', ascending=False).head(100)

    # Check if we have data
    if summary_df.empty:
        st.warning("⚠️ No valid matching results found.")
        return

    # Key insights
    st.markdown("### 🔍 Key Insights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_match = summary_df['final_match_rate_percentage'].mean()
        st.metric("Avg Match Rate", f"{avg_match:.1f}%")

    with col2:
        top_match = summary_df['final_match_rate_percentage'].max()
        st.metric("Top Match", f"{top_match:.1f}%")

    with col3:
        candidates_above_70 = (summary_df['final_match_rate_percentage'] >= 70).sum()
        st.metric("Matches ≥70%", candidates_above_70)

    with col4:
        candidates_above_80 = (summary_df['final_match_rate_percentage'] >= 80).sum()
        st.metric("Matches ≥80%", candidates_above_80)

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Match Score Distribution")
        try:
            fig_dist = plot_match_distribution(summary_df)
            st.plotly_chart(fig_dist, width='stretch')
        except Exception as e:
            st.error(f"Error creating distribution chart: {e}")

    with col2:
        st.markdown("#### 🏆 Top 10 Candidates")
        try:
            fig_top = plot_top_candidates(summary_df, top_n=10)
            st.plotly_chart(fig_top, width='stretch')
        except Exception as e:
            st.error(f"Error creating top candidates chart: {e}")

    st.markdown("---")

    # TGV Analysis
    st.markdown("### 📊 TGV-Level Analysis")

    # Select employee for radar chart
    employee_options = summary_df.head(20)['employee_id'].tolist()

    if not employee_options:
        st.warning("No employees found in results.")
        return

    selected_employee = st.selectbox(
        "Select Employee for TGV Profile:",
        options=employee_options,
        format_func=lambda x: f"{x} - {summary_df[summary_df['employee_id']==x]['fullname'].iloc[0]} ({summary_df[summary_df['employee_id']==x]['final_match_rate_percentage'].iloc[0]:.1f}%)"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 TGV Radar Profile")
        try:
            fig_radar = plot_tgv_radar(results_df, selected_employee)
            st.plotly_chart(fig_radar, width='stretch')
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")

    with col2:
        st.markdown("#### 🔥 TV Heatmap (Top TGVs)")
        try:
            fig_heatmap = plot_tv_heatmap(results_df, selected_employee)
            st.plotly_chart(fig_heatmap, width='stretch')
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")

    st.markdown("---")

    # Strengths & Gaps
    st.markdown("### ✅ Strengths & Gaps Analysis")

    try:
        fig_strengths_gaps = plot_strengths_gaps(results_df, selected_employee)
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
