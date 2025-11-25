# app.py
# -*- coding: utf-8 -*-
"""
Talent Match Intelligence System - Single-file app with defensive derivation logic.
Author: Revised for robust derivation and debugging
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

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
# Helpers: derive/normalize
# -------------------------
def derive_tv_match_rate_from_tv_score(df: pd.DataFrame) -> pd.DataFrame:
    """Derive tv_match_rate 0..100 using min-max scaling per tv_name, reversed if lower_better."""
    df = df.copy()
    if 'tv_score' not in df.columns:
        return df

    # ensure numeric tv_score
    df['tv_score'] = pd.to_numeric(df['tv_score'], errors='coerce')

    def _scale_group(g):
        s = g['tv_score']
        mn = s.min(skipna=True)
        mx = s.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx):
            g['tv_match_rate'] = np.nan
            return g
        if mx == mn:
            # constant: map to 100 or 0 depending scoring_direction
            if g['scoring_direction'].iat[0] == 'lower_better':
                g['tv_match_rate'] = 0.0
            else:
                g['tv_match_rate'] = 100.0
            return g
        scaled = (s - mn) / (mx - mn) * 100.0
        if g['scoring_direction'].iat[0] == 'lower_better':
            scaled = 100.0 - scaled
        g['tv_match_rate'] = scaled
        return g

    if 'tv_name' in df.columns:
        df = df.groupby('tv_name', group_keys=False).apply(_scale_group)
    else:
        df = _scale_group(df)

    df['tv_match_rate'] = pd.to_numeric(df.get('tv_match_rate'), errors='coerce').clip(0,100)
    return df

def derive_tgv_from_tv(df: pd.DataFrame, aggfunc: str = 'max') -> pd.DataFrame:
    """
    Create/derive tgv_match_rate per (employee_id, tgv_name) by aggregating tv_match_rate.
    aggfunc: 'max' or 'mean'
    """
    df = df.copy()
    if 'tv_match_rate' not in df.columns:
        return df
    group = df.dropna(subset=['tv_match_rate']).groupby(['employee_id','tgv_name'], as_index=False)['tv_match_rate']
    if group._selected_obj.empty:
        return df
    if aggfunc == 'mean':
        agg = group.mean().rename(columns={'tv_match_rate': 'tgv_match_rate'})
    else:
        agg = group.max().rename(columns={'tv_match_rate': 'tgv_match_rate'})
    # merge back
    df = df.merge(agg[['employee_id','tgv_name','tgv_match_rate']], on=['employee_id','tgv_name'], how='left', suffixes=('','_derived'))
    # if original exists, prefer it; else fill from derived
    if 'tgv_match_rate' in df.columns and 'tgv_match_rate_derived' in df.columns:
        df['tgv_match_rate'] = df['tgv_match_rate'].fillna(df['tgv_match_rate_derived'])
        df = df.drop(columns=['tgv_match_rate_derived'])
    elif 'tgv_match_rate_derived' in df.columns:
        df = df.rename(columns={'tgv_match_rate_derived':'tgv_match_rate'})
    if 'tgv_match_rate' in df.columns:
        df['tgv_match_rate'] = pd.to_numeric(df['tgv_match_rate'], errors='coerce').clip(0,100)
    return df

def normalize_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize names/scale for final_match_rate_percentage, tv_match_rate, tgv_match_rate, ensure employee_id string."""
    df = df.copy()
    if 'employee_id' in df.columns:
        df['employee_id'] = df['employee_id'].astype(object).where(~df['employee_id'].isna(), other=np.nan)
        df['employee_id'] = df['employee_id'].apply(lambda x: str(x).strip() if pd.notna(x) else x)

    # final match
    if 'final_match_rate' in df.columns and 'final_match_rate_percentage' not in df.columns:
        df['final_match_rate_percentage'] = pd.to_numeric(df['final_match_rate'], errors='coerce')
    if 'final_match_rate_percentage' in df.columns:
        df['final_match_rate_percentage'] = pd.to_numeric(df['final_match_rate_percentage'], errors='coerce')
        maxv = df['final_match_rate_percentage'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0:
            df['final_match_rate_percentage'] = df['final_match_rate_percentage'] * 100.0

    # ensure tv_score numeric if exists
    if 'tv_score' in df.columns:
        df['tv_score'] = pd.to_numeric(df['tv_score'], errors='coerce')

    # ensure tv_match_rate numeric if exists
    if 'tv_match_rate' in df.columns:
        df['tv_match_rate'] = pd.to_numeric(df['tv_match_rate'], errors='coerce')
        maxv = df['tv_match_rate'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0 and maxv > 0:
            df['tv_match_rate'] = df['tv_match_rate'] * 100.0
        df['tv_match_rate'] = df['tv_match_rate'].clip(0,100)

    # ensure tgv_match_rate numeric if exists
    if 'tgv_match_rate' in df.columns:
        df['tgv_match_rate'] = pd.to_numeric(df['tgv_match_rate'], errors='coerce')
        maxv = df['tgv_match_rate'].max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0 and maxv > 0:
            df['tgv_match_rate'] = df['tgv_match_rate'] * 100.0
        df['tgv_match_rate'] = df['tgv_match_rate'].clip(0,100)

    return df

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Talent Match Intelligence", page_icon="🎯", layout="wide")
st.markdown("<style>.main-header{font-size:2.2rem;font-weight:bold;color:#1f77b4}</style>", unsafe_allow_html=True)

# session init
for k in ['vacancy_created','job_vacancy_id','matching_results_summary','matching_results_detailed','job_profile']:
    if k not in st.session_state:
        st.session_state[k] = None
st.session_state.vacancy_created = bool(st.session_state.vacancy_created)

# DB init
@st.cache_resource
def init_db():
    return DatabaseManager()
db = init_db()

# --- (kept shortened) simple navigation: Home, Create Vacancy, View Results, Analytics ---
PAGES = ["🏠 Home", "➕ Create Vacancy", "📊 View Results", "📈 Analytics"]
with st.sidebar:
    page = st.radio("Module:", PAGES)

# --- implement show_create_vacancy_page, show_results_page as in your original app ---
# For brevity I keep them minimal here and focus on Analytics flow which is critical for debug.
def show_home_page():
    st.header("Home")
    st.write("Use sidebar to navigate.")

def show_create_vacancy_page():
    st.header("Create Vacancy")
    st.write("Use existing UI from your app (unchanged).")

def show_results_page():
    st.header("View Results")
    st.write("Use existing UI from your app (unchanged).")

# ---- Analytics (critical) ----
def show_analytics_page():
    st.header("Analytics & Insights (Debuggable)")

    detailed = st.session_state.get('matching_results_detailed')
    summary = st.session_state.get('matching_results_summary')

    # If no detailed available, allow load existing
    if detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty):
        st.warning("No detailed results in session.")
        if st.button("Load most recent vacancy results"):
            try:
                recent = db.get_recent_vacancies(10)
                if recent.empty:
                    st.info("No vacancies found in DB.")
                    return
                sel = st.selectbox("Select vacancy to load", recent['job_vacancy_id'].tolist())
                with st.spinner("Loading..."):
                    df = db.run_matching_query(int(sel))
                    df = normalize_match_columns(df)
                    # derive tv_match_rate from tv_score if needed
                    if ('tv_match_rate' not in df.columns or df['tv_match_rate'].isna().all()) and 'tv_score' in df.columns:
                        df = derive_tv_match_rate_from_tv_score(df)
                    # derive tgv
                    df = derive_tgv_from_tv(df, aggfunc='max')
                    st.session_state.matching_results_detailed = df
                    # build summary
                    summ = df.groupby('employee_id').agg({
                        'fullname':'first','directorate':'first','role':'first','grade':'first',
                        'final_match_rate_percentage':'first'
                    }).reset_index()
                    st.session_state.matching_results_summary = summ
                    st.success("Loaded and normalized results.")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Load failed: {e}")
        return

    # ensure normalized and derive if needed (idempotent)
    detailed = normalize_match_columns(detailed)
    if ('tv_match_rate' not in detailed.columns or detailed['tv_match_rate'].isna().all()) and 'tv_score' in detailed.columns:
        detailed = derive_tv_match_rate_from_tv_score(detailed)
    if ('tgv_match_rate' not in detailed.columns) or detailed['tgv_match_rate'].isna().all():
        detailed = derive_tgv_from_tv(detailed, aggfunc='max')
    st.session_state.matching_results_detailed = detailed

    # Debug expander
    with st.expander("🔍 Debug data frame info"):
        st.write("Detailed shape:", detailed.shape)
        st.write("Columns:", list(detailed.columns))
        for col in ['tv_score','tv_match_rate','tgv_match_rate','final_match_rate_percentage']:
            if col in detailed.columns:
                cnt = int(detailed[col].notna().sum())
                mn = detailed[col].min(skipna=True)
                mx = detailed[col].max(skipna=True)
                st.write(f"- {col}: {cnt} non-null, min/max {mn}/{mx}")
            else:
                st.write(f"- {col}: MISSING")
        st.dataframe(detailed.head(10))

    # Build summary if missing
    if summary is None or summary.empty:
        summary = detailed.groupby('employee_id').agg({
            'fullname':'first','directorate':'first','role':'first','grade':'first','final_match_rate_percentage':'first'
        }).reset_index()
        st.session_state.matching_results_summary = summary

    # Key metrics + plots (use your visualizations)
    st.subheader("Key Insights")
    if 'final_match_rate_percentage' in summary.columns:
        st.metric("Average match", f"{summary['final_match_rate_percentage'].mean():.1f}%")
    try:
        fig = plot_match_distribution(summary)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Distribution error: {e}")

    # TGV-level analysis (safe)
    if 'tgv_match_rate' not in detailed.columns or detailed['tgv_match_rate'].isna().all():
        st.error("❌ All tgv_match_rate values are NULL (no tv_match_rate available to aggregate)")
        st.info("If this persists, inspect DB baseline tables as described below.")
        return

    # selection and show charts
    # build tgv_summary for selection
    tgv_summary = (detailed[['employee_id','fullname']]
                   .drop_duplicates()
                   .merge(summary[['employee_id','final_match_rate_percentage']], on='employee_id', how='left'))
    tgv_summary = tgv_summary.sort_values('final_match_rate_percentage', ascending=False).head(200)
    selected_employee = st.selectbox("Select employee", tgv_summary['employee_id'].tolist(),
                                     format_func=lambda x: f"{x} - {tgv_summary[tgv_summary['employee_id']==x]['fullname'].iloc[0]}")
    # charts
    try:
        st.plotly_chart(plot_tgv_radar(detailed, selected_employee), use_container_width=True)
    except Exception as e:
        st.error(f"Radar error: {e}")
    try:
        st.plotly_chart(plot_tv_heatmap(detailed, selected_employee), use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap error: {e}")
    try:
        st.plotly_chart(plot_strengths_gaps(detailed, selected_employee), use_container_width=True)
    except Exception as e:
        st.error(f"Strengths/Gaps error: {e}")


# Main router
if page == "🏠 Home":
    show_home_page()
elif page == "➕ Create Vacancy":
    show_create_vacancy_page()
elif page == "📊 View Results":
    show_results_page()
elif page == "📈 Analytics":
    show_analytics_page()
