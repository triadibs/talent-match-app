# -*- coding: utf-8 -*-
"""
Talent Match Intelligence System - Step 3
AI-Powered Talent Matching Dashboard

Author: TRI ADI BASKORO (revised)
Date: 18 November 2025 (patched)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Custom modules
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
# Utility helpers (normalization / derivation)
# -------------------------
def derive_tv_match_from_baseline_row(row):
    """Mimic SQL logic to derive tv_match_rate from user_score & baseline_mean/stddev."""
    try:
        us = float(row.get('user_score'))
        bm = float(row.get('baseline_mean'))
    except Exception:
        return np.nan

    sd = row.get('baseline_stddev')
    try:
        sd = float(sd) if pd.notna(sd) else np.nan
    except Exception:
        sd = np.nan

    dirn = row.get('scoring_direction', 'higher_better') or 'higher_better'

    # If baseline mean missing -> cannot derive
    if pd.isna(bm):
        return np.nan

    # If stddev available and non-zero -> sigmoid-like mapping
    if pd.notna(sd) and sd != 0:
        val = 100.0 * (1.0 / (1.0 + np.exp(-(us - bm) / sd)))
        return float(np.clip(val, 0.0, 100.0))

    # Fallback ratio rules
    if dirn == 'lower_better':
        if bm == 0:
            return np.nan
        val = ((2.0 * bm - us) / bm) * 100.0
    else:
        if bm == 0:
            return np.nan
        val = (us / bm) * 100.0

    return float(np.clip(val, 0.0, 100.0))


def normalize_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric columns, derive missing tv_match_rate from baseline/user_score,
    and derive tgv_match_rate (agg) if missing.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Normalize employee_id & string cols
    if 'employee_id' in df.columns:
        df['employee_id'] = df['employee_id'].astype(object).where(~df['employee_id'].isna(), other=np.nan)
        df['employee_id'] = df['employee_id'].apply(lambda x: str(x).strip() if pd.notna(x) else x)

    for col in ['tv_score','user_score','baseline_mean','baseline_stddev','tv_match_rate','tgv_match_rate','final_match_rate','final_match_rate_percentage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # If tv_match_rate is entirely missing (all NaN) but baseline+user exist, derive it
    need_tv_derive = ('tv_match_rate' not in df.columns) or (df['tv_match_rate'].isna().all())
    if need_tv_derive and ('baseline_mean' in df.columns or 'user_score' in df.columns):
        # Ensure columns exist
        if 'tv_match_rate' not in df.columns:
            df['tv_match_rate'] = np.nan
        mask = df['tv_match_rate'].isna() & df['baseline_mean'].notna() & df['user_score'].notna()
        if mask.any():
            df.loc[mask, 'tv_match_rate'] = df[mask].apply(derive_tv_match_from_baseline_row, axis=1)

    # If still missing tv_match_rate but tv_score exists, derive via min-max per tv_name
    if ('tv_match_rate' not in df.columns or df['tv_match_rate'].isna().all()) and 'tv_score' in df.columns:
        # min-max scaling per tv_name
        def _scale_group(g):
            s = g['tv_score'].astype(float)
            if s.isna().all():
                g['tv_match_rate'] = np.nan
                return g
            mn = s.min()
            mx = s.max()
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                # if constant, map higher_better -> 100, lower_better -> 0
                if 'scoring_direction' in g.columns and g['scoring_direction'].iloc[0] == 'lower_better':
                    g['tv_match_rate'] = 0.0
                else:
                    g['tv_match_rate'] = 100.0
                return g
            scaled = (s - mn) / (mx - mn) * 100.0
            if 'scoring_direction' in g.columns and g['scoring_direction'].iloc[0] == 'lower_better':
                scaled = 100.0 - scaled
            g['tv_match_rate'] = scaled
            return g

        if 'tv_name' in df.columns:
            df = df.groupby('tv_name', group_keys=False).apply(_scale_group)
        else:
            df = _scale_group(df)

    # Derive tgv_match_rate if missing by aggregating tv_match_rate per (employee_id, tgv_name)
    need_tgv = ('tgv_match_rate' not in df.columns) or (df['tgv_match_rate'].isna().all())
    if need_tgv:
        if 'tv_match_rate' in df.columns:
            agg = (df.dropna(subset=['tv_match_rate'])
                   .groupby(['employee_id','tgv_name'], as_index=False)['tv_match_rate']
                   .agg(['max','mean']).reset_index())
            if not agg.empty:
                agg['tgv_match_rate_derived'] = agg['max'].combine_first(agg['mean'])
                to_merge = agg[['employee_id','tgv_name','tgv_match_rate_derived']].rename(columns={'tgv_match_rate_derived':'tgv_match_rate'})
                # merge back
                df = df.merge(to_merge, on=['employee_id','tgv_name'], how='left')
                # if original exists prefer it
                if 'tgv_match_rate' in df.columns:
                    df['tgv_match_rate'] = df['tgv_match_rate'].fillna(df.get('tgv_match_rate_derived'))
                else:
                    df = df.rename(columns={'tgv_match_rate_derived':'tgv_match_rate'})
                if 'tgv_match_rate_derived' in df.columns:
                    df = df.drop(columns=['tgv_match_rate_derived'])

    # Final numeric coercions & clipping
    for col in ['tv_match_rate','tgv_match_rate','final_match_rate','final_match_rate_percentage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(lower=0.0, upper=100.0)

    return df


# -------------------------
# Page config & UI setup
# -------------------------
st.set_page_config(
    page_title="Talent Match Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

def generate_fallback_profile(role_name: str, job_level: str, role_purpose: str) -> dict:
    return {
        "requirements": f"Experience in {role_name}, level: {job_level}. {role_purpose}",
        "description": f"{role_name} — {job_level}. {role_purpose}",
        "competencies": ["Analytical thinking","Problem solving","Communication","Domain knowledge"]
    }

# Session state
for key in ['vacancy_created','job_vacancy_id','matching_results_summary','matching_results_detailed','job_profile']:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.get('vacancy_created') is None:
    st.session_state.vacancy_created = False

# DB init
@st.cache_resource
def init_database():
    return DatabaseManager()
db = init_database()

# -------------------------
# Helper to fetch and normalize detailed results (used in multiple places)
# -------------------------
def fetch_and_prepare_detailed(vacancy_id: int) -> pd.DataFrame:
    """Fetch detailed rows from DB and normalize + derive missing scores."""
    df = db.run_matching_query(vacancy_id)
    if df is None:
        return pd.DataFrame()
    # QUICK DEBUG: print heads into Streamlit logs
    st.write(f"Fetched detailed rows: {len(df)}")
    df = normalize_and_derive(df)
    # store back
    st.session_state.matching_results_detailed = df
    return df

# -------------------------
# Pages (home/create/results/analytics) - keep structure minimal for brevity
# -------------------------
def show_home_page():
    st.markdown("## Home")
    try:
        st.metric("Total Employees", f"{db.get_total_employees():,}")
    except Exception:
        st.write("DB error")

def show_create_vacancy_page():
    st.markdown("## Create Vacancy")
    # reuse your existing create flow (omitted here for brevity)
    st.info("Create Vacancy UI preserved in your original app. Use 'View Results' to load and inspect.")

def show_results_page():
    st.markdown("## View Results")
    existing = db.get_recent_vacancies(20)
    if existing is None or existing.empty:
        st.info("No vacancies in DB.")
        return
    selected = st.selectbox("Select vacancy:", options=existing['job_vacancy_id'].tolist(),
                             format_func=lambda x: f"ID {x}: {existing[existing['job_vacancy_id']==x]['role_name'].iloc[0]}")
    if st.button("📥 Load Results"):
        with st.spinner("Loading..."):
            df = fetch_and_prepare_detailed(int(selected))
            if df is None or df.empty:
                st.error("No detailed rows returned.")
                return
            # build summary
            tmp = df.copy()
            if 'final_match_rate_percentage' not in tmp.columns and 'final_match_rate' in tmp.columns:
                tmp['final_match_rate_percentage'] = tmp['final_match_rate']
            if 'final_match_rate_percentage' not in tmp.columns:
                for c in ['final_match_rate','tgv_match_rate','tv_match_rate']:
                    if c in tmp.columns and pd.api.types.is_numeric_dtype(tmp[c]):
                        tmp['final_match_rate_percentage'] = tmp[c]
                        break
            if 'final_match_rate_percentage' in tmp.columns and tmp['final_match_rate_percentage'].max(skipna=True) <= 1.0:
                tmp['final_match_rate_percentage'] = tmp['final_match_rate_percentage'] * 100.0
            summary = tmp.groupby('employee_id').agg({
                'fullname':'first','directorate':'first','role':'first','grade':'first','final_match_rate_percentage':'first'
            }).reset_index()
            summary['final_match_rate_percentage'] = pd.to_numeric(summary['final_match_rate_percentage'], errors='coerce')
            st.session_state.matching_results_summary = summary
            st.success(f"Loaded vacancy {selected} with {len(summary)} summary rows.")
            st.experimental_rerun()

def show_analytics_page():
    st.markdown("## 📈 Analytics & Insights (Debuggable)")
    detailed = st.session_state.get('matching_results_detailed', None)
    summary = st.session_state.get('matching_results_summary', None)
    if detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty):
        st.warning("No detailed results loaded. Use View Results to load a vacancy.")
        return

    # DEBUG expander
    with st.expander("🔍 Debug data frame info", expanded=True):
        st.write(f"Detailed shape: {detailed.shape}")
        st.write("Columns:")
        st.write(list(detailed.columns))
        for col in ['tv_score','user_score','baseline_mean','baseline_stddev','tv_match_rate','tgv_match_rate','final_match_rate_percentage']:
            if col in detailed.columns:
                non_null = int(detailed[col].notna().sum())
                mn = detailed[col].min(skipna=True)
                mx = detailed[col].max(skipna=True)
                st.write(f"- {col}: {non_null} non-null (min/max: {mn}/{mx})")
            else:
                st.write(f"- {col}: MISSING")

        st.write("Sample rows (first 5):")
        st.dataframe(detailed.head(5))

    # Ensure numeric and derive if needed (idempotent)
    detailed = normalize_and_derive(detailed)
    st.session_state.matching_results_detailed = detailed

    # Build summary if missing
    if summary is None or summary.empty:
        summary = detailed.groupby('employee_id').agg({
            'fullname':'first','directorate':'first','role':'first','grade':'first','final_match_rate_percentage':'first'
        }).reset_index()
        summary['final_match_rate_percentage'] = pd.to_numeric(summary['final_match_rate_percentage'], errors='coerce')
        st.session_state.matching_results_summary = summary

    # Key insights
    if summary['final_match_rate_percentage'].dropna().empty:
        st.error("No valid final match scores available.")
    else:
        avg_match = summary['final_match_rate_percentage'].mean()
        st.metric("Avg Match Rate", f"{avg_match:.1f}%")
        top_match = summary['final_match_rate_percentage'].max()
        st.metric("Top Match", f"{top_match:.1f}%")

    st.markdown("---")
    # Distribution & Top
    try:
        fig_dist = plot_match_distribution(summary)
        st.plotly_chart(fig_dist, use_container_width=True)
    except Exception as e:
        st.error(f"Distribution error: {e}")

    try:
        fig_top = plot_top_candidates(summary, top_n=10)
        st.plotly_chart(fig_top, use_container_width=True)
    except Exception as e:
        st.error(f"Top candidates error: {e}")

    st.markdown("---")
    # TGV analysis — requires tgv_match_rate available
    if 'tgv_match_rate' not in detailed.columns or detailed['tgv_match_rate'].isna().all():
        st.error("❌ All tgv_match_rate values are NULL (no tv_match_rate available to aggregate)")
        st.info("Hint: ensure DB pipeline returns tv_match_rate OR use fallback derivation (app attempts this automatically when possible).")
        return

    # prepare employee selector
    tgv_summary = (detailed[['employee_id','fullname','tgv_name','tgv_match_rate']]
                   .groupby(['employee_id','fullname'], as_index=False)['tgv_match_rate']
                   .mean())
    if 'final_match_rate_percentage' in summary.columns:
        tgv_summary = tgv_summary.merge(summary[['employee_id','final_match_rate_percentage']], on='employee_id', how='left')
    tgv_summary = tgv_summary.sort_values('final_match_rate_percentage', ascending=False).head(200)
    if tgv_summary.empty:
        st.warning("No employees with TGV data found.")
        return

    employee_options = tgv_summary['employee_id'].tolist()
    selected_employee = st.selectbox("Select Employee for TGV Profile:", options=employee_options,
                                     format_func=lambda x: f"{x} - {tgv_summary[tgv_summary['employee_id']==x]['fullname'].iloc[0]} ({tgv_summary[tgv_summary['employee_id']==x]['final_match_rate_percentage'].iloc[0]:.1f}%)")

    # Show charts
    try:
        fig_radar = plot_tgv_radar(detailed, selected_employee)
        st.plotly_chart(fig_radar, use_container_width=True)
    except Exception as e:
        st.error(f"Radar error: {e}")

    try:
        fig_heatmap = plot_tv_heatmap(detailed, selected_employee)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap error: {e}")

    try:
        fig_strengths = plot_strengths_gaps(detailed, selected_employee)
        st.plotly_chart(fig_strengths, use_container_width=True)
    except Exception as e:
        st.error(f"Strengths/gaps error: {e}")

# Main routing (minimal to focus on analytics)
def main():
    st.markdown('<div class="main-header">🎯 Talent Match Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Talent Discovery & Succession Planning</div>', unsafe_allow_html=True)
    with st.sidebar:
        page = st.radio("Select Module:", ["🏠 Home", "📊 View Results", "📈 Analytics"])
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 View Results":
        show_results_page()
    elif page == "📈 Analytics":
        show_analytics_page()

if __name__ == "__main__":
    main()
