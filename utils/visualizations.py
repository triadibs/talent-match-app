# utils/visualizations.py
"""
Robust visualization helpers for Talent Match Intelligence System.

Functions:
- plot_match_distribution(summary_df)
- plot_top_candidates(summary_df, top_n=10)
- plot_tgv_radar(detailed_df, employee_id)
- plot_tv_heatmap(detailed_df, employee_id)
- plot_strengths_gaps(detailed_df, employee_id)

Each function validates inputs and coerces numeric columns, raising clear exceptions
that the Streamlit app can catch and show to the user.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List

# Utility helpers
def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce listed columns to numeric, inplace. Non-coercible -> NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"results_df must contain columns: {set(required)} (missing: {missing})")

# 1) Distribution of match scores (summary must have final_match_rate_percentage)
def plot_match_distribution(summary_df: pd.DataFrame):
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty")

    # Normalize column name
    if 'final_match_rate' in summary_df.columns and 'final_match_rate_percentage' not in summary_df.columns:
        summary_df = summary_df.rename(columns={'final_match_rate': 'final_match_rate_percentage'})

    _ensure_columns(summary_df, ['employee_id', 'final_match_rate_percentage'])

    df = summary_df.copy()
    df['final_match_rate_percentage'] = pd.to_numeric(df['final_match_rate_percentage'], errors='coerce')
    df = df.dropna(subset=['final_match_rate_percentage'])
    if df.empty:
        raise ValueError("No numeric final_match_rate_percentage values to plot")

    # Clip to 0..100
    df['final_match_rate_percentage'] = df['final_match_rate_percentage'].clip(lower=0, upper=100)

    fig = px.histogram(df, x='final_match_rate_percentage', nbins=20, title="Match Score Distribution",
                       labels={'final_match_rate_percentage': 'Match Score (%)'})
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

# 2) Top candidates bar chart
def plot_top_candidates(summary_df: pd.DataFrame, top_n: int = 10):
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty")

    if 'final_match_rate_percentage' in summary_df.columns:
        summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df['final_match_rate_percentage'], errors='coerce')
    else:
        # try alternative names
        for alt in ['final_match_rate', 'final_match']:
            if alt in summary_df.columns:
                summary_df['final_match_rate_percentage'] = pd.to_numeric(summary_df[alt], errors='coerce')
                break

    df = summary_df.dropna(subset=['final_match_rate_percentage']).nlargest(top_n, 'final_match_rate_percentage')
    if df.empty:
        raise ValueError("No numeric final_match_rate_percentage values for top candidates")

    fig = px.bar(df, x='final_match_rate_percentage', y='fullname', orientation='h',
                 labels={'final_match_rate_percentage': 'Match Score (%)', 'fullname': 'Candidate'},
                 title=f"Top {top_n} Candidates")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=120, r=10, t=40, b=10))
    return fig

# 3) TGV radar profile for one employee (requires tgv_name, tgv_match_rate)
def plot_tgv_radar(detailed_df: pd.DataFrame, employee_id: str):
    _ensure_columns(detailed_df, ['employee_id', 'tgv_name', 'tgv_match_rate'])
    df = detailed_df.copy()
    df = _coerce_numeric(df, ['tgv_match_rate'])
    # select employee
    emp = df[df['employee_id'] == employee_id]
    if emp.empty:
        raise ValueError(f"No TGV rows for employee_id {employee_id}")

    # aggregate in case multiple rows per tgv_name
    agg = emp.groupby('tgv_name', as_index=False)['tgv_match_rate'].mean()
    # drop NaN and sort
    agg = agg.dropna(subset=['tgv_match_rate'])
    if agg.empty:
        raise ValueError("No numeric tgv_match_rate values found for this employee")

    # radar requires closed polygon; use plotly go
    categories = agg['tgv_name'].tolist()
    values = agg['tgv_match_rate'].tolist()
    # ensure values are numeric
    values = [float(v) for v in values]
    # close the polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(r=values, theta=categories, fill='toself', name=str(employee_id))
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
        showlegend=False,
        title=f"TGV Radar: {employee_id}"
    )
    return fig

# 4) TV heatmap for selected employee (requires tv_name, tv_match_rate, tgv_name)
def plot_tv_heatmap(detailed_df: pd.DataFrame, employee_id: str):
    _ensure_columns(detailed_df, ['employee_id', 'tgv_name', 'tv_name', 'tv_match_rate'])
    df = detailed_df.copy()
    df = _coerce_numeric(df, ['tv_match_rate'])
    emp = df[df['employee_id'] == employee_id]
    if emp.empty:
        raise ValueError(f"No TV rows for employee_id {employee_id}")

    emp = emp.dropna(subset=['tv_match_rate'])
    if emp.empty:
        raise ValueError("No numeric tv_match_rate values for this employee")

    # pivot table tgv_name x tv_name with mean tv_match_rate
    pivot = emp.pivot_table(index='tgv_name', columns='tv_name', values='tv_match_rate', aggfunc='mean', fill_value=np.nan)
    if pivot.size == 0:
        raise ValueError("Pivot table is empty")

    # plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns.astype(str)),
        y=list(pivot.index.astype(str)),
        colorbar=dict(title='Match %'),
        zmin=0, zmax=100
    ))
    fig.update_layout(title=f"TV Heatmap: {employee_id}", xaxis_nticks=20, margin=dict(l=120, r=20, t=40, b=80))
    return fig

# 5) Strengths & gaps: return bar with top positive and negative deltas vs baseline (requires tv_match_rate, baseline_score)
def plot_strengths_gaps(detailed_df: pd.DataFrame, employee_id: str, top_n: int = 6):
    _ensure_columns(detailed_df, ['employee_id', 'tv_name', 'tv_match_rate', 'baseline_score'])
    df = detailed_df.copy()
    df = _coerce_numeric(df, ['tv_match_rate', 'baseline_score'])
    emp = df[df['employee_id'] == employee_id]
    if emp.empty:
        raise ValueError(f"No TV rows for employee_id {employee_id}")

    emp = emp.dropna(subset=['tv_match_rate', 'baseline_score'])
    if emp.empty:
        raise ValueError("No TV rows with both tv_match_rate and baseline_score numeric")

    # delta = tv_match_rate - baseline_normalized (baseline may be in same scale as user_score)
    emp['delta'] = emp['tv_match_rate'] - emp['baseline_score']
    # aggregate by tv_name
    agg = emp.groupby('tv_name', as_index=False)['delta'].mean()
    if agg.empty:
        raise ValueError("No numeric deltas to show")

    top_pos = agg.nlargest(top_n, 'delta')
    top_neg = agg.nsmallest(top_n, 'delta')

    fig = go.Figure()
    if not top_pos.empty:
        fig.add_trace(go.Bar(x=top_pos['delta'], y=top_pos['tv_name'], orientation='h', name='Strengths'))
    if not top_neg.empty:
        fig.add_trace(go.Bar(x=top_neg['delta'], y=top_neg['tv_name'], orientation='h', name='Gaps'))
    fig.update_layout(title=f"Strengths & Gaps: {employee_id}", barmode='relative', margin=dict(l=180, r=20, t=40, b=40))
    return fig
