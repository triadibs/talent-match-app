# utils/visualizations.py
# -*- coding: utf-8 -*-
"""
Robust visualization helpers for Talent Match Intelligence System.

This version:
- coerces columns to numeric safely
- avoids deprecated DataFrame.append (uses pd.concat)
- provides clear ValueError messages for missing/invalid data
- tolerates string-numeric inputs (e.g. '99.5') and converts to float
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

def _coerce_series_to_numeric(s: pd.Series) -> pd.Series:
    """Safely coerce a Series to numeric; return Series (float dtype) with NaN on failure."""
    if s is None:
        return s
    # first try direct to_numeric (handles strings with numeric content)
    try:
        out = pd.to_numeric(s, errors='coerce')
    except Exception:
        # fallback: convert to str then attempt
        out = pd.to_numeric(s.astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True), errors='coerce')
    return out

def _ensure_percent_scale(s: pd.Series) -> pd.Series:
    """
    Ensure series is on 0-100 percent scale.
    If values appear to be in 0-1 range (max <= 1.0), multiply by 100.
    Non-numeric values become NaN.
    """
    if s is None:
        return s
    s_num = _coerce_series_to_numeric(s)
    if s_num is None or s_num.empty:
        return s_num
    maxv = s_num.max(skipna=True)
    if pd.isna(maxv):
        return s_num
    if maxv <= 1.0:
        return s_num * 100.0
    return s_num

def _ensure_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"results_df must contain columns: {set(required)} (missing: {missing})")

# -------------------------
# Plot helpers
# -------------------------
def plot_match_distribution(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty")

    if 'final_match_rate_percentage' not in summary_df.columns:
        raise ValueError("summary_df must contain column 'final_match_rate_percentage'")

    x_series = _ensure_percent_scale(summary_df['final_match_rate_percentage'])
    x_series = x_series.dropna()
    if x_series.empty:
        raise ValueError("No numeric final_match_rate_percentage values to plot")

    # Clip 0..100
    x_series = x_series.clip(0, 100)

    fig = px.histogram(x=x_series, nbins=20, title="Match Score Distribution")
    mean_val = float(x_series.mean())
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.1f}%", annotation_position="top")
    fig.update_layout(xaxis_title="Match Rate (%)", yaxis_title="Number of Candidates", showlegend=False, height=400)
    return fig

def plot_top_candidates(summary_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    for col in ['final_match_rate_percentage', 'fullname', 'employee_id']:
        if col not in summary_df.columns:
            raise ValueError(f"summary_df must contain column '{col}'")

    df = summary_df.copy()
    df['final_match_rate_percentage'] = _ensure_percent_scale(df['final_match_rate_percentage'])
    df = df.dropna(subset=['final_match_rate_percentage'])
    if df.empty:
        raise ValueError("No numeric final_match_rate_percentage values for top candidates")

    top_df = df.nlargest(top_n, 'final_match_rate_percentage').copy()
    top_df = top_df.sort_values('final_match_rate_percentage', ascending=True)

    # build label defensively
    def _label(row):
        name = row.get('fullname', '')
        role = row.get('role', None)
        pct = row.get('final_match_rate_percentage', np.nan)
        try:
            pctf = float(pct)
            pct_str = f"{pctf:.1f}%"
        except Exception:
            pct_str = ""
        if pd.notna(role) and role:
            return f"{name} — {role} ({pct_str})"
        return f"{name} ({pct_str})"

    top_df['label'] = top_df.apply(_label, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_df['final_match_rate_percentage'],
        y=top_df['label'],
        orientation='h',
        text=top_df['final_match_rate_percentage'].apply(lambda x: f"{float(x):.1f}%" if pd.notna(x) else ""),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Match: %{x:.1f}%<extra></extra>"
    ))
    fig.update_layout(title=f"Top {len(top_df)} Candidates", xaxis_title="Match Rate (%)",
                      yaxis_title="Candidate", height=400 + 30 * min(len(top_df), top_n),
                      margin=dict(l=220 if len(top_df) > 6 else 120, r=40, t=60, b=40))
    return fig

def plot_tgv_radar(results_df: pd.DataFrame, employee_id: str) -> go.Figure:
    _ensure_columns(results_df, ['employee_id', 'tgv_name', 'tgv_match_rate'])
    emp = results_df[results_df['employee_id'] == employee_id].copy()
    if emp.empty:
        raise ValueError(f"No TGV rows for employee_id {employee_id}")

    emp['tgv_match_rate'] = _ensure_percent_scale(emp['tgv_match_rate'])
    emp = emp.dropna(subset=['tgv_match_rate'])
    if emp.empty:
        raise ValueError("No numeric tgv_match_rate values found for this employee")

    agg = emp.groupby('tgv_name', as_index=False)['tgv_match_rate'].mean().sort_values('tgv_match_rate', ascending=False)
    categories = agg['tgv_name'].tolist()
    values = agg['tgv_match_rate'].astype(float).tolist()

    if len(categories) == 1:
        categories = categories * 2
        values = values * 2
    else:
        categories = categories + [categories[0]]
        values = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name=str(employee_id)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], visible=True)), showlegend=False,
                      title=f"TGV Radar Profile — Employee {employee_id}", height=480)
    return fig

def plot_tv_heatmap(results_df: pd.DataFrame, employee_id: str, top_tgv_count: int = 4) -> go.Figure:
    required = {'employee_id', 'tgv_name', 'tv_name', 'tv_match_rate'}
    if not required.issubset(set(results_df.columns)):
        raise ValueError(f"results_df must contain columns: {required}")

    emp = results_df[results_df['employee_id'] == employee_id].copy()
    if emp.empty:
        raise ValueError(f"No rows for employee_id={employee_id}")

    emp['tv_match_rate'] = _ensure_percent_scale(emp['tv_match_rate'])
    emp = emp.dropna(subset=['tv_match_rate'])
    if emp.empty:
        raise ValueError("No numeric tv_match_rate values for this employee")

    if 'tgv_match_rate' in emp.columns:
        tgv_rank = emp.groupby('tgv_name', as_index=False)['tgv_match_rate'].max().sort_values('tgv_match_rate', ascending=False)
    else:
        tgv_rank = emp.groupby('tgv_name', as_index=False)['tv_match_rate'].mean().sort_values('tv_match_rate', ascending=False)

    top_tgvs = tgv_rank['tgv_name'].head(top_tgv_count).tolist()
    df_plot = emp[emp['tgv_name'].isin(top_tgvs)].copy()
    if df_plot.empty:
        raise ValueError("No TV rows after filtering for top TGVs.")

    pivot = df_plot.pivot_table(index='tv_name', columns='tgv_name', values='tv_match_rate', aggfunc='max')
    pivot = pivot.replace({np.nan: None})
    if pivot.size == 0:
        raise ValueError("Pivot table is empty")

    z = pivot.values.astype(float)
    x = pivot.columns.tolist()
    y = pivot.index.tolist()
    text = [[(f"{v:.1f}%" if v is not None and not np.isnan(v) else "") for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, text=text, texttemplate="%{text}", hovertemplate="TV: %{y}<br>TGV: %{x}<br>Match: %{z:.1f}%<extra></extra>"))
    fig.update_layout(title=f"TV Match Heatmap — Employee {employee_id} (Top TGVs)", xaxis_title="TGV", yaxis_title="TV", height=500)
    return fig

def plot_strengths_gaps(results_df: pd.DataFrame, employee_id: str, top_k: int = 10) -> go.Figure:
    if 'tv_name' not in results_df.columns or 'tv_match_rate' not in results_df.columns:
        raise ValueError("results_df must contain at least 'tv_name' and 'tv_match_rate'")

    emp = results_df[results_df['employee_id'] == employee_id].copy()
    if emp.empty:
        raise ValueError(f"No rows for employee_id={employee_id}")

    emp['tv_match_rate'] = _ensure_percent_scale(emp['tv_match_rate'])
    # ensure baseline/user numeric if present
    if 'baseline_score' in emp.columns:
        emp['baseline_score'] = _coerce_series_to_numeric(emp['baseline_score'])
    if 'user_score' in emp.columns:
        emp['user_score'] = _coerce_series_to_numeric(emp['user_score'])

    agg = emp.groupby(['tgv_name', 'tv_name'], as_index=False).agg({
        'tv_match_rate': 'max',
        'baseline_score': 'max' if 'baseline_score' in emp.columns else (lambda x: np.nan),
        'user_score': 'max' if 'user_score' in emp.columns else (lambda x: np.nan)
    })

    agg_sorted = agg.sort_values('tv_match_rate', ascending=False)
    top_part = agg_sorted.head(top_k)
    bottom_part = agg_sorted.tail(top_k)
    display_df = pd.concat([top_part, bottom_part]).drop_duplicates().reset_index(drop=True)
    display_df = display_df.sort_values('tv_match_rate', ascending=True)

    if display_df.empty:
        raise ValueError("No TV rows with numeric tv_match_rate to show")

    def _color(v):
        if pd.isna(v):
            return '#d3d3d3'
        try:
            v = float(v)
        except Exception:
            return '#d3d3d3'
        if v >= 80:
            return '#2ca02c'
        if v >= 50:
            return '#ff7f0e'
        return '#d62728'

    colors = [_color(v) for v in display_df['tv_match_rate']]

    def _fmt(v):
        try:
            return f"{float(v):.1f}%"
        except Exception:
            return ""

    hover_text = display_df.apply(
        lambda r: f"TGV: {r.get('tgv_name','')}<br>TV: {r.get('tv_name','')}<br>Match: {_fmt(r.get('tv_match_rate', np.nan))}"
                  + (f"<br>User Score: {r['user_score']:.2f}" if pd.notna(r.get('user_score', np.nan)) else "")
                  + (f"<br>Baseline: {r['baseline_score']:.2f}" if pd.notna(r.get('baseline_score', np.nan)) else ""),
        axis=1
    )

    fig = go.Figure(go.Bar(
        x=[float(x) if pd.notna(x) else 0 for x in display_df['tv_match_rate']],
        y=display_df.apply(lambda r: f"{r.get('tv_name','')} — {r.get('tgv_name','')}", axis=1),
        orientation='h',
        marker=dict(color=colors),
        hovertext=hover_text,
        hoverinfo='text',
        text=[_fmt(x) for x in display_df['tv_match_rate']],
        textposition='outside'
    ))

    fig.update_layout(title=f"Strengths & Gaps — Employee {employee_id}", xaxis_title="TV Match Rate (%)", yaxis_title="TV — TGV", height=420 + 25 * len(display_df), margin=dict(l=240 if len(display_df) > 8 else 120, r=40, t=60, b=40))
    return fig
