# utils/visualizations.py
# -*- coding: utf-8 -*-
"""
Robust Visualization utilities for Talent Match Dashboard (Plotly)

This version is defensive: coerces numeric-like strings to floats, supports percent strings,
handles 0..1 vs 0..100 scaling, and avoids deprecated pandas APIs.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional


def _to_numeric_percent(series: pd.Series) -> pd.Series:
    """
    Convert a Series that may contain numeric-like strings or percent-strings into floats on 0..100 scale.
    Safe: does not use Series.str accessor (avoids dtype errors).
    """
    if series is None:
        return pd.Series(dtype=float)

    # Work on a copy and preserve true NaN
    s = series.copy()
    s = s.where(~s.isna(), other=np.nan)

    # Convert each non-null value to stripped string representation safely
    def _norm_val(x):
        if pd.isna(x):
            return np.nan
        try:
            txt = str(x).strip()
        except Exception:
            return np.nan
        if txt == "" or txt.lower() in ("nan", "none"):
            return np.nan
        # remove percent sign and normalize comma decimal separators
        txt = txt.replace("%", "")
        txt = txt.replace(",", ".")
        return txt

    s = s.map(_norm_val)

    # Now coerce to numeric
    coerced = pd.to_numeric(s, errors='coerce')

    # If no numeric values found, return float series (may be all NaN)
    if coerced.dropna().empty:
        return coerced.astype(float)

    # If values appear to be 0..1 scale, convert to 0..100
    maxv = coerced.max(skipna=True)
    if pd.notna(maxv) and maxv <= 1.0:
        coerced = coerced * 100.0

    return coerced.astype(float)


def _ensure_percent_scale(s: pd.Series) -> pd.Series:
    """Compatibility wrapper kept for naming parity."""
    return _to_numeric_percent(s)


def _ensure_str_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure listed columns in df are strings (but keep NaN as NaN).
    Converts non-null, non-string items to str(x).
    Returns df (modified copy).
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(object).where(~df[c].isna(), other=np.nan)
            df[c] = df[c].apply(lambda x: x if pd.isna(x) or isinstance(x, str) else str(x))
    return df


def plot_match_distribution(summary_df: pd.DataFrame) -> go.Figure:
    """Histogram of final_match_rate_percentage (expects percent 0..100)."""
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty or None")

    if 'final_match_rate_percentage' not in summary_df.columns:
        raise ValueError("summary_df must contain column 'final_match_rate_percentage'")

    x_series = _ensure_percent_scale(summary_df['final_match_rate_percentage']).dropna()
    if x_series.empty:
        raise ValueError("No numeric final_match_rate_percentage values to plot")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x_series,
        nbinsx=20,
        name='Count',
        opacity=0.7,
        hovertemplate='Match Rate: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    ))

    mean_val = float(x_series.mean())
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.1f}%", annotation_position="top")
    fig.add_vline(x=70, line_dash="dot", line_color="green", opacity=0.5)
    fig.add_vline(x=80, line_dash="dot", line_color="darkgreen", opacity=0.5)

    fig.update_layout(title="Match Score Distribution",
                      xaxis_title="Match Rate (%)",
                      yaxis_title="Number of Candidates",
                      showlegend=False, height=400, template="plotly_white")
    return fig


def plot_top_candidates(summary_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart for top candidates."""
    for col in ['final_match_rate_percentage', 'fullname', 'employee_id']:
        if col not in summary_df.columns:
            raise ValueError(f"summary_df must contain column '{col}'")

    df = summary_df.copy()
    # ensure employee_id & fullname strings
    df = _ensure_str_cols(df, ['employee_id', 'fullname'])
    df['final_match_rate_percentage'] = _ensure_percent_scale(df['final_match_rate_percentage'])
    if df['final_match_rate_percentage'].dropna().empty:
        raise ValueError("No numeric final_match_rate_percentage values for top candidates")

    top_df = df.nlargest(top_n, 'final_match_rate_percentage').copy()
    top_df = top_df.sort_values('final_match_rate_percentage', ascending=True)

    def _label(row):
        name = row.get('fullname', '')
        role = row.get('role', None)
        pct = row.get('final_match_rate_percentage', np.nan)
        try:
            pct_str = f"{float(pct):.1f}%"
        except Exception:
            pct_str = "N/A"
        if pd.notna(role):
            return f"{name} — {role} ({pct_str})"
        return f"{name} ({pct_str})"

    top_df['label'] = top_df.apply(_label, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_df['final_match_rate_percentage'],
        y=top_df['label'],
        orientation='h',
        text=top_df['final_match_rate_percentage'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"),
        textposition='outside',
        marker=dict(color=top_df['final_match_rate_percentage'], colorscale='Blues'),
        hovertemplate="<b>%{y}</b><br>Match: %{x:.1f}%<extra></extra>"
    ))

    fig.update_layout(title=f"Top {len(top_df)} Candidates",
                      xaxis_title="Match Rate (%)",
                      yaxis_title="Candidate",
                      height=400 + 30 * min(len(top_df), top_n),
                      template="plotly_white",
                      margin=dict(l=220 if len(top_df) > 6 else 120, r=40, t=60, b=40))
    return fig


def plot_tgv_radar(results_df: pd.DataFrame, employee_id: str) -> go.Figure:
    """Radar chart for TGV-level match rates for a single employee."""
    if results_df is None or results_df.empty:
        raise ValueError("results_df is empty or None")

    required = {'employee_id', 'tgv_name', 'tgv_match_rate'}
    if not required.issubset(results_df.columns):
        raise ValueError("results_df must contain 'employee_id', 'tgv_name', and 'tgv_match_rate' columns")

    # Ensure string columns exist and are safe
    df = _ensure_str_cols(results_df, ['employee_id', 'tgv_name'])
    # Filter safely by employee_id (accept numeric or str)
    emp_mask = df['employee_id'].astype(str) == str(employee_id)
    emp_df = df[emp_mask].copy()
    if emp_df.empty:
        raise ValueError(f"No rows for employee_id={employee_id} in results_df")

    # coerce tgv_match_rate to numeric percent
    emp_df['tgv_match_rate'] = _ensure_percent_scale(emp_df['tgv_match_rate'])
    tgv_df = emp_df.groupby('tgv_name', as_index=False)['tgv_match_rate'].max()

    tgv_df = tgv_df.dropna(subset=['tgv_match_rate'])
    if tgv_df.empty:
        raise ValueError("No numeric tgv_match_rate values found for this employee")

    tgv_df = tgv_df.sort_values('tgv_match_rate', ascending=False)
    categories = tgv_df['tgv_name'].astype(str).tolist()
    values = tgv_df['tgv_match_rate'].tolist()

    # close loop for radar
    if len(categories) == 1:
        categories = categories * 2
        values = values * 2
    else:
        categories = categories + [categories[0]]
        values = values + [values[0]]

    max_r = max(100, max(values) if values else 100)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=str(employee_id),
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_r])),
        showlegend=False,
        title=f"TGV Radar Profile — Employee {employee_id}",
        template="plotly_white",
        height=480
    )
    return fig


def plot_tv_heatmap(results_df: pd.DataFrame, employee_id: str, top_tgv_count: int = 4) -> go.Figure:
    """Heatmap of TV vs TGV showing tv_match_rate for an employee."""

    required = {'employee_id', 'tgv_name', 'tv_name', 'tv_match_rate'}
    if not required.issubset(results_df.columns):
        raise ValueError(f"results_df must contain columns: {required}")

    df = _ensure_str_cols(results_df, ['employee_id', 'tgv_name', 'tv_name'])
    emp_df = df[df['employee_id'].astype(str) == str(employee_id)].copy()
    if emp_df.empty:
        raise ValueError(f"No rows for employee_id={employee_id} in results_df")

    emp_df['tv_match_rate'] = _ensure_percent_scale(emp_df['tv_match_rate'])
    emp_df = emp_df.dropna(subset=['tv_match_rate'])
    if emp_df.empty:
        raise ValueError("No numeric tv_match_rate values for this employee")

    # rank tgv by tgv_match_rate if present else by mean tv_match_rate
    if 'tgv_match_rate' in emp_df.columns:
        emp_df['tgv_match_rate'] = _ensure_percent_scale(emp_df['tgv_match_rate'])
        tgv_rank = emp_df.groupby('tgv_name', as_index=False)['tgv_match_rate'].max().sort_values('tgv_match_rate', ascending=False)
    else:
        tgv_rank = emp_df.groupby('tgv_name', as_index=False)['tv_match_rate'].mean().sort_values('tv_match_rate', ascending=False)

    # drop NaN names just in case
    tgv_rank = tgv_rank.dropna(subset=['tgv_name'])
    top_tgvs = tgv_rank['tgv_name'].head(max(1, int(top_tgv_count))).tolist()
    if not top_tgvs:
        raise ValueError("No TGVs found to build heatmap.")

    df_plot = emp_df[emp_df['tgv_name'].isin(top_tgvs)].copy()
    if df_plot.empty:
        raise ValueError("No TV rows after filtering for top TGVs.")

    pivot = df_plot.pivot_table(index='tv_name', columns='tgv_name', values='tv_match_rate', aggfunc='max')
    if pivot.empty:
        raise ValueError("Pivot resulted in empty heatmap matrix.")

    pivot = pivot.fillna(np.nan)

    # order rows by max
    pivot['__max_val'] = pivot.max(axis=1)
    pivot = pivot.sort_values('__max_val', ascending=False).drop(columns='__max_val')

    z = pivot.values
    x = [str(col) for col in pivot.columns.tolist()]
    y = [str(idx) for idx in pivot.index.tolist()]

    # build text annotations robustly
    text = []
    for row in z:
        text_row = []
        for val in row:
            if pd.isna(val):
                text_row.append("")
            else:
                try:
                    text_row.append(f"{float(val):.1f}%")
                except Exception:
                    text_row.append("")
        text.append(text_row)

    # Heatmap hover: Plotly will show z but for NaN the z formatting fails, so we provide hovertemplate that uses text
    hovertemplate = "TV: %{y}<br>TGV: %{x}<br>Match: %{customdata}<extra></extra>"

    # customdata aligned with z shaped text values
    customdata = np.array(text)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=text,
        texttemplate="%{text}",
        customdata=customdata,
        hovertemplate=hovertemplate,
        colorscale='Viridis',
        colorbar=dict(title="Match %")
    ))

    fig.update_layout(title=f"TV Match Heatmap — Employee {employee_id} (Top TGVs)",
                      xaxis_title="TGV", yaxis_title="TV", height=500, template="plotly_white")
    return fig


def plot_strengths_gaps(results_df: pd.DataFrame, employee_id: str, top_k: int = 10) -> go.Figure:
    """Bar chart showing strengths and gaps at TV level for the selected employee."""

    if results_df is None or results_df.empty:
        raise ValueError("results_df is empty or None")

    required_cols = {'employee_id', 'tv_name', 'tv_match_rate'}
    if not required_cols.issubset(results_df.columns):
        raise ValueError("results_df must contain at least 'tv_name' and 'tv_match_rate'")

    df = _ensure_str_cols(results_df, ['employee_id', 'tv_name', 'tgv_name'])
    emp_df = df[df['employee_id'].astype(str) == str(employee_id)].copy()
    if emp_df.empty:
        raise ValueError(f"No rows for employee_id={employee_id} in results_df")

    emp_df['tv_match_rate'] = _ensure_percent_scale(emp_df['tv_match_rate'])
    emp_df = emp_df.dropna(subset=['tv_match_rate'])
    if emp_df.empty:
        raise ValueError("No numeric tv_match_rate values for this employee")

    # Build aggregation dict safely
    agg_dict = {'tv_match_rate': 'max'}
    if 'baseline_score' in emp_df.columns:
        agg_dict['baseline_score'] = 'max'
    if 'user_score' in emp_df.columns:
        agg_dict['user_score'] = 'max'

    agg = emp_df.groupby(['tgv_name', 'tv_name'], as_index=False).agg(agg_dict)

    # Ensure expected column names present
    if 'tv_match_rate' not in agg.columns:
        # attempt to find candidate
        found = [c for c in agg.columns if 'tv_match_rate' in c]
        if found:
            agg = agg.rename(columns={found[0]: 'tv_match_rate'})
        else:
            raise ValueError("Aggregation failed to produce 'tv_match_rate' column")

    agg_sorted = agg.sort_values('tv_match_rate', ascending=False).reset_index(drop=True)

    # Build display df: top_k strengths + top_k gaps (don't duplicate)
    top_strengths = agg_sorted.head(int(top_k)).copy()
    top_gaps = agg_sorted.tail(int(top_k)).copy()
    display_df = pd.concat([top_strengths, top_gaps], ignore_index=True).drop_duplicates().reset_index(drop=True)

    if display_df.empty:
        raise ValueError("No TV rows with numeric tv_match_rate to display")

    display_df = display_df.sort_values('tv_match_rate', ascending=True).reset_index(drop=True)

    def _color(v):
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

    def _hover(r):
        tgv = r.get('tgv_name', '') if 'tgv_name' in r else ''
        tv = r.get('tv_name', '') if 'tv_name' in r else ''
        base = f"TGV: {tgv}<br>TV: {tv}"
        try:
            match = f"<br>Match: {float(r['tv_match_rate']):.1f}%"
        except Exception:
            match = ""
        user = ""
        base_score = ""
        if 'user_score' in r and pd.notna(r.get('user_score', np.nan)):
            try:
                user = f"<br>User Score: {float(r['user_score']):.2f}"
            except Exception:
                user = ""
        if 'baseline_score' in r and pd.notna(r.get('baseline_score', np.nan)):
            try:
                base_score = f"<br>Baseline: {float(r['baseline_score']):.2f}"
            except Exception:
                base_score = ""
        return base + match + user + base_score

    hover_text = display_df.apply(_hover, axis=1)

    y_labels = display_df.apply(lambda r: f"{r.get('tv_name','')} — {r.get('tgv_name','')}", axis=1)

    fig = go.Figure(go.Bar(
        x=display_df['tv_match_rate'],
        y=y_labels,
        orientation='h',
        marker=dict(color=colors),
        hovertext=hover_text,
        hoverinfo='text',
        text=display_df['tv_match_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Strengths & Gaps — Employee {employee_id}",
        xaxis_title="TV Match Rate (%)",
        yaxis_title="TV — TGV",
        height=420 + 25 * len(display_df),
        template="plotly_white",
        margin=dict(l=240 if len(display_df) > 8 else 120, r=40, t=60, b=40)
    )
    return fig
