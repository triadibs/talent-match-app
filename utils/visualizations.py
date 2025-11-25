# -*- coding: utf-8 -*-
"""
Fixed Visualization utilities for Talent Match Dashboard (Plotly)
Resolves: "Can only use .str accessor with string values!" error
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional


def _to_numeric_percent(series: pd.Series) -> pd.Series:
    """
    Convert a Series that may contain numeric-like strings or percent-strings into floats on 0..100 scale.
    FIXED: Properly handles all data types without using .str accessor on non-string data
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)

    s = series.copy()
    
    # Convert to string safely, preserving NaN
    def safe_to_str(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()
    
    s = s.apply(safe_to_str)
    
    # Replace empty strings and 'nan' text with actual NaN
    s = s.replace(['', 'nan', 'None', 'NaN', 'null'], np.nan)
    
    # Remove percent sign and normalize decimal separator - ONLY for non-NaN strings
    def clean_numeric_str(x):
        if pd.isna(x):
            return np.nan
        try:
            x = str(x).replace('%', '').replace(',', '.')
            return x
        except:
            return np.nan
    
    s = s.apply(clean_numeric_str)
    
    # Coerce to numeric
    coerced = pd.to_numeric(s, errors='coerce')

    # Scale 0..1 to 0..100 if needed
    if not coerced.dropna().empty:
        maxv = coerced.max(skipna=True)
        if pd.notna(maxv) and maxv <= 1.0:
            coerced = coerced * 100.0

    return coerced.astype(float)


def _ensure_str_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    FIXED: Ensure listed columns are properly converted to strings
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            def safe_str_convert(x):
                if pd.isna(x):
                    return np.nan
                if isinstance(x, str):
                    return x
                return str(x)
            
            df[c] = df[c].apply(safe_str_convert)
    return df


def plot_match_distribution(summary_df: pd.DataFrame) -> go.Figure:
    """Histogram of final_match_rate_percentage (expects percent 0..100)."""
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df is empty or None")

    if 'final_match_rate_percentage' not in summary_df.columns:
        raise ValueError("summary_df must contain column 'final_match_rate_percentage'")

    x_series = _to_numeric_percent(summary_df['final_match_rate_percentage']).dropna()
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
    df = _ensure_str_cols(df, ['employee_id', 'fullname', 'role', 'directorate'])
    df['final_match_rate_percentage'] = _to_numeric_percent(df['final_match_rate_percentage'])
    
    if df['final_match_rate_percentage'].dropna().empty:
        raise ValueError("No numeric final_match_rate_percentage values for top candidates")

    top_df = df.nlargest(top_n, 'final_match_rate_percentage').copy()
    top_df = top_df.sort_values('final_match_rate_percentage', ascending=True)

    def _label(row):
        name = str(row.get('fullname', '')) if pd.notna(row.get('fullname')) else ''
        role = str(row.get('role', '')) if pd.notna(row.get('role')) else ''
        pct = row.get('final_match_rate_percentage', np.nan)
        try:
            pct_str = f"{float(pct):.1f}%"
        except:
            pct_str = "N/A"
        if role:
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

    df = _ensure_str_cols(results_df, ['employee_id', 'tgv_name'])
    
    # Filter by employee_id - convert both to string for comparison
    emp_mask = df['employee_id'].apply(lambda x: str(x) if pd.notna(x) else '') == str(employee_id)
    emp_df = df[emp_mask].copy()
    
    if emp_df.empty:
        raise ValueError(f"No rows for employee_id={employee_id} in results_df")

    emp_df['tgv_match_rate'] = _to_numeric_percent(emp_df['tgv_match_rate'])
    tgv_df = emp_df.groupby('tgv_name', as_index=False)['tgv_match_rate'].max()

    tgv_df = tgv_df.dropna(subset=['tgv_match_rate'])
    if tgv_df.empty:
        raise ValueError("No numeric tgv_match_rate values found for this employee")

    tgv_df = tgv_df.sort_values('tgv_match_rate', ascending=False)
    categories = [str(x) if pd.notna(x) else '' for x in tgv_df['tgv_name'].tolist()]
    values = tgv_df['tgv_match_rate'].tolist()

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
    emp_df = df[df['employee_id'].apply(lambda x: str(x) if pd.notna(x) else '') == str(employee_id)].copy()
    
    if emp_df.empty:
        raise ValueError(f"No rows for employee_id={employee_id} in results_df")

    emp_df['tv_match_rate'] = _to_numeric_percent(emp_df['tv_match_rate'])
    emp_df = emp_df.dropna(subset=['tv_match_rate'])
    
    if emp_df.empty:
        raise ValueError("No numeric tv_match_rate values for this employee")

    if 'tgv_match_rate' in emp_df.columns:
        emp_df['tgv_match_rate'] = _to_numeric_percent(emp_df['tgv_match_rate'])
        tgv_rank = emp_df.groupby('tgv_name', as_index=False)['tgv_match_rate'].max().sort_values('tgv_match_rate', ascending=False)
    else:
        tgv_rank = emp_df.groupby('tgv_name', as_index=False)['tv_match_rate'].mean().sort_values('tv_match_rate', ascending=False)

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
    pivot['__max_val'] = pivot.max(axis=1)
    pivot = pivot.sort_values('__max_val', ascending=False).drop(columns='__max_val')

    z = pivot.values
    x = [str(col) for col in pivot.columns.tolist()]
    y = [str(idx) for idx in pivot.index.tolist()]

    text = []
    for row in z:
        text_row = []
        for val in row:
            if pd.isna(val):
                text_row.append("")
            else:
                try:
                    text_row.append(f"{float(val):.1f}%")
                except:
                    text_row.append("")
        text.append(text_row)

    customdata = np.array(text)
    hovertemplate = "TV: %{y}<br>TGV: %{x}<br>Match: %{customdata}<extra></extra>"

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
    emp_df = df[df['employee_id'].apply(lambda x: str(x) if pd.notna(x) else '') == str(employee_id)].copy()
    
    if emp_df.empty:
        raise ValueError(f"No rows for employee_id={employee_id} in results_df")

    emp_df['tv_match_rate'] = _to_numeric_percent(emp_df['tv_match_rate'])
    emp_df = emp_df.dropna(subset=['tv_match_rate'])
    
    if emp_df.empty:
        raise ValueError("No numeric tv_match_rate values for this employee")

    agg_dict = {'tv_match_rate': 'max'}
    if 'baseline_score' in emp_df.columns:
        agg_dict['baseline_score'] = 'max'
    if 'user_score' in emp_df.columns:
        agg_dict['user_score'] = 'max'

    agg = emp_df.groupby(['tgv_name', 'tv_name'], as_index=False).agg(agg_dict)

    if 'tv_match_rate' not in agg.columns:
        found = [c for c in agg.columns if 'tv_match_rate' in c]
        if found:
            agg = agg.rename(columns={found[0]: 'tv_match_rate'})
        else:
            raise ValueError("Aggregation failed to produce 'tv_match_rate' column")

    agg_sorted = agg.sort_values('tv_match_rate', ascending=False).reset_index(drop=True)

    top_strengths = agg_sorted.head(int(top_k)).copy()
    top_gaps = agg_sorted.tail(int(top_k)).copy()
    display_df = pd.concat([top_strengths, top_gaps], ignore_index=True).drop_duplicates().reset_index(drop=True)

    if display_df.empty:
        raise ValueError("No TV rows with numeric tv_match_rate to display")

    display_df = display_df.sort_values('tv_match_rate', ascending=True).reset_index(drop=True)

    def _color(v):
        try:
            v = float(v)
        except:
            return '#d3d3d3'
        if v >= 80:
            return '#2ca02c'
        if v >= 50:
            return '#ff7f0e'
        return '#d62728'

    colors = [_color(v) for v in display_df['tv_match_rate']]

    def _hover(r):
        tgv = str(r.get('tgv_name', '')) if pd.notna(r.get('tgv_name')) else ''
        tv = str(r.get('tv_name', '')) if pd.notna(r.get('tv_name')) else ''
        base = f"TGV: {tgv}<br>TV: {tv}"
        try:
            match = f"<br>Match: {float(r['tv_match_rate']):.1f}%"
        except:
            match = ""
        user = ""
        base_score = ""
        if 'user_score' in r and pd.notna(r.get('user_score', np.nan)):
            try:
                user = f"<br>User Score: {float(r['user_score']):.2f}"
            except:
                user = ""
        if 'baseline_score' in r and pd.notna(r.get('baseline_score', np.nan)):
            try:
                base_score = f"<br>Baseline: {float(r['baseline_score']):.2f}"
            except:
                base_score = ""
        return base + match + user + base_score

    hover_text = display_df.apply(_hover, axis=1)

    def safe_label(r):
        tv = str(r.get('tv_name', '')) if pd.notna(r.get('tv_name')) else ''
        tgv = str(r.get('tgv_name', '')) if pd.notna(r.get('tgv_name')) else ''
        return f"{tv} — {tgv}"
    
    y_labels = display_df.apply(safe_label, axis=1)

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
