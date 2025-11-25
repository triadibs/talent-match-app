def show_analytics_page():
    st.markdown("## 📈 Analytics & Insights")

    # get from session
    detailed = st.session_state.get('matching_results_detailed', None)
    summary = st.session_state.get('matching_results_summary', None)
    vacancy_id = st.session_state.get('job_vacancy_id', None)

    # If detailed missing/empty try auto-load from DB using job_vacancy_id (best-effort)
    if (detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty)) and vacancy_id is not None:
        with st.spinner(f"Attempting to load detailed results for vacancy {vacancy_id}..."):
            try:
                detailed_loaded = db.run_matching_query(vacancy_id)
                if isinstance(detailed_loaded, pd.DataFrame) and not detailed_loaded.empty:
                    st.session_state.matching_results_detailed = detailed_loaded
                    detailed = detailed_loaded
                    st.success(f"Loaded {len(detailed)} detailed rows from DB for vacancy {vacancy_id}.")
                else:
                    # also try summary fallback from DB
                    try:
                        summary_loaded = db.get_summary_results(vacancy_id, limit=5000)
                        if isinstance(summary_loaded, pd.DataFrame) and not summary_loaded.empty:
                            st.session_state.matching_results_summary = summary_loaded
                            summary = summary_loaded
                            st.info("Loaded summary results from DB, but detailed rows are not available.")
                        else:
                            st.info("No detailed or summary rows found in DB for this vacancy.")
                    except Exception:
                        st.info("No detailed results and failed to load summary from DB.")
            except Exception as e:
                st.warning(f"Auto-load from DB failed: {e}")

    # If still no detailed, show actionable debug and exit
    if detailed is None or (isinstance(detailed, pd.DataFrame) and detailed.empty):
        st.warning("⚠️ No detailed results available. Please load or refresh results from 'View Results'.")
        # Helpful debug: show session info & sample of summary if present
        st.markdown("**Debug info:**")
        st.write({
            "job_vacancy_id": vacancy_id,
            "has_matching_results_detailed": bool(detailed is not None and isinstance(detailed, pd.DataFrame) and not detailed.empty),
            "has_matching_results_summary": bool(summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty),
            "session_keys": list(st.session_state.keys())
        })
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            st.markdown("**Sample summary rows (top 5):**")
            st.dataframe(summary.head(5), width='stretch')
        else:
            st.info("No summary available to preview. Go to 'View Results' → select vacancy → '📥 Load Results' or '🔄 Refresh detailed results from DB'.")
        return

    # At this point we have detailed (non-empty) — reuse normalization from app (if present)
    detailed, summary = _normalize_results_dfs(detailed, summary)

    # After normalization, if summary still missing, rebuild it
    if summary is None or summary.empty:
        if not detailed.empty and 'employee_id' in detailed.columns:
            summary = detailed.groupby('employee_id').agg({
                'fullname': 'first',
                'directorate': 'first',
                'role': 'first',
                'grade': 'first',
                'final_match_rate_percentage': 'first'
            }).reset_index()
            summary['final_match_rate_percentage'] = pd.to_numeric(summary['final_match_rate_percentage'], errors='coerce')
            st.session_state.matching_results_summary = summary

    if summary is None or summary.empty:
        st.warning("⚠️ No valid matching summary found after cleaning. Your detailed rows may be placeholders or missing numeric scores.")
        st.write("Post-clean sample of detailed rows:")
        st.write(detailed.head(10))
        return

    # --- proceed with analytics visuals (same as before) ---
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
    # Distribution + Top chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Match Score Distribution")
        try:
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
    st.markdown("### 📊 TGV-Level Analysis")
    # build tgv_summary and selection as before...
    try:
        tgv_summary = detailed[['employee_id', 'fullname']].drop_duplicates().copy()
        if 'final_match_rate_percentage' in summary.columns:
            summary['employee_id'] = summary['employee_id'].astype(str)
            tgv_summary['employee_id'] = tgv_summary['employee_id'].astype(str)
            tgv_summary = tgv_summary.merge(summary[['employee_id', 'final_match_rate_percentage']], on='employee_id', how='left')
        tgv_summary['final_match_rate_percentage'] = pd.to_numeric(tgv_summary.get('final_match_rate_percentage'), errors='coerce').fillna(-1)
        tgv_summary = tgv_summary.sort_values('final_match_rate_percentage', ascending=False).head(200)
    except Exception as e:
        st.error(f"Cannot prepare TGV employee list: {e}")
        return

    employee_options = tgv_summary['employee_id'].astype(str).tolist()
    if not employee_options:
        st.warning("No employees found in results.")
        return

    def _format_employee(eid):
        eid_s = str(eid)
        row = tgv_summary.loc[tgv_summary['employee_id'] == eid_s]
        if row.empty:
            return eid_s
        fullname = row['fullname'].iloc[0] if pd.notna(row['fullname'].iloc[0]) else ''
        score = row['final_match_rate_percentage'].iloc[0]
        try:
            if pd.notna(score) and float(score) >= 0:
                return f"{eid_s} - {fullname} ({float(score):.1f}%)"
        except Exception:
            pass
        return f"{eid_s} - {fullname}"

    selected_employee = st.selectbox("Select Employee for TGV Profile:", options=employee_options, format_func=_format_employee)
    selected_employee = str(selected_employee)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 TGV Radar Profile")
        try:
            emp_rows = detailed.loc[detailed['employee_id'].astype(str) == selected_employee]
            emp_rows = _drop_placeholder_rows(emp_rows)
            if emp_rows.empty:
                st.warning("No detailed rows for this employee after cleaning. The dataset may contain placeholder rows only.")
                st.write(emp_rows.head(5))
            else:
                has_tgv_numeric = ('tgv_match_rate' in emp_rows.columns) and (pd.to_numeric(emp_rows['tgv_match_rate'], errors='coerce').dropna().size > 0)
                if not has_tgv_numeric:
                    st.warning("No numeric TGV match rate values for this employee. Radar cannot be created.")
                    st.write(emp_rows.head(5))
                else:
                    fig_radar = plot_tgv_radar(emp_rows, selected_employee)
                    st.plotly_chart(fig_radar, width='stretch')
        except Exception as e:
            st.error(f"Error creating radar chart: {e}")
            st.write("Sample rows for this employee (post-clean):")
            st.write(emp_rows.head(5).to_dict(orient='records') if 'emp_rows' in locals() else "No emp_rows available")

    with col2:
        st.markdown("#### 🔥 TV Heatmap (Top TGVs)")
        try:
            emp_rows = detailed.loc[detailed['employee_id'].astype(str) == selected_employee]
            emp_rows = _drop_placeholder_rows(emp_rows)
            if emp_rows.empty:
                st.warning("No detailed rows for this employee after cleaning. Heatmap cannot be created.")
                st.write(emp_rows.head(5))
            else:
                has_tv_numeric = ('tv_match_rate' in emp_rows.columns) and (pd.to_numeric(emp_rows['tv_match_rate'], errors='coerce').dropna().size > 0)
                if not has_tv_numeric:
                    st.warning("No numeric TV match rate values for this employee. Heatmap cannot be created.")
                    st.write(emp_rows.head(5))
                else:
                    fig_heatmap = plot_tv_heatmap(emp_rows, selected_employee)
                    st.plotly_chart(fig_heatmap, width='stretch')
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
            st.write("Sample rows for this employee (post-clean):")
            st.write(emp_rows.head(5).to_dict(orient='records') if 'emp_rows' in locals() else "No emp_rows available")

    st.markdown("---")
    st.markdown("### ✅ Strengths & Gaps Analysis")
    try:
        emp_rows = detailed.loc[detailed['employee_id'].astype(str) == selected_employee]
        emp_rows = _drop_placeholder_rows(emp_rows)
        if emp_rows.empty:
            st.warning("No detailed rows for this employee after cleaning. Strengths & Gaps chart cannot be created.")
            st.write(emp_rows.head(5))
        else:
            has_tv_numeric = ('tv_match_rate' in emp_rows.columns) and (pd.to_numeric(emp_rows['tv_match_rate'], errors='coerce').dropna().size > 0)
            if not has_tv_numeric:
                st.warning("No numeric TV match rate values for this employee. Strengths & Gaps chart cannot be created.")
                st.write(emp_rows.head(5))
            else:
                fig_strengths_gaps = plot_strengths_gaps(emp_rows, selected_employee)
                st.plotly_chart(fig_strengths_gaps, width='stretch')
    except Exception as e:
        st.error(f"Error creating strengths/gaps chart: {e}")
        st.write("Sample rows for this employee (post-clean):")
        st.write(emp_rows.head(5).to_dict(orient='records') if 'emp_rows' in locals() else "No emp_rows available")
