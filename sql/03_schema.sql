CREATE TABLE IF NOT EXISTS tb_baseline_per_tv AS
SELECT
  st.job_vacancy_id,
  tv.tgv_name,
  tv.tv_name,
  tv.scoring_direction,
  PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY tv.tv_score) AS baseline_score,
  COUNT(DISTINCT tv.employee_id) AS benchmark_count
FROM tb_selected_talents st
JOIN tb_tv_scores_long tv ON tv.employee_id = st.employee_id
GROUP BY st.job_vacancy_id, tv.tgv_name, tv.tv_name, tv.scoring_direction;


CREATE TABLE IF NOT EXISTS tb_employees_with_tv AS
SELECT
  e.employee_id,
  e.fullname,
  dir.name AS directorate,
  pos.name AS role,
  gr.name AS grade,
  tv.tgv_name,
  tv.tv_name,
  tv.tv_score AS user_score,
  tv.scoring_direction,
  b.baseline_score,
  v.weights_config,
  v.job_vacancy_id
FROM employees e
LEFT JOIN dim_directorates dir ON e.directorate_id = dir.directorate_id
LEFT JOIN dim_positions pos ON e.position_id = pos.position_id
LEFT JOIN dim_grades gr ON e.grade_id = gr.grade_id
CROSS JOIN tb_vacancy v
LEFT JOIN tb_tv_scores_long tv ON tv.employee_id = e.employee_id
LEFT JOIN tb_baseline_per_tv b
  ON b.job_vacancy_id = v.job_vacancy_id
  AND b.tgv_name = tv.tgv_name
  AND b.tv_name = tv.tv_name
WHERE tv.tv_score IS NOT NULL;  -- only rows where employee actually has that TV score


CREATE TABLE IF NOT EXISTS tb_tv_match_calc AS
SELECT
  e.*,
  CASE
    WHEN e.baseline_score IS NULL THEN NULL
    WHEN e.scoring_direction = 'boolean' THEN
      CASE WHEN e.user_score = e.baseline_score THEN 100.0 ELSE 0.0 END
    WHEN e.scoring_direction = 'lower_better' THEN
      CASE
        WHEN e.baseline_score = 0 OR e.user_score IS NULL THEN NULL
        ELSE LEAST(GREATEST(((2.0 * e.baseline_score - e.user_score) / NULLIF(e.baseline_score,0)) * 100.0, 0.0), 100.0)
      END
    ELSE
      CASE
        WHEN e.baseline_score = 0 OR e.user_score IS NULL THEN NULL
        ELSE LEAST(GREATEST((e.user_score / NULLIF(e.baseline_score,0)) * 100.0, 0.0), 100.0)
      END
  END AS tv_match_rate
FROM tb_employees_with_tv e;


CREATE TABLE IF NOT EXISTS tb_tv_match_with_weights AS
SELECT
  t.*,
  CASE
    WHEN t.weights_config IS NULL THEN 1.0
    WHEN (t.weights_config #>> ARRAY['TV_weights', t.tgv_name, t.tv_name]) ~ '^\s*[-+]?\d+(\.\d+)?\s*$'
      THEN (t.weights_config #>> ARRAY['TV_weights', t.tgv_name, t.tv_name])::numeric
    ELSE 1.0
  END AS tv_weight
FROM tb_tv_match_calc t;


CREATE TABLE IF NOT EXISTS tb_tgv_aggregation AS
SELECT
  employee_id,
  fullname,
  directorate,
  role,
  grade,
  tgv_name,
  CASE
    WHEN SUM(tv_weight) FILTER (WHERE tv_match_rate IS NOT NULL) = 0 THEN NULL
    ELSE SUM(tv_match_rate * tv_weight) FILTER (WHERE tv_match_rate IS NOT NULL)
         / NULLIF(SUM(tv_weight) FILTER (WHERE tv_match_rate IS NOT NULL), 0)
  END AS tgv_match_rate,
  COUNT(*) FILTER (WHERE tv_match_rate IS NOT NULL) AS tv_count_with_baseline,

  -- solusi untuk JSON
  MIN(weights_config::text)::jsonb AS weights_config

FROM tb_tv_match_with_weights
GROUP BY employee_id, fullname, directorate, role, grade, tgv_name;


CREATE TABLE IF NOT EXISTS tb_tgv_with_weights AS
SELECT
  t.*,
  CASE
    WHEN t.weights_config IS NULL THEN
      CASE t.tgv_name
        WHEN 'Interpersonal_Skills' THEN 0.612
        WHEN 'Leadership_Competencies' THEN 0.314
        WHEN 'Execution_Competencies' THEN 0.041
        WHEN 'Attention_Processing' THEN 0.016
        WHEN 'Cognitive_Ability' THEN 0.009
        WHEN 'Experience' THEN 0.005
        WHEN 'Work_Preferences' THEN 0.003
        ELSE 1.0 / 7.0
      END
    WHEN (t.weights_config #>> ARRAY['TGV_weights', t.tgv_name]) ~ '^\s*[-+]?\d+(\.\d+)?\s*$'
      THEN (t.weights_config #>> ARRAY['TGV_weights', t.tgv_name])::numeric
    ELSE
      CASE t.tgv_name
        WHEN 'Interpersonal_Skills' THEN 0.612
        WHEN 'Leadership_Competencies' THEN 0.314
        WHEN 'Execution_Competencies' THEN 0.041
        WHEN 'Attention_Processing' THEN 0.016
        WHEN 'Cognitive_Ability' THEN 0.009
        WHEN 'Experience' THEN 0.005
        WHEN 'Work_Preferences' THEN 0.003
        ELSE 1.0 / 7.0
      END
  END AS tgv_weight
FROM tb_tgv_aggregation t;
