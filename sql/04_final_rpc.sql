CREATE TABLE IF NOT EXISTS tb_final_aggregation AS
SELECT
  employee_id,
  fullname,
  directorate,
  role,
  grade,
  CASE
    WHEN SUM(tgv_weight) FILTER (WHERE tgv_match_rate IS NOT NULL) = 0 THEN NULL
    ELSE SUM(tgv_match_rate * tgv_weight) FILTER (WHERE tgv_match_rate IS NOT NULL)
         / NULLIF(SUM(tgv_weight) FILTER (WHERE tgv_match_rate IS NOT NULL), 0)
  END AS final_match_rate,
  SUM(CASE WHEN tgv_match_rate IS NOT NULL THEN 1 ELSE 0 END) AS tgv_count_with_baseline
FROM tb_tgv_with_weights
GROUP BY employee_id, fullname, directorate, role, grade;


SELECT
  tm.employee_id,
  tm.fullname,
  tm.directorate,
  tm.role,
  tm.grade,
  tm.tgv_name,
  tm.tv_name,
  ROUND(tm.baseline_score::numeric, 2) AS baseline_score,
  ROUND(tm.user_score::numeric, 2) AS user_score,
  ROUND(tm.tv_match_rate::numeric, 2) AS tv_match_rate,
  ROUND(tgv.tgv_match_rate::numeric, 4) AS tgv_match_rate,
  ROUND(fa.final_match_rate::numeric, 4) AS final_match_rate
FROM tb_tv_match_calc tm
LEFT JOIN tb_tgv_with_weights tgv
  ON tgv.employee_id = tm.employee_id AND tgv.tgv_name = tm.tgv_name
LEFT JOIN tb_final_aggregation fa ON fa.employee_id = tm.employee_id
WHERE tm.baseline_score IS NOT NULL
ORDER BY fa.final_match_rate DESC NULLS LAST, tm.employee_id, tm.tgv_name, tm.tv_name
LIMIT 1000;


SELECT
  fa.employee_id,
  fa.fullname,
  fa.directorate,
  fa.role,
  fa.grade,
  ROUND(fa.final_match_rate::numeric, 2) AS final_match_rate_percentage,
  fa.tgv_count_with_baseline
FROM tb_final_aggregation fa
WHERE fa.final_match_rate IS NOT NULL
ORDER BY fa.final_match_rate DESC
LIMIT 50;


