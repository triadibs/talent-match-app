CREATE TABLE IF NOT EXISTS tb_tv_scores_long AS

-- Cognitive: iq + gtq (single columns), all names lowercase
SELECT employee_id,
       'cognitive_ability'::text AS tgv_name,
       'iq'::text AS tv_name,
       (iq::text)::numeric AS tv_score,
       'higher_better'::text AS scoring_direction
FROM tb_profiles_psych_norm
WHERE iq IS NOT NULL
  AND trim(coalesce(iq::text,'')) <> ''
  AND (iq::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL

SELECT employee_id,
       'cognitive_ability',
       'gtq',
       (gtq::text)::numeric,
       'higher_better'
FROM tb_profiles_psych_norm
WHERE gtq IS NOT NULL
  AND trim(coalesce(gtq::text,'')) <> ''
  AND (gtq::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

-- Attention Processing
UNION ALL
SELECT employee_id,
       'attention_processing',
       'faxtor',
       (faxtor::text)::numeric,
       'higher_better'
FROM tb_profiles_psych_norm
WHERE faxtor IS NOT NULL
  AND trim(coalesce(faxtor::text,'')) <> ''
  AND (faxtor::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT employee_id,
       'attention_processing',
       'pauli',
       (pauli::text)::numeric,
       'higher_better'
FROM tb_profiles_psych_norm
WHERE pauli IS NOT NULL
  AND trim(coalesce(pauli::text,'')) <> ''
  AND (pauli::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
-- tiki single column
SELECT employee_id,
       'attention_processing',
       'tiki',
       (tiki::text)::numeric,
       'higher_better'
FROM tb_profiles_psych_norm
WHERE tiki IS NOT NULL
  AND trim(coalesce(tiki::text,'')) <> ''
  AND (tiki::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

-- Competencies (latest year) — safe-cast
UNION ALL
SELECT c.employee_id,
       'leadership_competencies',
       'gdr',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'GDR' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'leadership_competencies',
       'cex',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'CEX' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'leadership_competencies',
       'ids',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'IDS' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'execution_competencies',
       'qdd',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'QDD' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'execution_competencies',
       'sto',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'STO' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'execution_competencies',
       'ftc',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'FTC' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'interpersonal_skills',
       'sea',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'SEA' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'interpersonal_skills',
       'lie',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'LIE' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

UNION ALL
SELECT c.employee_id,
       'interpersonal_skills',
       'csi',
       (c.score::text)::numeric,
       'higher_better'
FROM competencies_yearly c, tb_latest_year ly
WHERE c.pillar_code = 'CSI' AND c.year = ly.max_year
  AND c.score IS NOT NULL AND trim(coalesce(c.score::text,'')) <> ''
  AND (c.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

-- PAPI (already long) — safe-cast score and normalize tv_name to lowercase
UNION ALL
SELECT ps.employee_id,
       'work_preferences',
       ('papi_' || lower(ps.scale_code))::text AS tv_name,
       (ps.score::text)::numeric AS tv_score,
       CASE WHEN ps.scale_code IN ('Z','K') THEN 'lower_better' ELSE 'higher_better' END
FROM papi_scores ps
WHERE ps.score IS NOT NULL AND trim(coalesce(ps.score::text, '')) <> ''
  AND (ps.score::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')

-- Experience
UNION ALL
SELECT e.employee_id,
       'experience',
       'years_of_service_months',
       (e.years_of_service_months::text)::numeric,
       'higher_better'
FROM employees e
WHERE e.years_of_service_months IS NOT NULL
  AND trim(coalesce(e.years_of_service_months::text, '')) <> ''
  AND (e.years_of_service_months::text ~ '^\s*[-+]?\d+(\.\d+)?\s*$')
;

