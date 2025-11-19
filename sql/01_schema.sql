-- Create table untuk menyimpan job vacancy & benchmark selection
CREATE TABLE IF NOT EXISTS talent_benchmarks (
    job_vacancy_id SERIAL PRIMARY KEY,
    role_name TEXT NOT NULL,
    job_level TEXT,
    role_purpose TEXT,
    selected_talent_ids TEXT[], -- Array of employee IDs
    weights_config JSONB, -- Custom weights (optional)
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert example vacancy
INSERT INTO talent_benchmarks (
    role_name, 
    job_level, 
    role_purpose, 
    selected_talent_ids,
    weights_config
) VALUES (
    'Data Analyst',
    'Middle',
    'Turn business questions into data-driven answers through SQL, Python, and visualization.',
    ARRAY['312', '335', '175'], -- Employee IDs dengan rating 5
    NULL -- NULL = equal weights, atau bisa custom JSON
);

-- 1) Vacancy (set vacancy id di sini)
CREATE TABLE IF NOT EXISTS tb_vacancy AS
SELECT
  j.job_vacancy_id,
  j.role_name,
  j.job_level,
  j.role_purpose,
  j.selected_talent_ids,
  j.weights_config
FROM talent_benchmarks j
WHERE j.job_vacancy_id = 1;  -- <-- Ganti ID di sini


CREATE TABLE IF NOT EXISTS tb_latest_year AS
SELECT MAX(year) AS max_year
FROM competencies_yearly;


-- UNNEST selected_talent_ids into rows (persistent table)
CREATE TABLE IF NOT EXISTS tb_selected_talents AS
SELECT v.job_vacancy_id,
       unnest(v.selected_talent_ids) AS employee_id,
       v.weights_config
FROM tb_vacancy v;


CREATE TABLE IF NOT EXISTS tb_profiles_psych_norm AS
SELECT
  employee_id,
  iq,
  gtq,
  faxtor, pauli, tiki
FROM profiles_psych;
