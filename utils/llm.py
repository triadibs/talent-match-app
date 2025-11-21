# utils/llm.py
# -*- coding: utf-8 -*-
"""
LLM Integration for Job Profile Generation (robust + fallback)
Uses OpenRouter API (OpenAI-compatible endpoints)
"""

import requests
import streamlit as st
import json
from typing import Dict
import pandas as pd
import time
import logging

logger = logging.getLogger("llm")
logger.setLevel(logging.INFO)

# ---- Configuration helpers ----
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
# fallback model hint (OpenRouter exposes many providers/models; this is only a hint)
DEFAULT_MODEL_HINT = "openai/gpt-4o"

def _get_api_key():
    # expects st.secrets["openrouter"]["api_key"]
    return st.secrets.get("openrouter", {}).get("api_key")

def _get_preferred_model():
    return st.secrets.get("openrouter", {}).get("model")


# ---- Utility: fetch available models from OpenRouter (best-effort) ----
def list_openrouter_models(api_key: str):
    """Return list of model ids from OpenRouter (or empty list on failure)."""
    try:
        resp = requests.get(
            f"{DEFAULT_OPENROUTER_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            # data may be dict with 'data' or 'models' depending on API shape
            if isinstance(data, dict):
                # try several keys
                for key in ("data", "models", "models_list"):
                    if key in data and isinstance(data[key], list):
                        return [m.get("id") or m.get("model") or m.get("name") for m in data[key]]
                # fallback: try top-level keys that look like model entries
                # if 'model' in data: ...
            # otherwise if it's a list already
            if isinstance(data, list):
                return [m.get("id") or m.get("model") or m.get("name") for m in data]
        else:
            logger.warning("Failed to list models from OpenRouter: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("Exception when listing models: %s", e)
    return []


# ---- Robust request with retries ----
def _post_with_retries(url, headers, json_body, retries=2, backoff=1):
    last_exc = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=json_body, timeout=30)
            # raise_for_status will raise for 4xx/5xx
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as he:
            # if 4xx/5xx, may be permanent; but for 429/5xx we might retry
            status = getattr(he.response, "status_code", None)
            # if model not found (404) -> break early so caller can handle
            if status == 404:
                raise
            last_exc = he
        except requests.exceptions.RequestException as rexc:
            last_exc = rexc

        # backoff for retryable errors
        if i < retries:
            sleep = backoff * (2 ** i)
            time.sleep(sleep)
    # if we exit loop, raise last exception
    if last_exc:
        raise last_exc
    return None


# ---- Main public function ----
def generate_job_profile(
    role_name: str,
    job_level: str,
    role_purpose: str,
    benchmark_employees: pd.DataFrame
) -> Dict[str, str]:
    """
    Generate job requirements, description, and competencies using LLM (OpenRouter).
    Falls back to template generator when API/model fails or response invalid.
    """
    try:
        api_key = _get_api_key()
        if not api_key:
            st.warning("OpenRouter API key not configured. Using fallback generation.")
            return generate_fallback_profile(role_name, job_level, role_purpose)

        # Prepare prompt context
        benchmark_context = prepare_benchmark_context(benchmark_employees)

        prompt = f"""You are an expert HR analyst and job description specialist. Based on the following information, generate a comprehensive job profile.

Role Name: {role_name}
Job Level: {job_level}
Role Purpose: {role_purpose}

Benchmark Employees:
{benchmark_context}

Task:
Return ONLY a JSON object with keys "requirements", "description", "competencies" (competencies as an array of strings).
No markdown, no extra text.
"""

        preferred_model = _get_preferred_model() or DEFAULT_MODEL_HINT

        payload = {
            "model": preferred_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1200
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        url = f"{DEFAULT_OPENROUTER_URL}/chat/completions"

        try:
            resp = _post_with_retries(url, headers, payload, retries=2, backoff=1)
        except requests.exceptions.HTTPError as he:
            # If 404, try to list models and pick an available one
            status = getattr(he.response, "status_code", None)
            body = he.response.text if he.response is not None else str(he)
            logger.warning("HTTPError calling OpenRouter: %s %s", status, body)
            if status == 404:
                # try to find a working model
                available = list_openrouter_models(api_key)
                logger.info("OpenRouter available models: %s", available[:10])
                # try to pick first model that contains common tokens
                pick = None
                for hint in ("gpt", "gpt-4", "gpt-4o", "claude", "meta-llama", "openai"):
                    for m in available:
                        if m and hint in str(m).lower():
                            pick = m
                            break
                    if pick:
                        break
                if pick:
                    logger.info("Retrying with model: %s", pick)
                    payload["model"] = pick
                    try:
                        resp = _post_with_retries(url, headers, payload, retries=1, backoff=1)
                    except Exception as e2:
                        logger.warning("Retry with alternate model failed: %s", e2)
                        st.warning(f"API request failed: {e2}. Using fallback generation.")
                        return generate_fallback_profile(role_name, job_level, role_purpose)
                else:
                    st.warning("Requested model not available on OpenRouter. Using fallback generation.")
                    return generate_fallback_profile(role_name, job_level, role_purpose)
            else:
                st.warning(f"API request failed: {he}. Using fallback generation.")
                return generate_fallback_profile(role_name, job_level, role_purpose)
        except Exception as e:
            st.warning(f"API request failed: {e}. Using fallback generation.")
            return generate_fallback_profile(role_name, job_level, role_purpose)

        # If we get here, resp is a successful requests.Response
        result = resp.json()

        # OpenRouter uses OpenAI-compatible schema: choices[*].message.content or choices[*].text
        content = None
        try:
            if isinstance(result, dict) and "choices" in result and len(result["choices"]) > 0:
                ch = result["choices"][0]
                # new style: ch['message']['content']
                if isinstance(ch.get("message"), dict) and "content" in ch["message"]:
                    content = ch["message"]["content"]
                elif "text" in ch:
                    content = ch["text"]
                elif "message" in ch and isinstance(ch["message"], str):
                    content = ch["message"]
        except Exception:
            content = None

        if not content:
            # fallback: try 'output' or 'data'
            content = result.get("output") or result.get("content") or json.dumps(result)

        # Strip code fences if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Attempt parse JSON
        try:
            profile = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed: %s. Content: %s", e, content[:500])
            st.warning(f"Failed to parse LLM response: {e}. Using fallback generation.")
            return generate_fallback_profile(role_name, job_level, role_purpose)

        # Validate structure
        required_keys = ["requirements", "description", "competencies"]
        if not all(key in profile for key in required_keys):
            st.warning("LLM response missing required fields. Using fallback.")
            return generate_fallback_profile(role_name, job_level, role_purpose)

        # Ensure competencies is a list
        if not isinstance(profile["competencies"], list):
            # try to coerce by splitting lines or commas
            comp_raw = profile.get("competencies")
            if isinstance(comp_raw, str):
                comps = [c.strip(" •-") for c in comp_raw.splitlines() if c.strip()]
                if len(comps) >= 1:
                    profile["competencies"] = comps
                else:
                    profile["competencies"] = []
            else:
                profile["competencies"] = list(profile["competencies"])

        return profile

    except Exception as e:
        logger.exception("Unexpected error in generate_job_profile: %s", e)
        st.warning(f"Unexpected error: {e}. Using fallback generation.")
        return generate_fallback_profile(role_name, job_level, role_purpose)


# ---- helper functions (unchanged from your original, small fixes) ----
def prepare_benchmark_context(benchmark_df: pd.DataFrame) -> str:
    """Prepare benchmark employee context for LLM prompt"""

    if benchmark_df is None or benchmark_df.empty:
        return "No benchmark employees provided."

    context_lines = []
    for idx, row in benchmark_df.iterrows():
        # try several possible column names for position & directorate
        fullname = row.get("fullname") or row.get("name") or "N/A"
        position = row.get("position") or row.get("role") or "N/A"
        directorate = row.get("directorate") or row.get("dir") or "N/A"
        line = f"- {fullname} | Position: {position} | Directorate: {directorate}"
        context_lines.append(line)

    return "\n".join(context_lines)


def generate_fallback_profile(role_name: str, job_level: str, role_purpose: str) -> Dict[str, str]:
    """Generate a basic job profile without LLM (fallback)"""

    level_requirements = {
        "Entry": "1-2 years of experience, bachelor's degree",
        "Junior": "2-3 years of experience, bachelor's degree",
        "Middle": "3-5 years of experience, bachelor's or master's degree",
        "Senior": "5-7 years of experience, bachelor's or master's degree",
        "Lead": "7-10 years of experience, proven leadership track record",
        "Manager": "8+ years of experience, team management experience",
        "Director": "10+ years of experience, strategic leadership experience"
    }

    exp_requirement = level_requirements.get(job_level, "Relevant experience")

    requirements = f"{role_name} ({job_level} Level)\n\n{role_purpose}\n\nRequirements:\n- {exp_requirement} in related field\n- Strong analytical and problem-solving skills\n- Proficiency in relevant tools and technologies\n- Excellent communication and collaboration abilities\n- Proven track record of delivering results\n- Ability to work independently and in teams\n- Continuous learning mindset\n- Stakeholder management skills"

    description = f"As a {role_name} at the {job_level} level, you will {role_purpose.lower()}. You'll work closely with cross-functional teams to analyze data, generate insights, and drive business decisions."

    competencies = [
        f"{role_name.split()[0]} expertise and domain knowledge",
        "Data analysis and statistical thinking",
        "Technical proficiency in relevant tools",
        "Problem-solving and critical thinking",
        "Communication and storytelling",
        "Project management and prioritization",
        "Stakeholder engagement",
        "Continuous improvement mindset",
        "Collaboration and teamwork",
        "Business acumen and strategic thinking"
    ]

    return {
        "requirements": requirements,
        "description": description,
        "competencies": competencies
    }
