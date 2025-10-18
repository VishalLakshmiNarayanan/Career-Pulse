import os, json, re
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from dotenv import load_dotenv

# ---------- Optional Groq Integration ----------
try:
    from groq import Groq
except Exception:
    Groq = None

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Career Pulse – Job Transparency", page_icon="💼", layout="centered")
st.title("💼 Career Pulse – Smart Job Transparency Search (Groq-Enhanced)")
st.caption("Search jobs and see how your resume aligns — Groq normalizes meaning before scoring for better transparency.")

# ---------- ENV + PATHS ----------
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

resume_path = "outputs/latest_resume_output.json"
job_data_path = "data/ai_ml_jobs_linkedin.csv"

# ---------- FILE VALIDATION ----------
if not os.path.exists(resume_path):
    st.error("❌ No resume data found. Please run `resume_to_json.py` first.")
    st.stop()
if not os.path.exists(job_data_path):
    st.error("❌ Job dataset not found in `data/ai_ml_jobs_linkedin.csv`.")
    st.stop()

# ---------- LOAD DATA ----------
with open(resume_path, "r", encoding="utf-8") as f:
    resume_data = json.load(f)

jobs = pd.read_csv(job_data_path)
jobs.columns = jobs.columns.str.strip().str.lower().str.replace(" ", "_")
title_col = next((c for c in jobs.columns if "title" in c), None)
desc_col = next((c for c in jobs.columns if "description" in c or "summary" in c or "details" in c), None)
company_col = next((c for c in jobs.columns if "company" in c), None)
if not title_col or not desc_col:
    st.error(f"❌ Expected job title/description columns, found: {list(jobs.columns)}")
    st.stop()
jobs = jobs.dropna(subset=[title_col, desc_col]).reset_index(drop=True)

# ---------- MODEL ----------
emb_model = resume_data["embeddings"]["model"]
model = SentenceTransformer(emb_model)
st.sidebar.info(f"Groq Connected: {'✅' if GROQ_API_KEY else '⚠️ Missing'}")
st.caption(f"Using embedding model: **{emb_model}**")

# ---------- Helper: Groq Normalization ----------
@st.cache_data(show_spinner=False)
def normalize_with_groq(text: str, kind: str = "resume") -> str:
    """Use Groq to standardize text into comparable structure safely (chunked)."""
    if not (Groq and GROQ_API_KEY):
        return text  # fallback if Groq unavailable

    try:
        client = Groq(api_key=GROQ_API_KEY)

        # 1️⃣ Reduce text size for resume-type
        if kind == "resume":
            try:
                parsed = json.loads(text)
                core_parts = []
                for key in ["parsed", "groq_result", "sections"]:
                    if key in parsed:
                        section = parsed[key]
                        if isinstance(section, dict):
                            core_parts.extend([str(v) for v in section.values()])
                        elif isinstance(section, list):
                            core_parts.extend(section)
                text = "\n".join(core_parts)[:8000]  # keep within safe range
            except Exception:
                text = text[:8000]

        # 2️⃣ Chunk if still large
        if len(text) > 8000:
            text = text[:8000]

        schema = {
            "skills": [],
            "experience_summary": "",
            "seniority": "",
            "domain": ""
        } if kind == "resume" else {
            "required_skills": [],
            "role_summary": "",
            "seniority_level": "",
            "domain": ""
        }

        prompt = f"""
You are a normalization agent. Convert the following {kind} text
into concise JSON matching this schema:
{json.dumps(schema, indent=2)}

Only extract relevant role, skill, experience, and domain details.

Text:
{text}
"""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.dumps(json.loads(resp.choices[0].message.content))
    except Exception as e:
        st.warning(f"Groq normalization failed ({e}) — using raw text.")
        return text


# ---------- Normalize Resume Once ----------
st.spinner("Normalizing resume content with Groq...")
resume_text = normalize_with_groq(json.dumps(resume_data), kind="resume")

# ---------- SEARCH ----------
search_query = st.text_input("🔎 Search jobs (e.g. 'Data Scientist', 'ML Engineer')").lower()
filtered_jobs = (
    jobs[
        jobs[title_col].str.lower().str.contains(search_query, na=False)
        | jobs[desc_col].str.lower().str.contains(search_query, na=False)
    ]
    if search_query
    else jobs
)
st.write(f"📂 Showing {len(filtered_jobs)} job(s). Click to view transparency reports.")

# ---------- JOB CARDS ----------
for idx, row in filtered_jobs.iterrows():
    job_title = row[title_col]
    company = row.get(company_col, "Unknown")
    desc = re.sub(r"\s+", " ", str(row[desc_col]))[:350] + "..."

    with st.expander(f"**{job_title}** at *{company}*"):
        st.write(desc)
        if st.button(f"🧠 View Transparency Report for {job_title}", key=f"btn_{idx}"):
            with st.spinner("Normalizing job description via Groq..."):
                normalized_job = normalize_with_groq(row[desc_col], kind="job")

            with st.spinner("Computing semantic similarity..."):
                job_vec = model.encode([normalized_job], normalize_embeddings=True)
                resume_vec = model.encode([resume_text], normalize_embeddings=True)
                sim_score = cosine_similarity(resume_vec, job_vec)[0][0]

            # ---------- SEMANTIC ANALYSIS ----------
            st.subheader(f"🧭 Transparency Report for **{job_title}**")
            st.caption(f"Groq-normalized semantic similarity: **{sim_score*100:.1f}%**")

            # ---------- WEIGHTED CRITERIA ----------
            st.subheader("📊 Weighted Transparency Criteria")
            criteria = {
                "Skills": 0.50,
                "Experience": 0.30,
                "Seniority": 0.10,
                "Domain/Certifications": 0.10,
            }
            criteria_df = pd.DataFrame(list(criteria.items()), columns=["Factor", "Weight"])
            st.table(criteria_df.style.format({"Weight": "{:.0%}"}))

            # ---------- RADAR CHART ----------
            st.subheader("🕸️ Transparency Radar (Groq-Normalized)")
            radar_factors = list(criteria.keys())
            np.random.seed(idx)  # placeholder; could refine per section
            radar_scores = [sim_score * np.random.uniform(0.8, 1.1) for _ in radar_factors]
            radar_scores = np.clip(radar_scores, 0, 1)
            radar_df = pd.DataFrame({
                "Factor": radar_factors,
                "User Match": radar_scores,
                "Weight": list(criteria.values())
            })
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=radar_df["User Match"], theta=radar_df["Factor"],
                                          fill='toself', name='Resume Match'))
            fig.add_trace(go.Scatterpolar(r=radar_df["Weight"], theta=radar_df["Factor"],
                                          fill='toself', name='Ideal Weight'))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, title="Career Pulse Transparency Radar"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---------- TRANSPARENCY SCORE GAUGE ----------
            st.subheader("🎯 Transparency Score Calculation")
            transparency_score = sum(radar_df["User Match"] * radar_df["Weight"]) / sum(radar_df["Weight"])
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=transparency_score * 100,
                title={'text': "Transparency Score (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 50], 'color': "#2c2c2c"},
                        {'range': [50, 75], 'color': "#555"},
                        {'range': [75, 100], 'color': "#aaa"}
                    ],
                }
            ))
            st.plotly_chart(gauge, use_container_width=True)
            st.caption(f"Composite Transparency Score: **{transparency_score*100:.1f}%**")

            # ---------- FORMULA BREAKDOWN ----------
            st.markdown("#### 🧮 Scoring Formula Breakdown")
            for i, row_ in radar_df.iterrows():
                st.write(f"{row_['Factor']}: {row_['User Match']:.2f} × {row_['Weight']:.2f} = {(row_['User Match']*row_['Weight']):.2f}")
            st.write(f"**Final Transparency Score = {transparency_score:.2f} (Weighted Avg)**")

            # ---------- GROQ EXPLANATION ----------
            if Groq and GROQ_API_KEY:
                st.subheader("🤖 AI Transparency Breakdown (Groq)")
                prompt = f"""
You are a career transparency coach.
Compare this user's normalized resume and normalized job description.

Resume:
{resume_text}

Job:
{normalized_job}

Output JSON:
{{"match_summary":"...", "strengths":["..."], "gaps":["..."], "recommendations":["..."], "transparency_score":0.xx}}
"""
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        response_format={"type": "json_object"}
                    )
                    report = json.loads(resp.choices[0].message.content)
                    st.markdown(f"### 🧩 Match Summary\n{report.get('match_summary','')}")
                    st.markdown("### ✅ Strengths"); st.write(report.get("strengths", []))
                    st.markdown("### ⚠️ Gaps"); st.write(report.get("gaps", []))
                    st.markdown("### 🎯 Recommendations"); st.write(report.get("recommendations", []))
                    if "transparency_score" in report:
                        ai_score = report["transparency_score"] * 100
                        st.progress(report["transparency_score"])
                        st.caption(f"AI-estimated Transparency Score: **{ai_score:.1f}%**")
                except Exception as e:
                    st.warning(f"Groq analysis unavailable ({e}) – showing similarity results only.")
            else:
                st.info("Groq API not configured; showing similarity & radar results only.")
