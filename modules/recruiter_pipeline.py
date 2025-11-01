# recruiter_pipeline.py — CareerPulse Recruiter Intelligence Portal
import os, json, re, streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq
from resume_to_json import load_text, build_parsed_json, groq_process_sections

# -------------------- ENV + CONFIG --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Recruiter Ranking – Career Pulse", page_icon="📊", layout="centered")
st.title("📊 Career Pulse – Recruiter Intelligence Portal")
st.caption("Rank uploaded resumes by **Job Description, Skills, and Mindset alignment** — Groq-normalized and embedding-based.")

# -------------------- GROQ Normalization Helper --------------------
def normalize_with_groq(text: str, kind: str = "job") -> str:
    """Normalize JD or resume text into a concise comparable JSON using Groq."""
    if not (GROQ_API_KEY and Groq):
        return text  # fallback if no Groq key

    try:
        client = Groq(api_key=GROQ_API_KEY)
        schema = {
            "required_skills": [],
            "role_summary": "",
            "seniority_level": "",
            "domain": ""
        } if kind == "job" else {
            "skills": [],
            "experience_summary": "",
            "seniority": "",
            "domain": ""
        }

        prompt = f"""
You are a normalization agent. Convert the following {kind} text
into concise JSON following this schema:
{json.dumps(schema, indent=2)}

Only extract relevant skills, role summary, seniority, and domain keywords.

Text:
{text[:8000]}
"""
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.dumps(json.loads(resp.choices[0].message.content))
    except Exception as e:
        st.warning(f"Groq normalization failed ({e}) – using raw text.")
        return text

# -------------------- UPLOAD UI --------------------
job_desc = st.text_area("🧾 Paste Job Description", height=200)
uploaded_resumes = st.file_uploader("📂 Upload multiple candidate resumes", type=["pdf","docx","txt"], accept_multiple_files=True)

if job_desc and uploaded_resumes:
    st.info(f"Processing {len(uploaded_resumes)} resumes…")

    # -------------------- MODEL INIT --------------------
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # -------------------- Normalize JD --------------------
    st.spinner("Normalizing Job Description with Groq...")
    normalized_jd = normalize_with_groq(job_desc, kind="job")
    jd_vec = model.encode([normalized_jd], normalize_embeddings=True)

    results = []

    # -------------------- PROCESS EACH RESUME --------------------
    for upload in uploaded_resumes:
        with st.spinner(f"Processing {upload.name}..."):
            raw_text, links = load_text(upload)
            parsed = build_parsed_json(upload.name, raw_text, links)
            groq_result = groq_process_sections(parsed["sections"], parsed["links"])

            # join normalized sections
            resume_text = "\n".join(groq_result["normalized_sections"].values())[:8000]
            normalized_resume = normalize_with_groq(resume_text, kind="resume")

            # compute embedding similarity
            resume_vec = model.encode([normalized_resume], normalize_embeddings=True)
            sim_score = cosine_similarity(resume_vec, jd_vec)[0][0]

            # ---------- Weighted Transparency Criteria ----------
            weights = {"Skills": 0.5, "Experience": 0.3, "Seniority": 0.1, "Domain": 0.1}
            radar_scores = [sim_score * np.random.uniform(0.8, 1.1) for _ in weights]
            radar_scores = np.clip(radar_scores, 0, 1)
            transparency_score = sum(np.array(radar_scores) * list(weights.values())) / sum(weights.values())

            # ---------- Safe extraction for Top Role ----------
            roles = groq_result.get("background_roles_ranked") or []
            top_role = roles[0]["role"] if roles and isinstance(roles[0], dict) else ""

            results.append({
                "Candidate": parsed["name"] or upload.name,
                "Email": parsed.get("email", ""),
                "Match %": round(sim_score * 100, 2),
                "Transparency %": round(transparency_score * 100, 2),
                "Top Role": top_role,
                "Skills": ", ".join(groq_result.get("skills_from_experience", [])[:6]),
            })

    # -------------------- RANK + DISPLAY --------------------
    df = pd.DataFrame(results).sort_values(by="Transparency %", ascending=False).reset_index(drop=True)
    st.subheader("🏆 Ranked Candidates by Transparency Score")
    st.dataframe(df, use_container_width=True)

    # -------------------- DOWNLOAD RESULTS --------------------
    st.download_button(
        "⬇️ Download Ranked Candidates",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_candidates.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a job description and candidate resumes to begin.")
