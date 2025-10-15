# resume_to_json.py
import os, re, json, time
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
from dotenv import load_dotenv

# -------------------- ENV --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------- IO --------------------
import pdfplumber
import docx

# -------------------- Embeddings --------------------
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------- Groq LLM --------------------
try:
    from groq import Groq
except Exception:
    Groq = None  # handle gracefully if lib missing

# -------------------- NLTK (optional) --------------------
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt', quiet=True)

# -------------------- Regex/Heuristics --------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9.+_-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?[\d\s.-]{6,15}")
URL_RE   = re.compile(r"(?i)\b((?:https?://|www\.)[^\s<>()]+)")
SECTION_HEADINGS = [
    "summary","about","education","experience","work experience","professional experience",
    "skills","projects","certifications","publications","awards","activities",
    "volunteer","research","leadership"
]

ALIASES = {  # canonicalization for skills
    r"tailwind\s*css": "Tailwind CSS",
    r"shadcn(\s*/\s*ui)?": "shadcn/ui",
    r"ml\s*flow|mlflow": "MLflow",
    r"postgres(\s*ql)?": "PostgreSQL",
    r"sql\s*server": "SQL Server",
    r"n-?8-?n|n8n": "n8n",
    r"efficientnet[- ]?b0": "EfficientNetB0",
    r"geo\s*pandas|geopandas": "GeoPandas",
    r"large language models|llms?": "LLM",
}

TECH_WORD = re.compile(
    r"\b("
    r"python|java|c\+\+|c#|sql|postgres|mysql|sqlite|sql server|nosql|mongodb|spark|pyspark|hadoop|airflow|mlflow|docker|kubernetes|"
    r"aws|gcp|azure|snowflake|bigquery|redshift|pandas|numpy|scikit-learn|tensorflow|pytorch|"
    r"nlp|transformers|bert|llama|sbert|llm|rag|faiss|pgvector|supabase|streamlit|fastapi|flask|opencv|"
    r"react|next\.js|typescript|javascript|tailwind|tailwind css|shadcn|git|linux|bash|tableau|power bi|dbt|geopandas|"
    r"time series|forecasting|demand forecasting|efficientnetb0|cplex|databricks|rest apis?"
    r")\b",
    flags=re.I
)

# -------------------- Utils --------------------
def clean_url(u: str) -> str:
    u = u.strip().rstrip(").,;")
    if u.lower().startswith("www."):
        u = "https://" + u
    return u

def _string_list(x) -> List[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if isinstance(i, (str,int,float)) and str(i).strip()]
    if isinstance(x, str) and x.strip():
        parts = re.split(r"[\n,;]+", x)
        return [p.strip() for p in parts if p.strip()]
    return []

def _as_str(x) -> str:
    if isinstance(x, str): return x.strip()
    return ""

def _norm_links(links) -> List[str]:
    if not isinstance(links, list): return []
    return sorted({clean_url(str(u)) for u in links if str(u).strip()})

def canonicalize_skills(skills: List[str]) -> List[str]:
    out = []
    for s in skills:
        t = s.strip()
        if not t: continue
        low = t.lower()
        rep = None
        for pat, canonical in ALIASES.items():
            if re.fullmatch(pat, low):
                rep = canonical; break
        out.append(rep or t)
    seen, dedup = set(), []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k); dedup.append(s)
    return dedup

# -------------------- File loaders --------------------
def _strip_pdf_artifacts(text: str) -> str:
    text = re.sub(r"\(cid:\d+\)", "", text)  # remove (cid:###)
    # normalize bullets and spaces
    text = text.replace("•", "\n- ").replace("", "\n- ").replace("◦", "\n- ")
    # hyphenation fix (common PDF line breaks)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r\n", "\n")
    return text

def extract_text_from_pdf(file) -> Tuple[str, List[str]]:
    text, urls = [], set()
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                t = _strip_pdf_artifacts(t)
                text.append(t)
                for u in URL_RE.findall(t):
                    urls.add(clean_url(u))
            # try hyperlink objects if available
            try:
                links = getattr(p, "hyperlinks", None)
                if links:
                    for ln in links:
                        uri = ln.get("uri") or ln.get("url")
                        if uri:
                            urls.add(clean_url(uri))
            except Exception:
                pass
    return "\n".join(text), sorted(urls)

def extract_text_from_docx(file) -> Tuple[str, List[str]]:
    doc = docx.Document(file)
    text = "\n".join(p.text for p in doc.paragraphs)
    text = _strip_pdf_artifacts(text)
    urls = sorted({clean_url(u) for u in URL_RE.findall(text)})
    return text, urls

def load_text(upload) -> Tuple[str, List[str]]:
    name = upload.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(upload)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(upload)
    elif name.endswith(".txt"):
        txt = upload.read().decode("utf-8", errors="ignore")
        txt = _strip_pdf_artifacts(txt)
        urls = sorted({clean_url(u) for u in URL_RE.findall(txt)})
        return txt, urls
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")

# -------------------- Parsers --------------------
def find_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text);  return m.group(0) if m else None

def find_phone(text: str) -> Optional[str]:
    candidates = PHONE_RE.findall(text)
    if not candidates: return None
    cleaned = []
    for groups in candidates:
        raw = "".join(groups)
        cleaned.append(re.sub(r"[^\d+]", "", raw))
    cleaned = sorted(set(cleaned), key=lambda s: -len(s))
    return cleaned[0] if cleaned else None

def guess_name(text: str, email: Optional[str]) -> Optional[str]:
    if email:
        user = email.split("@")[0]
        parts = re.split(r"[._\-]", user)
        parts = [p.capitalize() for p in parts if p]
        if parts: return " ".join(parts)
    for line in text.splitlines()[:8]:
        s = line.strip()
        if s and len(s) < 60 and "@" not in s and len(s.split()) <= 4:
            return s
    return None

def _looks_like_heading(line: str) -> Optional[str]:
    raw = line.strip()
    low = raw.lower().strip(": ")
    for h in SECTION_HEADINGS:
        if low == h or low.startswith(h + " "): return h
    if raw.endswith(":") and len(raw) <= 80 and len(raw.split()) <= 8:
        return low.rstrip(":")
    words = raw.split()
    if 0 < len(words) <= 6 and len(raw) <= 80:
        letters = re.sub(r"[^A-Za-z]", "", raw)
        upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0
        title_like = raw == raw.title()
        if (upper_ratio > 0.6 or title_like) and not re.search(r"\d|@", raw):
            return low
    return None

def split_into_sections(text: str) -> Dict[str, str]:
    lines = [l.rstrip() for l in text.splitlines()]
    if not lines: return {"document": ""}

    heading_positions: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        h = _looks_like_heading(line)
        if h: heading_positions.append((idx, h))

    if not heading_positions:
        return {"document": "\n".join(lines)}

    merged = []
    for idx, h in heading_positions:
        if not merged or merged[-1][1] != h:
            merged.append((idx, h))
    heading_positions = merged

    sections: Dict[str, str] = {}
    for i, (pos, h) in enumerate(heading_positions):
        start = pos + 1
        end = heading_positions[i+1][0] if i+1 < len(heading_positions) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        key = h.lower()
        sections[key] = (sections.get(key, "") + ("\n\n" if key in sections else "") + content).strip()
    return sections

def extract_skills(section_text: str) -> List[str]:
    if not section_text: return []
    parts = re.split(r"[\n,•\-;|/]+", section_text)
    skills = [p.strip() for p in parts if p.strip() and len(p.strip()) < 60]
    seen, out = set(), []
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key); out.append(s)
    return out[:400]

def build_parsed_json(filename_hint: str, text: str, links: List[str]) -> Dict[str, Any]:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n\s+\n", "\n\n", text)
    email = find_email(text)
    phone = find_phone(text)
    name = guess_name(text, email)
    sections = split_into_sections(text)
    skills = extract_skills(sections.get("skills") or sections.get("skill") or "")
    return {
        "source_file": filename_hint,
        "parsed_at": int(time.time()),
        "name": name, "email": email, "phone": phone,
        "sections": sections, "skills": skills,
        "links": links
    }

# -------------------- Item splitting --------------------
def split_items(t: str) -> List[str]:
    if not t or not t.strip(): return []
    parts = re.split(r"(?:\n\s*[-•●◦·]\s+|\n{2,})", "\n" + t)
    return [p.strip() for p in parts if p and p.strip() and len(p.strip()) > 2]

# -------------------- Heuristic fallback if no Groq --------------------
def heuristic_groq_fallback(all_sections: Dict[str,str], links: List[str]) -> Dict[str, Any]:
    normalized = {k: v.strip() for k, v in all_sections.items()}

    exp_text = (all_sections.get("experience")
                or all_sections.get("work experience")
                or all_sections.get("professional experience") or "")
    proj_text = all_sections.get("projects") or ""

    def mine_skills(txt: str) -> List[str]:
        return canonicalize_skills(sorted({m.group(0).strip() for m in TECH_WORD.finditer(txt)}, key=str.lower))

    skills_exp = mine_skills(exp_text)
    skills_proj = mine_skills(proj_text)

    # Simple role scoring by keyword presence
    role_scores = [
        ("Data Scientist", 0.0), ("Machine Learning Engineer", 0.0),
        ("Data Analyst", 0.0), ("Data Engineer", 0.0),
        ("MLOps Engineer", 0.0), ("Applied Scientist", 0.0), ("AI Engineer", 0.0)
    ]
    text_all = "\n".join([exp_text, proj_text, all_sections.get("skills","")]).lower()
    add = {
        "data scientist": ["scikit-learn","tensorflow","pytorch","time series","forecast","model","ml"],
        "machine learning engineer": ["pipeline","deployment","mlflow","airflow","docker","api","efficientnet"],
        "data analyst": ["tableau","power bi","excel","visualiz","geopandas","report","analysis"],
        "data engineer": ["spark","pyspark","databricks","kafka","etl","orchestrat","warehouse","dbt"],
        "mlops engineer": ["mlflow","airflow","docker","cicd","monitor","deployment","kubernetes"],
        "applied scientist": ["research","publications","paper","experiments","ablation"],
        "ai engineer": ["llm","rag","prompt","inference","latency","groq","n8n"]
    }
    scores = {}
    for role, _ in role_scores:
        key = role.lower()
        score = sum(1 for kw in add[key] if kw in text_all) / max(1, len(add[key]))
        scores[role] = round(min(1.0, score), 2)

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:5]
    background_roles_ranked = [
        {"role": r, "score": s, "evidence": "Heuristic score based on keywords present across experience, projects, and skills."}
        for r, s in ranked
    ]
    bg_edu = "Education includes degrees and coursework; see normalized sections."

    return {
        "normalized_sections": normalized,
        "skills_from_experience": skills_exp,
        "skills_from_projects": skills_proj,
        "skills_with_evidence": {"experience": [], "projects": []},
        "background_roles_ranked": background_roles_ranked,
        "background_education": bg_edu,
        "links": sorted({clean_url(u) for u in links})
    }

# -------------------- Groq processing (strict schema) --------------------
def groq_process_sections(all_sections: Dict[str, str], links: List[str],
                          model: str = "llama-3.1-8b-instant") -> Dict[str, Any]:
    if not (GROQ_API_KEY and Groq):
        return heuristic_groq_fallback(all_sections, links)

    exp_text = (all_sections.get("experience")
                or all_sections.get("work experience")
                or all_sections.get("professional experience") or "")
    proj_text = all_sections.get("projects") or ""
    experience_items = split_items(exp_text)[:30]
    project_items    = split_items(proj_text)[:30]

    canonical_hints = [
        "Python","R","SQL","PostgreSQL","MySQL","SQLite","SQL Server","PySpark","Spark",
        "Databricks","Apache Airflow","MLflow","Docker","REST APIs","GeoPandas","Pandas","NumPy",
        "scikit-learn","TensorFlow","PyTorch","NLP","Recommendation Systems","LLM",
        "Time Series","Causal Inference","A/B Testing","Tableau","Power BI","Streamlit",
        "Matplotlib","Seaborn","Excel","CPLEX","Java","Julia","C","C++","Git","n8n","Next.js",
        "TypeScript","Tailwind CSS","shadcn/ui","Groq LLM","Flask","OpenCV","EfficientNetB0"
    ]

    role_taxonomy = [
        "Data Scientist","Machine Learning Engineer","Data Analyst",
        "Data Engineer","MLOps Engineer","Applied Scientist","AI Engineer"
    ]

    payload = {
        "instruction": (
            "You will get resume sections and decomposed items from Experience and Projects.\n"
            "Return STRICT JSON with the following keys:\n"
            "1) normalized_sections: object<string,string> — clean each input section (expand acronyms once, keep bullets, no fabrication).\n"
            "2) skills_from_experience: array<string> — DEDUPED, CANONICAL technical skills/tools from experience_items.\n"
            "3) skills_from_projects: array<string> — DEDUPED, CANONICAL technical skills/tools from project_items.\n"
            "4) skills_with_evidence: object with arrays 'experience' and 'projects'. Each element:\n"
            "   { 'skill':'<canonical>', 'sources':[indexes of items], 'evidence': 'short phrase citing what was done' }\n"
            "5) background_roles_ranked: array (max 5) of objects:\n"
            "   { 'role':'<from role_taxonomy>', 'score': 0..1, 'evidence': '1-2 sentences referencing concrete items/sections' }\n"
            "6) background_education: 2-4 sentences on education highlights.\n"
            "7) links: array<string> — EXACTLY the provided URLs, deduped.\n"
            "Guidelines:\n"
            "- Canonicalize names (e.g., TailwindCSS→Tailwind CSS). Keep skill tags short. Infer obvious domain tags (e.g., 'demand forecasting' → 'Time Series').\n"
            "- Never invent employers, dates, or credentials.\n"
        ),
        "sections": all_sections,
        "experience_items": experience_items,
        "project_items": project_items,
        "canonical_skill_hints": canonical_hints,
        "role_taxonomy": role_taxonomy,
        "links": links
    }

    client = Groq(api_key=GROQ_API_KEY)
    msgs = [
        {"role":"system","content":"You are a precise resume information extractor and normalizer; you never fabricate facts."},
        {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.2,
            response_format={"type":"json_object"}
        )
        data = json.loads(resp.choices[0].message.content)

        out = {
            "normalized_sections": data.get("normalized_sections") or {},
            "skills_from_experience": canonicalize_skills(_string_list(data.get("skills_from_experience"))),
            "skills_from_projects": canonicalize_skills(_string_list(data.get("skills_from_projects"))),
            "skills_with_evidence": {
                "experience": data.get("skills_with_evidence", {}).get("experience", []) or [],
                "projects":   data.get("skills_with_evidence", {}).get("projects", []) or [],
            },
            "background_roles_ranked": data.get("background_roles_ranked") or [],
            "background_education": _as_str(data.get("background_education")),
            "links": _norm_links(data.get("links")),
        }
        # ensure normalized contains all originals
        for k, v in all_sections.items():
            out["normalized_sections"].setdefault(k, v)
        # clamp scores
        for r in out.get("background_roles_ranked", []):
            if "score" in r:
                try: r["score"] = max(0.0, min(1.0, float(r["score"])))
                except Exception: r["score"] = 0.0
        return out
    except Exception:
        return heuristic_groq_fallback(all_sections, links)

# -------------------- Embeddings --------------------
def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Resume → Groq processed → Transformers embeddings", page_icon="🧭", layout="centered")
st.title("🧭 Parse → Groq Sections & New Sections → SBERT Embeddings")

uploaded = st.file_uploader("Upload your resume (PDF/DOCX/TXT)", type=["pdf","docx","doc","txt"])

c1, c2 = st.columns([1,1])
with c1:
    emb_model = st.selectbox("Embedding model (Sentence-Transformers)", ["all-MiniLM-L6-v2","all-mpnet-base-v2","multi-qa-MiniLM-L6-cos-v1"], index=0)
with c2:
    include_norm = st.toggle("Embed normalized sections", value=True)

st.caption("Groq normalizes and creates new sections (skills/background/roles/links). Embeddings are computed locally with Sentence-Transformers. `.env` needs GROQ_API_KEY (no quotes).")

if uploaded is not None:
    with st.spinner("Parsing resume…"):
        raw_text, found_links = load_text(uploaded)
        parsed = build_parsed_json(uploaded.name, raw_text, found_links)

    st.subheader("Parsed JSON (raw)")
    st.json(parsed, expanded=False)

    with st.spinner("Processing sections with Groq…"):
        groq_result = groq_process_sections(parsed["sections"], parsed["links"])

    st.subheader("Groq Result (normalized + new sections)")
    st.code(json.dumps(groq_result, ensure_ascii=False, indent=2))

    # -------- Build embedding items --------
    embedding_items: List[Tuple[str, str]] = []

    # A) normalized sections
    if include_norm:
        for sec_name, sec_text in groq_result["normalized_sections"].items():
            if sec_text and sec_text.strip():
                embedding_items.append((f"section::{sec_name}", sec_text.strip()))

    # B) skills sections (as searchable blocks)
    skills_from_exp = groq_result.get("skills_from_experience", [])
    skills_from_proj = groq_result.get("skills_from_projects", [])
    if skills_from_exp:
        embedding_items.append(("section::skills_from_experience", "\n".join(sorted(set(skills_from_exp), key=str.lower))))
    if skills_from_proj:
        embedding_items.append(("section::skills_from_projects", "\n".join(sorted(set(skills_from_proj), key=str.lower))))

    # C) background roles (each role+evidence as its own item)
    for role_obj in groq_result.get("background_roles_ranked", [])[:5]:
        role = role_obj.get("role","")
        score = role_obj.get("score", 0)
        ev = role_obj.get("evidence","")
        if role and ev:
            text = f"Role: {role}\nScore: {score}\nEvidence: {ev}"
            embedding_items.append((f"section::background_role::{role}", text))

    # D) background education
    bg_edu = groq_result.get("background_education", "")
    if bg_edu.strip():
        embedding_items.append(("section::background_education", bg_edu.strip()))

    # E) links block
    links = groq_result.get("links", [])
    if links:
        embedding_items.append(("section::links", "\n".join(links)))

    # -------- Compute embeddings --------
    if embedding_items:
        labels = [lab for lab, _ in embedding_items]
        contents = [txt for _, txt in embedding_items]

        with st.spinner("Embedding with Sentence-Transformers…"):
            vecs = embed_texts(contents, model_name=emb_model)

        st.success(f"Generated {len(contents)} vectors of size {vecs.shape[1]} with {emb_model}.")
        st.write("**Preview (first 6 items, vector head):**")
        for i in range(min(6, len(contents))):
            st.code({
                "label": labels[i],
                "text_sample": (contents[i][:200] + "…") if len(contents[i]) > 220 else contents[i],
                "vector_first_12_dims": vecs[i].tolist()[:12]
            })

        # -------- Single JSON output --------
        out = {
            **parsed,
            "groq_result": groq_result,
            "embeddings": {
                "model": emb_model,
                "items": [
                    {"label": labels[i], "text": contents[i], "vector": vecs[i].tolist()}
                    for i in range(len(contents))
                ]
            }
        }
        st.download_button(
            "⬇️ Download JSON (parsed + groq_result + vectors)",
            data=json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=os.path.splitext(uploaded.name)[0] + f"_groq_{emb_model.replace('/','-')}_with_vectors.json",
            mime="application/json"
        )
    else:
        st.warning("No content to embed after Groq processing.")
else:
    st.info("Upload a resume to begin.")
