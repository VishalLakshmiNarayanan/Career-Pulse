import streamlit as st
import pdfplumber, docx, re, os, json, time
import dateparser

# Try to load spaCy; fall back gracefully if model missing
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

EMAIL_RE = re.compile(r"[a-zA-Z0-9.+_-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?[\d\s.-]{6,15}")
SECTION_HEADINGS = [
    "education", "experience", "work experience", "professional experience",
    "skills", "projects", "certifications", "summary", "about", "publications"
]

def extract_text_from_pdf(file):
    text = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t: text.append(t)
    return "\n".join(text)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def load_text(upload):
    name = upload.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(upload)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(upload)
    elif name.endswith(".txt"):
        return upload.read().decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")

def find_email(text):
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

def find_phone(text):
    candidates = PHONE_RE.findall(text)
    if not candidates: return None
    cleaned = []
    for groups in candidates:
        raw = "".join(groups)
        cleaned.append(re.sub(r"[^\d+]", "", raw))
    cleaned = sorted(set(cleaned), key=lambda s: -len(s))
    return cleaned[0] if cleaned else None

def guess_name(text, email):
    # 1) spaCy NER (if available)
    if nlp:
        doc = nlp(text[:4000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cand = ent.text.strip()
                if 1 < len(cand) < 60 and len(cand.split()) <= 4:
                    return cand
    # 2) from email
    if email:
        username = email.split("@")[0]
        parts = re.split(r"[._\-]", username)
        parts = [p.capitalize() for p in parts if p]
        if parts: return " ".join(parts)
    # 3) first plausible line
    for line in text.splitlines():
        s = line.strip()
        if s and len(s) < 60 and "@" not in s and len(s.split()) <= 4:
            return s
    return None

def split_into_sections(text):
    lines = [l.rstrip() for l in text.splitlines()]
    if not lines: return {"document": ""}
    heading_positions = []
    for idx, line in enumerate(lines):
        low = line.strip().lower()
        for h in SECTION_HEADINGS:
            if low == h or low.startswith(h + ":") or low.startswith(h + " "):
                heading_positions.append((idx, h))
    if not heading_positions:
        return {"document": "\n".join(lines)}
    sections = {}
    for i, (pos, h) in enumerate(heading_positions):
        start = pos + 1
        end = heading_positions[i+1][0] if i+1 < len(heading_positions) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        key = h.lower()
        sections[key] = (sections.get(key, "") + ("\n\n" if key in sections else "") + content).strip()
    return sections

def extract_skills(section_text):
    if not section_text: return []
    parts = re.split(r"[\n,•\-;]+", section_text)
    skills = [p.strip() for p in parts if p.strip() and len(p.strip()) < 60]
    # keep order, remove dups
    seen, out = set(), []
    for s in skills:
        if s.lower() not in seen:
            seen.add(s.lower()); out.append(s)
    return out[:200]

def parse_experience(section_text):
    if not section_text: return []
    items = re.split(r"\n{2,}|\n[-•]\s+|\n\d+\.\s+", section_text)
    parsed = []
    for it in items:
        s = it.strip()
        if not s: continue
        # naive title/company from first line
        lines = [l for l in s.splitlines() if l.strip()]
        first = lines[0] if lines else s
        if " at " in first.lower():
            parts = re.split(r"(?i)\s+at\s+", first, maxsplit=1)
            title, company = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else None)
        elif "," in first:
            parts = first.split(",", 1)
            title, company = parts[0].strip(), parts[1].strip()
        else:
            title, company = first.strip(), None
        # rough date range detection
        m = re.search(r"(\b\d{4}\b|\b[A-Za-z]{3,9}\s*\d{2,4}\b)\s*[-–—to]+\s*(Present|\b\d{4}\b|\b[A-Za-z]{3,9}\s*\d{2,4}\b)", s, flags=re.I)
        dates = m.groups() if m else None
        parsed.append({"raw": s, "title": title, "company": company, "dates": dates})
    return parsed

def resume_to_json(file, filename_hint="Resume"):
    text = load_text(file)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n\s+\n", "\n\n", text)

    email = find_email(text)
    phone = find_phone(text)
    name = guess_name(text, email)
    sections = split_into_sections(text)
    skills = extract_skills(sections.get("skills") or sections.get("skill") or "")
    experience = parse_experience(sections.get("experience") or sections.get("work experience") or "")

    return {
        "source_file": filename_hint,
        "parsed_at": int(time.time()),
        "name": name,
        "email": email,
        "phone": phone,
        "sections": sections,
        "skills": skills,
        "experience": experience,
        "raw_text_snippet": text[:2000]
    }

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Resume → JSON", page_icon="🧾", layout="centered")
st.title("🧾 Resume → JSON Parser")
st.caption("Upload a PDF/DOCX/TXT resume and get structured JSON. Runs locally.")

uploaded = st.file_uploader("Upload your resume file", type=["pdf","docx","doc","txt"])

col1, col2 = st.columns([1,1])

with col1:
    pretty = st.toggle("Pretty JSON", value=True)
with col2:
    show_text = st.toggle("Show raw text snippet", value=False)

if uploaded is not None:
    with st.spinner("Parsing..."):
        data = resume_to_json(uploaded, filename_hint=uploaded.name)

    st.subheader("Parsed JSON")
    if pretty:
        st.json(data, expanded=False)
    else:
        st.code(json.dumps(data, ensure_ascii=False), language="json")

    # Download button
    out_name = os.path.splitext(uploaded.name)[0] + "_parsed.json"
    st.download_button(
        label="⬇️ Download JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=out_name,
        mime="application/json"
    )

    if show_text:
        st.subheader("Raw text (first 2,000 chars)")
        st.code(data["raw_text_snippet"])
else:
    st.info("Choose a file to begin. Supported: PDF, DOCX, DOC, TXT.")
