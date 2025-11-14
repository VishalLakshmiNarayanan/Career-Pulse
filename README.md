<h1><b>Career Pulse</b></h1>
<h3>AI-Powered Resume-Job Matching Platform with Semantic Transparency Scoring</h3>

<p>
Traditional ATS systems match keywords but miss <b>transferable skills</b>, <b>semantic alignment</b>, and <b>role fit</b>.<br>
Career Pulse introduces a semantic intelligence layer that goes beyond keyword matching to measure <b>true job-candidate compatibility</b> using <b>LLM-powered normalization</b>, <b>sentence embeddings</b>, and <b>multi-criteria transparency scoring</b>.
</p>

<p>
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.10+-skyblue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30+-red?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Groq-LLM-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>
</p>

---

## Overview

Career Pulse solves a critical recruitment problem:

> **"How do you match candidates to jobs based on *capability* rather than just *keywords*?"**

This platform provides:

- **For Job Seekers:** Instant transparency scores showing alignment across skills, experience, seniority, and domain, plus AI-generated job-specific resumes
- **For Recruiters:** Automated candidate ranking with semantic matching that finds qualified candidates even if they don't use exact job description terminology

Built on a **dual-portal Streamlit architecture**, Career Pulse processes resumes in any format (PDF/DOCX/TXT), normalizes them using **Groq LLM**, computes **SBERT semantic embeddings**, and calculates **weighted transparency scores** to power both candidate discovery and recruiter efficiency.

---

## Key Features

| Category               | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| **Semantic Matching**  | SBERT embeddings (all-MiniLM-L6-v2) + cosine similarity for role alignment |
| **LLM Normalization**  | Groq llama-3.1-8b-instant extracts skills, experience, seniority, domain   |
| **Transparency Engine**| 4-factor weighted scoring: Skills (50%), Experience (30%), Seniority (10%), Domain (10%) |
| **Resume Intelligence**| Multi-format parsing (PDF/DOCX/TXT) with section detection & skill extraction |
| **ATS Optimization**   | LaTeX-based resume generation with job-specific tailoring via Groq        |
| **Dual Portals**       | Separate interfaces for job seekers (search/apply) and recruiters (post/rank) |
| **Batch Processing**   | Automated ranking of multiple candidates with CSV export                  |
| **Interactive Reports**| Plotly-powered radar charts, gauge visualizations, and AI-generated insights |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CAREER PULSE PLATFORM                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐          ┌─────────────────┐           │
│  │  Job Seeker     │          │   Recruiter     │           │
│  │  Portal         │          │   Portal        │           │
│  └────────┬────────┘          └────────┬────────┘           │
│           │                            │                    │
│           └──────────┬─────────────────┘                    │
│                      │                                      │
│         ┌────────────▼─────────────┐                        │
│         │   Resume Parser          │                        │
│         │   (PDF/DOCX/TXT)         │                        │
│         └────────────┬─────────────┘                        │
│                      │                                      │
│         ┌────────────▼─────────────┐                        │
│         │   Groq LLM Normalizer    │                        │
│         │   (llama-3.1-8b-instant) │                        │
│         └────────────┬─────────────┘                        │
│                      │                                      │
│         ┌────────────▼─────────────┐                        │
│         │   SBERT Embedder         │                        │
│         │   (all-MiniLM-L6-v2)     │                        │
│         └────────────┬─────────────┘                        │
│                      │                                      │
│         ┌────────────▼─────────────┐                        │
│         │  Transparency Scoring    │                        │
│         │  (Weighted Cosine Sim)   │                        │
│         └────────────┬─────────────┘                        │
│                      │                                      │
│         ┌────────────▼─────────────┐                        │
│         │  Output Layer            │                        │
│         │  • Rankings              │                        │ 
│         │  • Reports (Plotly)      │                        │
│         │  • LaTeX Resumes         │                        │
│         │  • CSV Exports           │                        │
│         └──────────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline

<table>
<tr>
<td width="50%" valign="top">

### **1. Resume Processing**
- **Multi-format extraction** (pdfplumber, python-docx)
- **Artifact removal** (cid:###, hyphenation fixes)
- **Section detection** (experience, skills, education, projects)
- **Entity extraction** (name, email, phone, URLs via regex)

</td>
<td width="50%" valign="top">

### **2. LLM Normalization**
- **Groq API** (llama-3.1-8b-instant, temp=0.2)
- **Skill canonicalization** (TailwindCSS to Tailwind CSS)
- **Role ranking** with confidence scores
- **Structured JSON output** (skills, seniority, domain)

</td>
</tr>

<tr>
<td width="50%" valign="top">

### **3. Semantic Embedding**
- **Sentence-BERT** (all-MiniLM-L6-v2)
- **Normalized embeddings** (L2 norm)
- **Cosine similarity** for resume-job alignment
- **Cached transformers** for performance

</td>
<td width="50%" valign="top">

### **4. Transparency Scoring**
- **Skills Match** (50% weight)
- **Experience Alignment** (30% weight)
- **Seniority Level** (10% weight)
- **Domain/Certifications** (10% weight)
- **Composite score** via weighted average

</td>
</tr>
</table>

---

## Transparency Report Components

### **Match Analysis Dashboard**

Each transparency report includes:

1. **Overall Similarity Score**: Semantic cosine similarity (0-100%)
2. **Weighted Transparency Score**: Multi-criteria composite metric
3. **Radar Chart**: Visual breakdown across 4 factors
4. **Gauge Visualization**: Composite score with color-coded ranges
5. **Formula Breakdown**: Transparent calculation showing weights × scores
6. **AI Insights** (Groq-powered):
   - **Match Summary**: Brief compatibility assessment
   - **Strengths**: Top 3 alignment areas
   - **Gaps**: Missing qualifications
   - **Recommendations**: Actionable improvement suggestions

---

## ATS Resume Generation

### **Job-Tailored Resume Pipeline**

1. **Input:** User resume JSON + Target job description
2. **Groq Processing:**
   - Analyzes job requirements
   - Restructures resume sections for relevance
   - Generates optimized professional summary
   - Prioritizes skills by job alignment
3. **LaTeX Templating:**
   - Custom `resume.cls` document class
   - ATS-friendly formatting
   - Parameterized sections (education, skills, experience, projects)
4. **PDF Compilation:** pdflatex rendering
5. **Output:** Download-ready optimized resume PDF

---

## Data Flow

### **Job Seeker Workflow**
```
Upload Resume > Parse & Normalize > Browse Jobs > View Transparency Report
> Generate Optimized Resume > Apply with Match Score
```

### **Recruiter Workflow**
```
Post Job > Receive Applications > Auto-Rank Candidates by Transparency Score
> View Detailed Metrics > Export Ranked CSV
```

---

## Performance Metrics

| Module | Capability | Details |
|--------|-----------|---------|
| **Resume Parser** | Multi-format support | PDF (pdfplumber), DOCX (python-docx), TXT |
| **Groq Normalizer** | Skill canonicalization | ~50 tech skill aliases, role taxonomy (7 categories) |
| **SBERT Embeddings** | Vector dimensions | 384-dim semantic vectors (all-MiniLM-L6-v2) |
| **Transparency Engine** | Scoring criteria | 4-factor weighted system (Skills, Experience, Seniority, Domain) |
| **LaTeX Generator** | Resume optimization | Custom resume.cls template with 5 sections |
| **Batch Processing** | Candidate ranking | Automatic sorting by transparency score + CSV export |

---

## Tech Stack

<p>
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" height="45" />
<img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" height="45" />
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" height="45" />
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" height="45" />
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" height="45" />
<img src="https://avatars.githubusercontent.com/u/56403942?s=200&v=4" height="45" title="Sentence-Transformers"/>
<img src="https://cdn.worldvectorlogo.com/logos/latex.svg" height="45" />
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/git/git-original.svg" height="45" />
</p>

### **Core Dependencies**

| Library | Purpose | Version |
|---------|---------|---------|
| **streamlit** | Web application framework | 1.30+ |
| **groq** | LLM API client | Latest |
| **sentence-transformers** | Semantic embeddings | 2.2+ |
| **scikit-learn** | Cosine similarity | 1.3+ |
| **plotly** | Interactive visualizations | 5.14+ |
| **pdfplumber** | PDF text extraction | 0.10+ |
| **python-docx** | DOCX parsing | 1.1+ |
| **pdflatex** | Resume PDF compilation | System |

---

## Installation

### **Prerequisites**
```bash
# Python 3.10+
python --version

# LaTeX distribution (for resume generation)
# Windows: MiKTeX or TeX Live
# macOS: MacTeX
# Linux: sudo apt-get install texlive-full
```

### **Setup**
```bash
# Clone repository
git clone https://github.com/VishalLakshmiNarayanan/CareerPulse.git
cd CareerPulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Configure environment
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### **Run Application**
```bash
# Job Seeker & Recruiter Portal
streamlit run modules/career_pulse_main.py

# Resume Parser (standalone)
streamlit run modules/resume_to_json.py

# Transparency Search (standalone)
streamlit run modules/transparency_report.py
```

---

## Project Structure

```
Career Pulse/
├── modules/
│   ├── career_pulse_main.py       # Main dual-portal application
│   ├── resume_to_json.py           # Resume parser + Groq processor
│   ├── transparency_report.py      # Job transparency search engine
│   ├── recruiter_pipeline.py       # Standalone recruiter ranking tool
│   └── linkedin_scraper.py         # LinkedIn profile scraper (optional)
├── data/
│   ├── jobs_database.json          # Posted job listings
│   ├── applications_database.json  # User applications
│   ├── users_database.json         # Authentication database
│   └── ai_ml_jobs_linkedin.csv     # Sample job dataset
├── outputs/
│   ├── {user}_{job}_optimized_resume.pdf
│   ├── {user}_{job}_optimized_resume.tex
│   └── latest_resume_output.json   # Processed resume connector
├── resume.cls                      # LaTeX resume template
├── .env                            # API keys (GROQ_API_KEY)
└── requirements.txt
```

---

## Usage Examples

### **1. Job Seeker: Upload Resume & View Transparency**
```python
# Upload resume (PDF/DOCX/TXT)
# System automatically:
# 1. Parses sections (experience, skills, education, projects)
# 2. Normalizes via Groq (canonical skills, role ranking)
# 3. Generates SBERT embeddings

# Browse jobs > Click "View Transparency Report"
# Output:
# • Overall Match: 78.3%
# • Transparency Score: 82.1%
# • Radar Chart: Skills (0.85), Experience (0.79), Seniority (0.81), Domain (0.83)
# • AI Insights: "Strong ML background; gap in cloud deployment experience"
```

### **2. Job Seeker: Generate Optimized Resume**
```python
# Select target job > Click "View Transparency Report"
# System generates job-tailored resume via Groq:
# • Rewrites professional summary for role alignment
# • Prioritizes relevant skills
# • Restructures experience with role-specific achievements
# • Compiles LaTeX to PDF

# Download: {username}_{jobid}_optimized_resume.pdf
```

### **3. Recruiter: Post Job & Rank Candidates**
```python
# Recruiter Portal > Post Job
job = {
    "title": "Senior ML Engineer",
    "company": "Tech Corp",
    "location": "Remote",
    "description": "Looking for ML engineer with production experience..."
}

# Candidates apply > Automatic ranking by transparency score
# View Applications:
# Rank | Candidate       | Email              | Match % | Transparency %
# 1    | Alice Johnson   | alice@example.com  | 87.2    | 89.5
# 2    | Bob Smith       | bob@example.com    | 82.1    | 85.3
# 3    | Carol Lee       | carol@example.com  | 78.9    | 81.7

# Export > applications_{jobid}.csv
```

---

## Transparency Scoring Formula

```
Transparency Score = Weighted Average of Factor Scores

Factor Scores (via Semantic Similarity):
• Skills Score          = cosine_sim(resume_skills_embedding, job_skills_embedding)
• Experience Score      = cosine_sim(resume_exp_embedding, job_exp_embedding)
• Seniority Score       = cosine_sim(resume_seniority_embedding, job_seniority_embedding)
• Domain Score          = cosine_sim(resume_domain_embedding, job_domain_embedding)

Weights:
• Skills:               50%
• Experience:           30%
• Seniority:            10%
• Domain/Certs:         10%

Final Score = (0.5 × Skills) + (0.3 × Experience) + (0.1 × Seniority) + (0.1 × Domain)
```

**Why These Weights?**
- **Skills (50%):** Primary technical qualifier
- **Experience (30%):** Years and project complexity matter
- **Seniority (10%):** Role level alignment (junior/mid/senior)
- **Domain (10%):** Industry-specific knowledge

---

## Sample Transparency Report

```
Job: Senior Data Scientist @ Analytics Corp
Candidate: John Doe

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATCH SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Similarity:      84.7%
Transparency Score:      86.2%

FACTOR BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Skills:          0.89 × 0.50 = 0.445
Experience:      0.82 × 0.30 = 0.246
Seniority:       0.87 × 0.10 = 0.087
Domain:          0.84 × 0.10 = 0.084
─────────────────────────────────────
Final Score:              0.862 (86.2%)

AI INSIGHTS (Groq)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strengths:
• Strong Python/scikit-learn background with 4+ ML projects
• Time series forecasting experience aligns with role needs
• Tableau + Power BI proficiency matches visualization requirements

Gaps:
• Limited cloud deployment experience (AWS/GCP)
• No explicit mention of A/B testing frameworks

Recommendations:
• Highlight any AWS/cloud projects in resume
• Add certifications in cloud ML (SageMaker, Vertex AI)
• Include A/B testing metrics from past projects
```

---

## Resume Class Template Features

The custom LaTeX `resume.cls` provides:

| Command | Purpose |
|---------|---------|
| `\introduction` | Header with name, email, phone, LinkedIn, GitHub |
| `\summary` | Professional summary section |
| `\educationSection` | Education items with university, GPA, coursework |
| `\skillsSection` | Categorized technical skills |
| `\experienceSection` | Work experience with achievements |
| `\projectItem` | Project highlights with details |

**Optimizations:**
- ATS-friendly single-column layout
- Standard fonts (Computer Modern)
- Consistent spacing and margins
- Bullet-point achievements
- No graphics/images (ensures ATS compatibility)

---

## Research Directions

- **Neural Ranking Models:** Train transformer-based re-rankers on transparency scores
- **Multi-Modal Embeddings:** Incorporate certifications, GitHub activity, portfolio projects
- **Bias Detection:** Analyze demographic fairness in transparency scoring
- **Dynamic Weight Optimization:** Learn optimal weights per job category
- **Skill Graph Integration:** Build knowledge graphs for skill relationships and transferability
- **Temporal Modeling:** Account for recency and progression of experience
- **Interview Prediction:** Extend transparency scores to predict interview success probability

---

## Contributing

Contributions are welcome! Areas of interest:
- Additional resume formats (LinkedIn JSON, HTML)
- Alternative embedding models (OpenAI, Cohere)
- Custom LaTeX templates
- Advanced visualization options
- Additional LLM providers (Anthropic, OpenAI)

---

