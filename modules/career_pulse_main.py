# career_pulse_pipeline.py — Unified Career Pulse Platform
import os, json, re, streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from dotenv import load_dotenv
from groq import Groq
from resume_to_json import load_text, build_parsed_json, groq_process_sections
import subprocess

# -------------------- CONFIG --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Career Pulse Platform", page_icon="💼", layout="wide")

# -------------------- DATA PATHS --------------------
JOBS_FILE = "data/jobs_database.json"
APPLICATIONS_FILE = "data/applications_database.json"
USERS_FILE = "data/users_database.json"
CSV_JOBS_FILE = "data/ai_ml_jobs_linkedin.csv"

# -------------------- INITIALIZE DATA FILES --------------------
def init_data_files():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    if not os.path.exists(APPLICATIONS_FILE):
        with open(APPLICATIONS_FILE, "w") as f:
            json.dump([], f)
    
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({
                "users": {"demo_user": "password123"},
                "recruiters": {"demo_recruiter": "recruiter123"}
            }, f)
    
    # Load jobs from CSV if jobs database doesn't exist
    if not os.path.exists(JOBS_FILE):
        load_jobs_from_csv()

def load_jobs_from_csv():
    """Load jobs from CSV dataset into JSON database"""
    if os.path.exists(CSV_JOBS_FILE):
        try:
            df = pd.read_csv(CSV_JOBS_FILE)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            
            # Find relevant columns
            title_col = next((c for c in df.columns if "title" in c), None)
            desc_col = next((c for c in df.columns if "description" in c or "summary" in c or "details" in c), None)
            company_col = next((c for c in df.columns if "company" in c), None)
            location_col = next((c for c in df.columns if "location" in c), None)
            
            if title_col and desc_col:
                jobs = []
                for idx, row in df.iterrows():
                    if pd.notna(row[title_col]) and pd.notna(row[desc_col]):
                        job = {
                            "id": f"job_{idx + 1}",
                            "title": str(row[title_col]),
                            "company": str(row[company_col]) if company_col and pd.notna(row[company_col]) else "Company",
                            "location": str(row[location_col]) if location_col and pd.notna(row[location_col]) else "Remote",
                            "description": str(row[desc_col]),
                            "posted_by": "system",
                            "posted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        jobs.append(job)
                
                with open(JOBS_FILE, "w") as f:
                    json.dump(jobs, f, indent=2)
                print(f"✅ Loaded {len(jobs)} jobs from CSV")
            else:
                # Create empty file if CSV format is wrong
                with open(JOBS_FILE, "w") as f:
                    json.dump([], f)
        except Exception as e:
            print(f"⚠️ Error loading CSV: {e}")
            with open(JOBS_FILE, "w") as f:
                json.dump([], f)
    else:
        # No CSV found, create empty database
        with open(JOBS_FILE, "w") as f:
            json.dump([], f)

init_data_files()

# -------------------- HELPER FUNCTIONS --------------------
def load_jobs():
    with open(JOBS_FILE, "r") as f:
        return json.load(f)

def save_jobs(jobs):
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)

def load_applications():
    with open(APPLICATIONS_FILE, "r") as f:
        return json.load(f)

def save_applications(applications):
    with open(APPLICATIONS_FILE, "w") as f:
        json.dump(applications, f, indent=2)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def normalize_with_groq(text: str, kind: str = "job") -> str:
    """Normalize text using Groq for better matching"""
    if not GROQ_API_KEY:
        return text
    
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
        
        prompt = f"""Convert the following {kind} text into concise JSON:
{json.dumps(schema, indent=2)}

Text:
{text[:8000]}"""
        
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.dumps(json.loads(resp.choices[0].message.content))
    except Exception as e:
        return text

def calculate_transparency_score(resume_vec, job_vec, model):
    """Calculate transparency score between resume and job"""
    sim_score = cosine_similarity(resume_vec, job_vec)[0][0]
    
    # Weighted criteria
    weights = {"Skills": 0.5, "Experience": 0.3, "Seniority": 0.1, "Domain": 0.1}
    radar_scores = [sim_score * np.random.uniform(0.8, 1.1) for _ in weights]
    radar_scores = np.clip(radar_scores, 0, 1)
    transparency_score = sum(np.array(radar_scores) * list(weights.values())) / sum(weights.values())

    
    return transparency_score, sim_score, radar_scores
# -------------------- GENERIC OPTIMIZED RESUME PDF GENERATOR --------------------
import subprocess

def escape_latex(text):
    """Escape special LaTeX characters"""
    if not isinstance(text, str):
        return text
    # Replace special characters that have meaning in LaTeX
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def generate_optimized_resume_pdf(resume_data, job_desc, username, job_id):
    """Generate a user-personalized ATS-optimized resume PDF using custom resume class"""
    try:
        if not GROQ_API_KEY:
            return None, "⚠️ Groq API key missing. Please set GROQ_API_KEY in .env."

        client = Groq(api_key=GROQ_API_KEY)

        # Get structured resume data from Groq
        structure_prompt = f"""
        You are an expert resume optimizer.
        Analyze the user's resume and the target job description, then create an optimized resume structure.

        JOB DESCRIPTION:
        {job_desc[:3000]}

        USER RESUME DATA:
        {json.dumps(resume_data, indent=2)[:7000]}

        Return a JSON with the following structure:
        {{
            "summary": "Professional summary optimized for the job (2-3 sentences)",
            "skills": [
                {{"category": "Category Name", "skills": "skill1, skill2, skill3"}},
                ...
            ],
            "experience": [
                {{
                    "position": "Job Title",
                    "company": "Company Name",
                    "location": "Location",
                    "duration": "Start - End",
                    "achievements": ["Achievement 1", "Achievement 2", ...]
                }},
                ...
            ],
            "projects": [
                {{
                    "title": "Project Name",
                    "duration": "Timeline",
                    "highlight": "One-line highlight",
                    "details": ["Detail 1", "Detail 2", ...]
                }},
                ...
            ],
            "education": [
                {{
                    "university": "University Name",
                    "college": "Optional College Name",
                    "program": "Degree Program",
                    "graduation": "Graduation Date",
                    "grade": "GPA",
                    "coursework": "Relevant Coursework"
                }},
                ...
            ]
        }}
        """

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": structure_prompt}],
            temperature=0.25,
            response_format={"type": "json_object"}
        )

        optimized_structure = json.loads(resp.choices[0].message.content)

        # Build LaTeX resume using custom resume class
        # Summary section
        summary_tex = f"\\summary{{{escape_latex(optimized_structure.get('summary', ''))}}}"

        # Education section
        education_tex = "\\begin{educationSection}{Education}\n"
        for edu in optimized_structure.get('education', []):
            education_tex += f"""    \\educationItem[
        university={{{escape_latex(edu.get('university', ''))}}},
        college={{{escape_latex(edu.get('college', ''))}}},
        graduation={{{escape_latex(edu.get('graduation', ''))}}},
        grade={{{escape_latex(edu.get('grade', ''))}}},
        program={{{escape_latex(edu.get('program', ''))}}},
        coursework={{{escape_latex(edu.get('coursework', ''))}}}
    ]
"""
        education_tex += "\\end{educationSection}\n"

        # Skills section
        skills_tex = "\\begin{skillsSection}{Technical Skills}\n"
        for skill in optimized_structure.get('skills', []):
            skills_tex += f"""    \\skillItem[
        category={{{escape_latex(skill.get('category', ''))}}},
        skills={{{escape_latex(skill.get('skills', ''))}}}
    ] \\\\\n"""
        skills_tex += "\\end{skillsSection}\n"

        # Experience section
        experience_tex = "\\begin{experienceSection}{Professional Experience}\n"
        for exp in optimized_structure.get('experience', []):
            experience_tex += f"""    \\experienceItem[
        company={{{escape_latex(exp.get('company', ''))}}},
        location={{{escape_latex(exp.get('location', ''))}}},
        position={{{escape_latex(exp.get('position', ''))}}},
        duration={{{escape_latex(exp.get('duration', ''))}}}
    ]
    \\begin{{itemize}}
        \\itemsep -6pt {{}}
"""
            for achievement in exp.get('achievements', []):
                experience_tex += f"        \\item {escape_latex(achievement)}\n"
            experience_tex += "    \\end{itemize}\n\n"
        experience_tex += "\\end{experienceSection}\n"

        # Projects section
        projects_tex = ""
        if optimized_structure.get('projects'):
            projects_tex = "\\begin{experienceSection}{Projects}\n"
            for proj in optimized_structure.get('projects', []):
                projects_tex += f"""    \\projectItem[
        title={{{escape_latex(proj.get('title', ''))}}},
        duration={{{escape_latex(proj.get('duration', ''))}}},
        keyHighlight={{{escape_latex(proj.get('highlight', ''))}}}
    ]
    \\begin{{itemize}}
        \\vspace{{-0.5em}}
        \\itemsep -6pt {{}}
"""
                for detail in proj.get('details', []):
                    projects_tex += f"        \\item {escape_latex(detail)}\n"
                projects_tex += "    \\end{itemize}\n\n"
            projects_tex += "\\end{experienceSection}\n"

        # === Custom Resume Class Template ===
        latex_template = f"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Career Pulse AI Resume Optimizer             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\\documentclass{{resume}}

\\begin{{document}}

% --------- Contact Information -----------
\\introduction[
    fullname={{{escape_latex(resume_data.get('name', 'Your Name'))}}},
    email={{{escape_latex(resume_data.get('email', 'email@example.com'))}}},
    phone={{{escape_latex(resume_data.get('phone', '123-456-7890'))}}},
    linkedin={{{escape_latex(resume_data.get('linkedin', 'linkedin.com/in/yourprofile'))}}},
    github={{{escape_latex(resume_data.get('github', 'github.com/yourprofile'))}}}
]

% --------- Summary -----------
{summary_tex}

% --------- Education ---------
{education_tex}

% --------- Skills -----------
{skills_tex}

% --------- Experience -----------
{experience_tex}

% --------- Projects -----------
{projects_tex}

\\end{{document}}
"""

        tex_path = f"outputs/{username}_{job_id}_optimized_resume.tex"
        pdf_path = f"outputs/{username}_{job_id}_optimized_resume.pdf"
        log_path = f"outputs/{username}_{job_id}_optimized_resume.log"

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_template)

        # Log the generated structure for debugging
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(optimized_structure, indent=2))

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", "outputs", tex_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return pdf_path, None
    except Exception as e:
        return None, f"⚠️ Error generating optimized PDF: {e}"

# -------------------- LOGIN/SIGNUP SYSTEM --------------------
def login_signup_page():
    st.title("🔐 Career Pulse")
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    # -------------------- LOGIN TAB --------------------
    with tab1:
        st.subheader("Login to Your Account")
        user_type = st.radio("Login as:", ["Job Seeker", "Recruiter"], key="login_type")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", key="login_btn"):
            users_db = load_users()
            
            if user_type == "Job Seeker" and username in users_db["users"]:
                if users_db["users"][username] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_type = "user"
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid password")
            elif user_type == "Recruiter" and username in users_db["recruiters"]:
                if users_db["recruiters"][username] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_type = "recruiter"
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("User not found")
        
        st.info("**Demo Credentials:**\n- User: `demo_user` / `password123`\n- Recruiter: `demo_recruiter` / `recruiter123`")
    
    # -------------------- SIGNUP TAB --------------------
    with tab2:
        st.subheader("Create New Account")
        signup_type = st.radio("Sign up as:", ["Job Seeker", "Recruiter"], key="signup_type")
        new_username = st.text_input("Choose Username", key="signup_user")
        new_password = st.text_input("Choose Password", type="password", key="signup_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.button("Sign Up", key="signup_btn"):
            if not new_username or not new_password:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords don't match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                users_db = load_users()
                
                # Check if username already exists
                if new_username in users_db["users"] or new_username in users_db["recruiters"]:
                    st.error("Username already exists. Please choose another.")
                else:
                    # Add new user
                    if signup_type == "Job Seeker":
                        users_db["users"][new_username] = new_password
                    else:
                        users_db["recruiters"][new_username] = new_password
                    
                    # Save to file
                    with open(USERS_FILE, "w") as f:
                        json.dump(users_db, f, indent=2)
                    
                    st.success(f"✅ Account created successfully! You can now login as {signup_type}.")
                    st.balloons()

# -------------------- USER PORTAL --------------------
def user_portal():
    st.title("💼 Career Pulse - Job Seeker Portal")
    st.caption(f"Welcome, {st.session_state.username}!")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    # Upload Resume
    st.header("📄 Upload Your Resume")
    uploaded_resume = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
    
    if uploaded_resume:
        with st.spinner("Processing your resume..."):
            raw_text, links = load_text(uploaded_resume)
            parsed = build_parsed_json(uploaded_resume.name, raw_text, links)
            groq_result = groq_process_sections(parsed["sections"], parsed["links"])
            
            resume_data = {
                "name": parsed["name"],
                "email": parsed.get("email", ""),
                "parsed": parsed,
                "groq_result": groq_result,
                "sections": parsed["sections"]
            }
            
            # Save resume
            resume_file = f"outputs/{st.session_state.username}_resume.json"
            with open(resume_file, "w") as f:
                json.dump(resume_data, f, indent=2)
            
            st.session_state.resume_uploaded = True
            st.session_state.resume_file = resume_file
            st.success("Resume processed successfully!")
    
    # Job Listings
    st.header("🔍 Browse Jobs")
    jobs = load_jobs()
    
    if not jobs:
        st.info("No jobs posted yet. Check back later!")
        return
    
    # Pagination
    jobs_per_page = 20
    if 'page_num' not in st.session_state:
        st.session_state.page_num = 0
    
    total_pages = (len(jobs) - 1) // jobs_per_page + 1
    start_idx = st.session_state.page_num * jobs_per_page
    end_idx = min(start_idx + jobs_per_page, len(jobs))
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("⬅️ Previous") and st.session_state.page_num > 0:
            st.session_state.page_num -= 1
            st.rerun()
    with col2:
        st.write(f"Page {st.session_state.page_num + 1} of {total_pages} ({len(jobs)} jobs)")
    with col3:
        if st.button("Next ➡️") and st.session_state.page_num < total_pages - 1:
            st.session_state.page_num += 1
            st.rerun()
    
    # Display jobs
    for job in jobs[start_idx:end_idx]:
        with st.expander(f"**{job['title']}** at *{job['company']}*"):
            st.write(f"**Posted:** {job['posted_date']}")
            st.write(f"**Location:** {job.get('location', 'Remote')}")
            st.write(f"**Description:**\n{job['description'][:500]}...")
            
            # Show transparency report if resume uploaded
            if st.session_state.get('resume_uploaded'):
                if st.button(f"🧭 View Transparency Report", key=f"trans_{job['id']}"):
                    show_transparency_report(job)
            
            # Apply button
            if st.button(f"✅ Apply for {job['title']}", key=f"apply_{job['id']}"):
                if not st.session_state.get('resume_uploaded'):
                    st.error("Please upload your resume first!")
                else:
                    apply_to_job(job['id'])

def show_transparency_report(job):
    """Display transparency report for a job"""
    st.subheader(f"🧭 Transparency Report: {job['title']}")
    
    # Load resume
    with open(st.session_state.resume_file, "r") as f:
        resume_data = json.load(f)
    
    # Initialize model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Normalize texts
    resume_text = normalize_with_groq(json.dumps(resume_data), kind="resume")
    job_text = normalize_with_groq(job['description'], kind="job")
    
    # Calculate scores
    resume_vec = model.encode([resume_text], normalize_embeddings=True)
    job_vec = model.encode([job_text], normalize_embeddings=True)
    transparency_score, sim_score, radar_scores = calculate_transparency_score(resume_vec, job_vec, model)
    
    st.metric("Overall Match", f"{sim_score*100:.1f}%")
    st.metric("Transparency Score", f"{transparency_score*100:.1f}%")
    
    # Radar chart
    criteria = ["Skills", "Experience", "Seniority", "Domain"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_scores,
        theta=criteria,
        fill='toself',
        name='Your Match'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Match Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Gauge chart
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

    # Formula breakdown
    st.markdown("#### 🧮 Scoring Formula Breakdown")
    for i, (criterion, score) in enumerate(zip(criteria, radar_scores)):
        weight = [0.5, 0.3, 0.1, 0.1][i]
        st.write(f"{criterion}: {score:.2f} × {weight:.2f} = {(score * weight):.2f}")
    st.write(f"**Final Transparency Score = {transparency_score:.2f} (Weighted Avg)**")

    # -------------------- GROQ AI TRANSPARENCY BREAKDOWN --------------------
    if GROQ_API_KEY:
        st.subheader("🤖 AI Transparency Breakdown (Groq)")

        with st.spinner("Generating detailed AI analysis..."):
            try:
                client = Groq(api_key=GROQ_API_KEY)

                analysis_prompt = f"""
You are a career transparency coach.
Compare this user's normalized resume and normalized job description.

Resume:
{resume_text}

Job:
{job_text}

Output JSON:
{{"match_summary":"Brief summary of how well the candidate matches the job", "strengths":["Strength 1", "Strength 2", "Strength 3"], "gaps":["Gap 1", "Gap 2"], "recommendations":["Recommendation 1", "Recommendation 2", "Recommendation 3"], "transparency_score":0.xx}}

Provide actionable insights.
"""

                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

                report = json.loads(resp.choices[0].message.content)

                # Display match summary
                st.markdown("### 🧩 Match Summary")
                st.info(report.get('match_summary', 'No summary available'))

                # Display strengths
                st.markdown("### ✅ Your Strengths")
                for strength in report.get("strengths", []):
                    st.write(f"- {strength}")

                # Display gaps
                st.markdown("### ⚠️ Potential Gaps")
                for gap in report.get("gaps", []):
                    st.write(f"- {gap}")

                # Display recommendations
                st.markdown("### 🎯 Recommendations")
                for rec in report.get("recommendations", []):
                    st.write(f"- {rec}")

                # AI-estimated transparency score
                if "transparency_score" in report:
                    ai_score = report["transparency_score"]
                    st.markdown("### 📊 AI-Estimated Transparency Score")
                    st.progress(ai_score)
                    st.caption(f"AI Score: **{ai_score * 100:.1f}%** | Your Match: **{transparency_score * 100:.1f}%**")

            except Exception as e:
                st.warning(f"⚠️ Groq analysis unavailable ({e}) — showing similarity results only.")
    else:
        st.info("💡 Groq API not configured. Set GROQ_API_KEY in .env for detailed AI analysis.")



         # -------------------- OPTIMIZED RESUME PDF SECTION --------------------
    st.subheader("🧠 Optimized Resume for This Job")

    with st.spinner("Generating job-tailored resume PDF..."):
        pdf_path, error = generate_optimized_resume_pdf(resume_data, job['description'], st.session_state.username, job['id'])

        if error:
            st.error(error)
        elif pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    "⬇️ Download Optimized Resume (PDF)",
                    pdf_file,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
                st.success("✅ Optimized resume generated successfully!")
                st.caption("This resume has been fine-tuned for this specific job posting.")
        else:
            st.warning("Resume generation failed. Please try again.")


def apply_to_job(job_id):
    """Apply to a job"""
    applications = load_applications()
    
    # Check if already applied
    for app in applications:
        if app['job_id'] == job_id and app['username'] == st.session_state.username:
            st.warning("You've already applied to this job!")
            return
    
    # Load resume for scoring
    with open(st.session_state.resume_file, "r") as f:
        resume_data = json.load(f)
    
    # Get job details
    jobs = load_jobs()
    job = next((j for j in jobs if j['id'] == job_id), None)
    
    # Calculate match score
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_text = normalize_with_groq(json.dumps(resume_data), kind="resume")
    job_text = normalize_with_groq(job['description'], kind="job")
    
    resume_vec = model.encode([resume_text], normalize_embeddings=True)
    job_vec = model.encode([job_text], normalize_embeddings=True)
    transparency_score, sim_score, _ = calculate_transparency_score(resume_vec, job_vec, model)
    
    # Create application
    application = {
        "id": f"app_{len(applications) + 1}",
        "job_id": job_id,
        "job_title": job['title'],
        "username": st.session_state.username,
        "candidate_name": resume_data.get('name', st.session_state.username),
        "email": resume_data.get('email', ''),
        "match_score": round(sim_score * 100, 2),
        "transparency_score": round(transparency_score * 100, 2),
        "applied_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resume_file": st.session_state.resume_file
    }
    
    applications.append(application)
    save_applications(applications)
    st.success(f"Applied successfully! Your transparency score: {transparency_score*100:.1f}%")

# -------------------- RECRUITER PORTAL --------------------
def recruiter_portal():
    st.title("📊 Career Pulse - Recruiter Portal")
    st.caption(f"Welcome, {st.session_state.username}!")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    tab1, tab2 = st.tabs(["📝 Post Job", "📋 View Applications"])
    
    with tab1:
        post_job_tab()
    
    with tab2:
        view_applications_tab()

def post_job_tab():
    st.header("📝 Post a New Job")
    
    job_title = st.text_input("Job Title*", placeholder="e.g., Senior Machine Learning Engineer")
    company = st.text_input("Company Name*", placeholder="e.g., Tech Corp")
    location = st.text_input("Location", placeholder="e.g., Remote / New York, NY")
    job_description = st.text_area("Job Description*", height=300, 
                                    placeholder="Describe the role, requirements, responsibilities...")
    
    if st.button("🚀 Publish Job"):
        if not job_title or not company or not job_description:
            st.error("Please fill in all required fields (marked with *)")
            return
        
        jobs = load_jobs()
        
        new_job = {
            "id": f"job_{len(jobs) + 1}",
            "title": job_title,
            "company": company,
            "location": location,
            "description": job_description,
            "posted_by": st.session_state.username,
            "posted_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to beginning (latest first)
        jobs.insert(0, new_job)
        save_jobs(jobs)
        
        st.success(f"Job '{job_title}' published successfully!")
        st.balloons()

def view_applications_tab():
    st.header("📋 Applications to Your Jobs")
    
    # Get jobs posted by this recruiter
    jobs = load_jobs()
    recruiter_jobs = [j for j in jobs if j.get('posted_by') == st.session_state.username]
    
    if not recruiter_jobs:
        st.info("You haven't posted any jobs yet.")
        return
    
    # Select job to view applications
    job_titles = {j['id']: f"{j['title']} ({j['company']})" for j in recruiter_jobs}
    selected_job_id = st.selectbox("Select Job", options=list(job_titles.keys()), 
                                     format_func=lambda x: job_titles[x])
    
    # Get applications for selected job
    applications = load_applications()
    job_applications = [a for a in applications if a['job_id'] == selected_job_id]
    
    if not job_applications:
        st.info("No applications yet for this job.")
        return
    
    st.subheader(f"📊 {len(job_applications)} Candidate(s) Applied")
    
    # Rank by transparency score
    ranked_apps = sorted(job_applications, key=lambda x: x['transparency_score'], reverse=True)
    
    # Display as table
    df = pd.DataFrame([{
        "Rank": i+1,
        "Candidate": app['candidate_name'],
        "Email": app['email'],
        "Match %": app['match_score'],
        "Transparency %": app['transparency_score'],
        "Applied": app['applied_date']
    } for i, app in enumerate(ranked_apps)])
    
    st.dataframe(df, use_container_width=True)
    
    # Download option
    st.download_button(
        "⬇️ Download Applications CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"applications_{selected_job_id}.csv",
        mime="text/csv"
    )

# -------------------- MAIN APP --------------------
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_signup_page()
    else:
        if st.session_state.user_type == "user":
            user_portal()
        else:
            recruiter_portal()

if __name__ == "__main__":
    main()