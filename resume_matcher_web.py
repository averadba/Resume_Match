import streamlit as st
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
import os

# --------- Configuration ---------
EMB_MODEL_ID = 'all-MiniLM-L6-v2'

# Use Streamlit secrets for your Hugging Face token!
HF_TOKEN = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else os.environ.get("HF_TOKEN", "")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/sshleifer/tiny-gpt2"  # You can swap to another if rate limited

# --------- Download/Cache Embedding Model ---------
@st.cache_resource(show_spinner=True)
def load_embedder():
    try:
        model = SentenceTransformer(EMB_MODEL_ID)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()
    return model

# --------- Helper Functions ---------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([p.text for p in doc.paragraphs])

def get_text(file):
    if file.name.lower().endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.name.lower().endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        return ""

def hf_llm_explain(resume_text, job_text, max_chars=1200):
    prompt = (
        "You are an HR recruiter. Given the following RESUME and JOB DESCRIPTION, do all of the following:\n"
        "1. List 3-6 *specific skills or experiences* from the RESUME that match the JOB DESCRIPTION. "
        "Use bullet points. Only use evidence from the resume.\n"
        "2. Then, write a short paragraph (5-8 sentences) explaining why the candidate is a good fit for this job, "
        "citing their skills, experience, and education from the resume, and referencing the job's requirements. "
        "DO NOT copy the job description or resume text; instead, synthesize a fresh explanation based on overlaps.\n\n"
        f"RESUME:\n{resume_text[:max_chars]}\n\nJOB DESCRIPTION:\n{job_text[:max_chars]}\n\n"
        "First, list the matching skills/experiences from the resume. Second, give the explanation."
    )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 350,
            "temperature": 0.4,
            "do_sample": True
        }
    }
    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            # Hugging Face sometimes returns a list or dict
            if isinstance(data, list):
                explanation = data[0].get("generated_text", "")
            else:
                explanation = data.get("generated_text", "")
            explanation = explanation.strip() or "The AI could not generate an explanation. Please review manually."
        elif response.status_code == 503:
            explanation = "Model is currently loading or too busy. Please try again in a minute."
        elif response.status_code == 429:
            explanation = "Rate limit exceeded. Try again later or use your own HF endpoint."
        else:
            explanation = f"Error from Hugging Face: {response.text}"
    except Exception as e:
        explanation = f"Error communicating with Hugging Face: {e}"
    return explanation

# --------- Streamlit UI ---------
st.set_page_config(page_title="Resume-to-Job Matcher (Hugging Face LLM)", layout="wide")
st.title("📄🤖 Resume-to-Job Matcher (Hugging Face LLM, Deployable)")

st.write("""
Upload multiple resumes and job descriptions (PDF or DOCX). The app will show best matches and AI-generated explanations using a free open LLM from Hugging Face.
**Note:** For real production/corporate use, self-host the LLM to ensure data confidentiality.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Resumes")
    resumes = st.file_uploader("Resumes", type=['pdf', 'docx'], accept_multiple_files=True)

with col2:
    st.header("Upload Job Descriptions")
    jobs = st.file_uploader("Job Descriptions", type=['pdf', 'docx'], accept_multiple_files=True)

top_n = st.slider("Show top N matches per job", 1, 10, 3)
run_button = st.button("Run")

if resumes and jobs and run_button:
    st.info("Processing resumes and job descriptions...")

    num_resumes = len(resumes)
    num_jobs = len(jobs)
    max_steps = num_resumes + num_jobs + 2 + num_jobs  # Extraction + embeddings + per-job explain
    current_step = 0
    progress_bar = st.progress(0.0, text="Starting...")

    # Step 1: Extract texts
    resume_texts = {}
    for file in resumes:
        resume_texts[file.name] = get_text(file)
        current_step += 1
        progress_bar.progress(
            min(current_step / max_steps, 1.0),
            text=f"Extracting resumes... ({current_step}/{max_steps})"
        )

    job_texts = {}
    for file in jobs:
        job_texts[file.name] = get_text(file)
        current_step += 1
        progress_bar.progress(
            min(current_step / max_steps, 1.0),
            text=f"Extracting job descriptions... ({current_step}/{max_steps})"
        )

    # Step 2: Embeddings
    with st.spinner("Generating semantic embeddings..."):
        embedder = load_embedder()
        resume_embs = {k: embedder.encode(v) for k, v in resume_texts.items()}
        current_step += 1
        progress_bar.progress(
            min(current_step / max_steps, 1.0),
            text=f"Embedding resumes... ({current_step}/{max_steps})"
        )
        job_embs = {k: embedder.encode(v) for k, v in job_texts.items()}
        current_step += 1
        progress_bar.progress(
            min(current_step / max_steps, 1.0),
            text=f"Embedding job descriptions... ({current_step}/{max_steps})"
        )

    # Step 3: Compute similarity and display (with progress per job)
    total_jobs = len(job_embs)
    completed_jobs = 0

    for job_name, job_vec in job_embs.items():
        st.subheader(f"Top matches for **{job_name}**")
        # Compute similarities
        sims = []
        for res_name, res_vec in resume_embs.items():
            score = cosine_similarity([job_vec], [res_vec])[0][0]
            sims.append((res_name, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        top_resumes = sims[:top_n]

        # Main best match
        if top_resumes:
            best_res_name, best_score = top_resumes[0]
            with st.expander(f"Best Match: {best_res_name} (Similarity: {best_score:.2f})"):
                st.markdown(f"**Resume Preview:**\n\n{resume_texts[best_res_name][:400]}...")
                st.markdown(f"**Job Description Preview:**\n\n{job_texts[job_name][:400]}...")
                with st.spinner("Generating detailed explanation (using Hugging Face LLM, may take up to 60s)..."):
                    explanation = hf_llm_explain(resume_texts[best_res_name], job_texts[job_name])
                    st.markdown(f"**Why it matches:** {explanation.strip()}")

        # Runner-ups
        if len(top_resumes) > 1:
            st.markdown("#### Runner-up Candidates")
            for idx, (res_name, score) in enumerate(top_resumes[1:4], start=1):
                st.markdown(f"**{idx}ᵗʰ Runner-up:** {res_name} (Similarity: {score:.2f})")

        completed_jobs += 1
        current_step += 1
        progress_bar.progress(
            min(current_step / max_steps, 1.0),
            text=f"Processed {completed_jobs} of {total_jobs} job(s)..."
        )
        time.sleep(0.1)  # for animation smoothness

    progress_bar.progress(1.0, text="All done! 🎉")
    st.success("All matching and explanations complete!")
    st.balloons()

else:
    st.warning("Upload at least one resume, one job description, select top N matches, and click 'Run' to get started.")

st.markdown("---")
st.caption("Powered by SentenceTransformers & Hugging Face Zephyr LLM. **Do not upload confidential documents if using public endpoints!**")
