"""Resume-to-job matcher.

Ranks resumes against job descriptions with a hybrid score:
- Semantic similarity (chunked sentence embeddings, so long documents aren't truncated)
- Skill coverage (which skills/keywords from the job appear in the resume)

Optionally generates a recruiter-style summary with a Hugging Face chat model.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --------- Configuration ---------
EMB_MODEL_ID = "all-MiniLM-L6-v2"
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_LLM = "meta-llama/Llama-3.1-8B-Instruct"
CHUNK_WORDS = 150  # MiniLM truncates ~256 tokens, so embed in chunks
CHUNK_OVERLAP = 30

# Use Streamlit secrets for your Hugging Face token!
HF_TOKEN = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else os.environ.get("HF_TOKEN", "")
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"  # You can swap to another if rate limited


def get_hf_token() -> str:
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:  # no secrets.toml present
        pass
    return os.environ.get("HF_TOKEN", "")


# Curated skill vocabulary. Multi-word entries are matched as phrases.
SKILL_VOCAB = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang", "rust", "scala",
    "kotlin", "swift", "ruby", "php", "r", "sql", "bash", "matlab", "perl",
    # Data / ML
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost", "spark",
    "pyspark", "hadoop", "airflow", "dbt", "snowflake", "databricks", "bigquery", "redshift",
    "tableau", "power bi", "looker", "excel", "statistics", "data analysis", "data engineering",
    "data visualization", "etl", "llm", "generative ai", "mlops", "a/b testing", "forecasting",
    # Web / backend
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring", "graphql",
    "rest", "api", "microservices", "html", "css",
    # Cloud / devops
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "jenkins", "git",
    "linux", "devops", "ansible",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "oracle", "nosql", "kafka",
    # Practices / soft skills
    "agile", "scrum", "project management", "product management", "leadership", "communication",
    "stakeholder management", "mentoring", "jira", "testing", "unit testing", "security",
    # Business
    "salesforce", "sap", "crm", "marketing", "seo", "finance", "accounting", "budgeting",
}

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers him his how i if in into is it its itself just me more most my
    no nor not now of off on once only or other our ours out over own same she should so some such
    than that the their them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with you your yours role job work
    working team teams experience years year strong ability skills skill knowledge using use within
    including candidate candidates position company responsibilities requirements required preferred
    plus must new join looking well good excellent etc across help build make ensure support""".split()
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)


# --------- Text extraction ---------
@st.cache_data(show_spinner=False, max_entries=500)
def extract_text(name: str, data: bytes) -> str:
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if lower.endswith(".docx"):
            doc = Document(io.BytesIO(data))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:  # many resumes put skills in tables
                for row in table.rows:
                    parts.append(" | ".join(c.text for c in row.cells))
            return "\n".join(parts)
        if lower.endswith(".txt"):
            return data.decode("utf-8", errors="ignore")
    except Exception as e:  # corrupt or encrypted files
        return f"__ERROR__ {e}"
    return ""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, max(len(words) - CHUNK_OVERLAP, 1), step)]


# --------- Skills / keywords ---------
def contains_term(norm_text: str, term: str) -> bool:
    pattern = r"(?<![\w+#])" + re.escape(term) + r"(?![\w+#])"
    return re.search(pattern, norm_text) is not None


def extract_job_terms(job_text: str, extra_keywords: int) -> tuple[list[str], list[str]]:
    """Return (known skills found in the job, top additional keywords)."""
    norm = normalize(job_text)
    skills = sorted(s for s in SKILL_VOCAB if contains_term(norm, s))
    tokens = re.findall(r"[a-z][a-z+#./-]{2,}", norm)
    counts = Counter(t.strip(".-/") for t in tokens if t not in STOPWORDS and t not in SKILL_VOCAB)
    keywords = [w for w, c in counts.most_common(extra_keywords * 2) if c >= 2 and w not in STOPWORDS][:extra_keywords]
    return skills, keywords


def candidate_profile(text: str) -> dict:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    years = [int(y) for y in YEARS_RE.findall(text) if int(y) < 50]
    return {
        "email": email.group(0) if email else "",
        "phone": phone.group(0).strip() if phone else "",
        "years": max(years) if years else None,
        "words": len(text.split()),
    }


# --------- Embeddings ---------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL_ID)


@st.cache_data(show_spinner=False, max_entries=500)
def embed_chunks(text_hash: str, text: str) -> np.ndarray:
    del text_hash  # cache key only
    vecs = load_embedder().encode(chunk_text(text), normalize_embeddings=True)
    return np.asarray(vecs)


def semantic_score(job_vecs: np.ndarray, res_vecs: np.ndarray) -> float:
    """For each job chunk, take its best-matching resume chunk; average those."""
    sims = job_vecs @ res_vecs.T
    return float(np.clip(sims.max(axis=1).mean(), 0, 1))


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# --------- LLM summary (optional) ---------
@st.cache_data(show_spinner=False, max_entries=200, ttl=3600)
def llm_summary(resume_text: str, job_text: str, matched: tuple, missing: tuple, model: str, token: str) -> str:
    prompt = (
        "You are an experienced technical recruiter. Assess the candidate against the job.\n"
        "Respond in Markdown with these sections:\n"
        "**Fit verdict:** one of Strong / Moderate / Weak, plus one sentence.\n"
        "**Key strengths:** 3-5 bullets citing concrete evidence from the resume.\n"
        "**Gaps & risks:** 2-4 bullets.\n"
        "**Interview questions:** 3 targeted questions to probe the gaps.\n"
        "Only use evidence from the resume. Be concise.\n\n"
        f"Skills detected as matching: {', '.join(matched) or 'none'}\n"
        f"Skills detected as missing: {', '.join(missing) or 'none'}\n\n"
        f"JOB DESCRIPTION:\n{job_text[:4000]}\n\nRESUME:\n{resume_text[:6000]}"
    )
    resp = requests.post(
        HF_CHAT_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 700,
            "temperature": 0.3,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Hugging Face returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def rule_based_summary(row: dict) -> str:
    cov = row["skill_coverage"]
    verdict = "Strong" if row["score"] >= 0.65 else "Moderate" if row["score"] >= 0.45 else "Weak"
    lines = [f"**Fit verdict:** {verdict} (overall score {row['score']:.0%})."]
    lines.append(
        f"- Covers **{len(row['matched'])}** of {len(row['matched']) + len(row['missing'])} "
        f"job skills/keywords ({cov:.0%})."
    )
    if row["matched"]:
        lines.append(f"- Strengths: {', '.join(row['matched'][:10])}.")
    if row["missing"]:
        lines.append(f"- Not found in resume: {', '.join(row['missing'][:10])}.")
    if row["years"]:
        lines.append(f"- Mentions up to {row['years']} years of experience.")
    return "\n".join(lines)


# --------- Matching ---------
def run_matching(resumes: dict[str, str], jobs: dict[str, str], semantic_weight: float, extra_kw: int, progress) -> dict:
    results: dict[str, list[dict]] = {}
    total = len(resumes) + len(jobs)
    done = 0
    res_vecs = {}
    for name, text in resumes.items():
        res_vecs[name] = embed_chunks(sha(text), text)
        done += 1
        progress.progress(done / total, text=f"Embedding resume: {name}")
    profiles = {name: candidate_profile(text) for name, text in resumes.items()}
    norm_resumes = {name: normalize(text) for name, text in resumes.items()}

    for job_name, job_text in jobs.items():
        job_vecs = embed_chunks(sha(job_text), job_text)
        skills, keywords = extract_job_terms(job_text, extra_kw)
        terms = skills + keywords
        rows = []
        for res_name in resumes:
            sem = semantic_score(job_vecs, res_vecs[res_name])
            matched = [t for t in terms if contains_term(norm_resumes[res_name], t)]
            missing = [t for t in terms if t not in matched]
            # Known skills count double vs. generic keywords
            weight_total = 2 * len(skills) + len(keywords)
            weight_hit = sum(2 if t in skills else 1 for t in matched)
            coverage = weight_hit / weight_total if weight_total else sem
            score = semantic_weight * sem + (1 - semantic_weight) * coverage
            rows.append(
                {
                    "resume": res_name,
                    "score": score,
                    "semantic": sem,
                    "skill_coverage": coverage,
                    "matched": matched,
                    "missing": missing,
                    **profiles[res_name],
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        results[job_name] = rows
        done += 1
        progress.progress(done / total, text=f"Scoring job: {job_name}")
    return results


def collect_docs(files, pasted: str, pasted_label: str) -> tuple[dict[str, str], list[str]]:
    docs, problems = {}, []
    for f in files or []:
        text = extract_text(f.name, f.getvalue())
        if text.startswith("__ERROR__"):
            problems.append(f"{f.name}: could not be read ({text[10:]})")
        elif len(text.split()) < 30:
            problems.append(f"{f.name}: little or no text extracted (scanned image PDF?)")
            if text.strip():
                docs[f.name] = text
        else:
            docs[f.name] = text
    if pasted.strip():
        docs[pasted_label] = pasted
    return docs, problems


# --------- UI ---------
st.set_page_config(page_title="Resume matcher", page_icon=":material/work:", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.resume_texts = {}
    st.session_state.job_texts = {}

with st.sidebar:
    st.header(":material/tune: Settings")
    semantic_weight = st.slider(
        "Semantic vs. skill weight",
        0.0, 1.0, 0.6, 0.05,
        help="1.0 = only meaning-based similarity, 0.0 = only skill/keyword coverage.",
    )
    extra_kw = st.slider("Extra job keywords to track", 0, 20, 8, help="Frequent job-specific terms beyond the built-in skill list.")
    top_n = st.slider("Candidates shown per job", 1, 20, 5)
    min_score = st.slider("Hide candidates below score", 0.0, 1.0, 0.0, 0.05)

    st.divider()
    st.subheader(":material/smart_toy: AI summaries (optional)")
    hf_token = get_hf_token()
    if not hf_token:
        hf_token = st.text_input("Hugging Face token", type="password", help="Or set HF_TOKEN in secrets / env.")
    llm_model = st.text_input("Chat model", DEFAULT_LLM)
    st.caption("Resume text is sent to Hugging Face only when you click *Generate AI summary*.")

st.title(":material/work: Resume-to-job matcher")
st.caption("Rank candidates for each role by semantic fit and skill coverage, see exactly which skills are missing, and export a shortlist.")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.subheader(":material/description: Resumes")
        resume_files = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
with col2:
    with st.container(border=True):
        st.subheader(":material/assignment: Job descriptions")
        job_files = st.file_uploader("Upload job descriptions", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        pasted_job = st.text_area("Or paste a job description", height=120, placeholder="Paste the job posting here...")

run = st.button("Match candidates", type="primary", icon=":material/play_arrow:")

if run:
    resume_texts, r_problems = collect_docs(resume_files, "", "")
    job_texts, j_problems = collect_docs(job_files, pasted_job, "Pasted job description")
    for p in r_problems + j_problems:
        st.warning(p)
    if not resume_texts or not job_texts:
        st.error("Add at least one readable resume and one job description.")
    else:
        progress = st.progress(0.0, text="Starting...")
        st.session_state.results = run_matching(resume_texts, job_texts, semantic_weight, extra_kw, progress)
        st.session_state.resume_texts = resume_texts
        st.session_state.job_texts = job_texts
        progress.empty()

results = st.session_state.results
if not results:
    st.info("Upload resumes and at least one job description, then click **Match candidates**.")
    st.stop()

resume_texts = st.session_state.resume_texts
job_texts = st.session_state.job_texts

# ---- Overview ----
st.header("Overview")
m1, m2, m3 = st.columns(3)
m1.metric("Resumes", len(resume_texts))
m2.metric("Jobs", len(job_texts))
best = max((rows[0]["score"] for rows in results.values() if rows), default=0)
m3.metric("Best score", f"{best:.0%}")

matrix = pd.DataFrame(
    {job: {r["resume"]: r["score"] for r in rows} for job, rows in results.items()}
)
matrix["Best job"] = matrix.idxmax(axis=1)
st.subheader("Score matrix (resume × job)")
st.dataframe(
    matrix,
    column_config={
        c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="percent") for c in results
    },
)

export_rows = [
    {
        "job": job,
        "rank": i + 1,
        "resume": r["resume"],
        "score": round(r["score"], 3),
        "semantic": round(r["semantic"], 3),
        "skill_coverage": round(r["skill_coverage"], 3),
        "matched": ", ".join(r["matched"]),
        "missing": ", ".join(r["missing"]),
        "email": r["email"],
        "phone": r["phone"],
        "years_mentioned": r["years"],
    }
    for job, rows in results.items()
    for i, r in enumerate(rows)
]
st.download_button(
    "Download full results (CSV)",
    pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
    "resume_matches.csv",
    "text/csv",
    icon=":material/download:",
)

# ---- Per-job detail ----
st.header("Candidates by job")
tabs = st.tabs(list(results))
for tab, (job_name, rows) in zip(tabs, results.items()):
    with tab:
        shown = [r for r in rows if r["score"] >= min_score][:top_n]
        all_terms = (rows[0]["matched"] + rows[0]["missing"]) if rows else []
        with st.expander("Skills & keywords tracked for this job"):
            st.write(", ".join(all_terms) or "No recognizable skills found; ranking uses semantic similarity only.")
            st.text(job_texts[job_name][:3000])

        table = pd.DataFrame(
            [
                {
                    "Rank": i + 1,
                    "Resume": r["resume"],
                    "Score": r["score"],
                    "Semantic": r["semantic"],
                    "Skill coverage": r["skill_coverage"],
                    "Missing": ", ".join(r["missing"][:8]),
                    "Email": r["email"],
                }
                for i, r in enumerate(shown)
            ]
        )
        if table.empty:
            st.info("No candidates above the score threshold.")
            continue
        pct = st.column_config.ProgressColumn
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Score": pct("Score", min_value=0, max_value=1, format="percent"),
                "Semantic": pct("Semantic", min_value=0, max_value=1, format="percent"),
                "Skill coverage": pct("Skill coverage", min_value=0, max_value=1, format="percent"),
            },
        )

        for i, r in enumerate(shown):
            with st.expander(f"#{i + 1} {r['resume']} — {r['score']:.0%}", expanded=(i == 0)):
                c1, c2, c3 = st.columns(3)
                c1.metric("Semantic fit", f"{r['semantic']:.0%}")
                c2.metric("Skill coverage", f"{r['skill_coverage']:.0%}")
                c3.metric("Years mentioned", r["years"] if r["years"] else "n/a")
                if r["email"] or r["phone"]:
                    st.caption(f":material/contact_mail: {r['email']}  {r['phone']}")
                st.markdown(
                    "**Matched:** " + (" ".join(f":green-badge[{t}]" for t in r["matched"]) or "_none_")
                )
                st.markdown(
                    "**Missing:** " + (" ".join(f":red-badge[{t}]" for t in r["missing"]) or "_none_")
                )
                st.markdown(rule_based_summary(r))

                key = f"llm_{job_name}_{r['resume']}"
                if st.button("Generate AI summary", key=key, icon=":material/auto_awesome:", disabled=not hf_token):
                    with st.spinner("Asking the model..."):
                        try:
                            st.session_state[key + "_out"] = llm_summary(
                                resume_texts[r["resume"]], job_texts[job_name],
                                tuple(r["matched"]), tuple(r["missing"]), llm_model, hf_token,
                            )
                        except Exception as e:
                            st.session_state[key + "_out"] = f":red[AI summary failed: {e}]"
                if key + "_out" in st.session_state:
                    with st.container(border=True):
                        st.markdown(st.session_state[key + "_out"])

                with st.popover("View resume text", icon=":material/visibility:"):
                    st.text(resume_texts[r["resume"]][:5000])

st.divider()
st.caption("Embeddings run locally. AI summaries use the Hugging Face API; avoid sending confidential resumes to public endpoints.")
