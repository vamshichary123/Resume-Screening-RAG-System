import os
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="Resume Screening using RAG",
    page_icon="📄",
    layout="wide"
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to your .env file.")
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_FOLDER = os.path.join(BASE_DIR, "resumes")
VECTOR_DB_FOLDER = os.path.join(BASE_DIR, "ResumeRAG")

st.title("📄 ResumeIQ RAG")
st.caption(
    "Retrieve resume information, compare candidates against a job description, "
    "and review ATS insights in one dashboard."
)
st.divider()

# --------------------------------------------------
# LOAD RESUMES AND BUILD THE SAME RETRIEVER PIPELINE
# FROM THE UPDATED NOTEBOOK
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading resumes and preparing the RAG system...")
def load_rag_components(resume_folder, vector_db_folder, api_key):
    if not os.path.isdir(resume_folder):
        raise FileNotFoundError(
            f"Resume folder not found: {resume_folder}. Create a 'resumes' folder "
            "beside app.py and place the PDF resumes inside it."
        )

    loader = DirectoryLoader(
        path=resume_folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=900,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(docs)

    # Rebuild the per-resume text used by the notebook's ATS function.
    resume_texts = {}
    for doc in chunks:
        source = doc.metadata.get("source", "Unknown resume")
        resume_name = os.path.basename(source)
        resume_texts[resume_name] = (
            resume_texts.get(resume_name, "") + " " + doc.page_content
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key,
        dimensions=1536
    )

    if not os.path.isdir(vector_db_folder):
        raise FileNotFoundError(
            f"Vector database folder not found: {vector_db_folder}. "
            "Copy the existing ResumeRAG folder beside app.py."
        )

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=vector_db_folder
    )

    retriever1 = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )
    retriever2 = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    retriever3 = BM25Retriever.from_documents(chunks)

    final_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2, retriever3],
        weights=[0.4, 0.3, 0.3]
    )

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=api_key,
        temperature=0
    )

    return resume_texts, final_retriever, model, len(docs), len(chunks)


try:
    resumes, final_retriever, model, document_count, chunk_count = load_rag_components(
        RESUME_FOLDER, VECTOR_DB_FOLDER, OPENAI_API_KEY
    )
except Exception as error:
    st.error(f"Could not initialize the resume screening system: {error}")
    st.stop()


# --------------------------------------------------
# RAG CHATBOT — SAME GROUNDED RESPONSE RULES AS NOTEBOOK
# --------------------------------------------------
def RAG_SYSTEM(query):
    retrieved_docs = final_retriever.invoke(query)
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    system_prompt = f"""
You are a professional AI Resume Retrieval Assistant.

You are provided with resume excerpts retrieved from a vector database.
These excerpts are your ONLY source of information.

Instructions:
- Read all retrieved excerpts carefully before answering.
- Base every statement strictly on the provided resume text.
- Never fabricate or infer missing information.
- If multiple resumes satisfy the query, list every relevant candidate.
- If the answer cannot be determined from the retrieved excerpts, respond exactly:
  "I couldn't find sufficient information in the retrieved resumes."
- Mention the candidate name or resume filename whenever available.
- Summarize information rather than copying long passages.
- Use bullet points when multiple candidates are returned.
- Keep answers factual and concise.
- If confidence is low because the retrieved context is incomplete, explicitly state that.

Retrieved Resume Context
========================
{context_text}
========================

Question:
{query}

Grounded Answer:
"""
    response = model.invoke(system_prompt)
    return response.content


# --------------------------------------------------
# ATS SCORING — MATCHES THE NOTEBOOK'S CURRENT FORMULA
# --------------------------------------------------
def calculate_ats_score(resume_text, job_description):
    skills = [
        "python", "sql", "machine learning",
        "pandas", "numpy", "scikit-learn",
        "power bi", "tensorflow", "pytorch",
        "nlp", "generative ai", "rag", "langchain"
    ]

    jd_lower = job_description.lower()
    resume_lower = resume_text.lower()

    required_skills = [skill for skill in skills if skill in jd_lower]
    matched_skills = [skill for skill in required_skills if skill in resume_lower]

    score = (
        len(matched_skills) / len(required_skills) * 100
        if required_skills else 0
    )
    return round(score, 2), matched_skills


def screen_candidates(job_description):
    results = []
    for resume_name, resume_text in resumes.items():
        score, matched_skills = calculate_ats_score(
            resume_text, job_description
        )
        results.append({
            "Resume": resume_name,
            "ATS Score": score,
            "Matched Skills": ", ".join(matched_skills)
        })

    return pd.DataFrame(results).sort_values(
        by="ATS Score", ascending=False
    ).reset_index(drop=True)


def extract_job_description(uploaded_file):
    import tempfile

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        jd_docs = PyPDFLoader(temp_path).load()
        return "\n".join(doc.page_content for doc in jd_docs).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# SAME THREE-COLUMN INTERFACE
# --------------------------------------------------
left, middle, right = st.columns([1, 2, 1], gap="medium")

with left:
    st.subheader("💬 Query Chatbot")
    query = st.text_input(
        "Ask about resumes",
        placeholder="Example: What skills does this candidate have?",
        key="resume_query"
    )

    if st.button("🔍 Search Resumes", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching resumes..."):
                try:
                    answer = RAG_SYSTEM(query.strip())
                    st.session_state["rag_answer"] = answer
                    st.session_state["rag_query"] = query.strip()
                except Exception as error:
                    st.error(f"Resume search failed: {error}")

    if "rag_answer" in st.session_state:
        st.markdown("**Answer**")
        st.write(st.session_state["rag_answer"])


with middle:
    st.subheader("👥 Candidate Match Results")
    st.markdown("**📄 Upload Job Description (PDF)**")
    uploaded_jd = st.file_uploader(
        "Upload JD PDF",
        type=["pdf"],
        help="Upload a text-based PDF containing the job description.",
        key="jd_pdf"
    )

    st.markdown("**OR**")
    manual_jd = st.text_area(
        "📝 Paste Job Description",
        placeholder=(
            "Paste the full job description here...\n\n"
            "Example: We are hiring a Data Scientist with Python, SQL, "
            "Machine Learning, Pandas, NumPy, Scikit-learn and Power BI."
        ),
        height=170,
        key="manual_jd"
    )

    if st.button("🔍 Screen Candidates", use_container_width=True):
        job_description = ""

        if uploaded_jd is not None:
            with st.spinner("Reading the job description PDF..."):
                try:
                    job_description = extract_job_description(uploaded_jd)
                except Exception as error:
                    st.error(f"Could not read the JD PDF: {error}")

            if manual_jd.strip():
                st.info("Both options were provided. The uploaded PDF will be used.")
        elif manual_jd.strip():
            job_description = manual_jd.strip()

        if not job_description.strip():
            st.warning("Please upload a JD PDF or paste a job description.")
        else:
            with st.spinner("Calculating ATS scores for candidates..."):
                candidate_results = screen_candidates(job_description)

            st.session_state["candidate_results"] = candidate_results
            st.session_state["active_job_description"] = job_description
            st.success("Candidate screening completed.")

    if "active_job_description" in st.session_state:
        with st.expander("📋 View Job Description"):
            st.write(st.session_state["active_job_description"])

    if "candidate_results" in st.session_state:
        candidate_results = st.session_state["candidate_results"]
        st.dataframe(
            candidate_results,
            use_container_width=True,
            hide_index=True
        )

        csv_data = candidate_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download ATS Results (CSV)",
            data=csv_data,
            file_name="ats_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Upload a JD PDF or paste a job description to screen candidates.")


with right:
    st.header("📊 ATS Insights")

    if "candidate_results" in st.session_state:
        candidate_results = st.session_state["candidate_results"]
        total_resumes = len(candidate_results)
        average_score = (
            candidate_results["ATS Score"].mean()
            if total_resumes else 0
        )
        shortlisted = int((candidate_results["ATS Score"] >= 60).sum())
        qualification_fit = (
            shortlisted / total_resumes * 100 if total_resumes else 0
        )

        st.metric("🎯 Average ATS Score", f"{average_score:.1f}%")
        st.divider()

        st.subheader("📈 Skill Match")
        skills_to_show = [
            "Python", "SQL", "Machine Learning", "Pandas",
            "NumPy", "Scikit-learn", "Power BI"
        ]

        for skill in skills_to_show:
            matched_count = sum(
                skill.lower() in str(matched).lower()
                for matched in candidate_results["Matched Skills"]
            )
            percentage = matched_count / total_resumes * 100 if total_resumes else 0
            st.write(f"**{skill} — {percentage:.0f}%**")
            st.progress(int(percentage))

        st.divider()
        st.subheader("🏆 Top Candidate")
        if total_resumes:
            top_candidate = candidate_results.iloc[0]
            st.info(
                f"**{top_candidate['Resume']}**\n\n"
                f"ATS Score: **{top_candidate['ATS Score']:.1f}%**"
            )

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("🎯 Qualification Fit", f"{qualification_fit:.0f}%")
        with metric_col2:
            st.metric("✅ Shortlisted", shortlisted)

        st.metric("📄 Resumes Processed", total_resumes)

        with st.expander("🏆 Top 5 Candidates"):
            st.dataframe(
                candidate_results.head(5),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.metric("🎯 Average ATS Score", "0%")
        st.divider()
        st.subheader("📈 Skill Match")
        st.info("Screen candidates to view skill matching.")
        st.divider()
        st.subheader("🏆 Top Candidate")
        st.info("No candidates screened yet.")
        st.metric("📄 Resumes Processed", len(resumes))
        st.metric("✅ Shortlisted", 0)

st.divider()
st.caption(
    f"RAG corpus: {document_count} resume documents · {chunk_count} text chunks. "
    "ATS score is a rule-based skill-overlap estimate, not a guarantee of employer ATS ranking."
)
