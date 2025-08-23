# application.py
from io import BytesIO, StringIO
import os
import re
import math
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

# ---- documents ----
import docx2txt
import fitz  # PyMuPDF

# ---- summarization (offline) ----
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from gensim.summarization import summarize as gensim_summarize  # type: ignore

# ---- NLTK tokenizer fix ----
import nltk  # type: ignore
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# ---------------------------
# Streamlit Config + CSS
# ---------------------------
st.set_page_config(page_title="AI-Assisted Data & Document Insights", layout="wide")

st.markdown("""
    <style>
      .stApp { background: #0b1222; color: #e5e7eb; }
      h1,h2,h3 { color: #93c5fd; }
      .accent { color:#93c5fd; }
      .muted { color:#94a3b8; }
      .chip { display:inline-block; padding:4px 10px; border-radius:999px; background:rgba(148,163,184,0.15); margin-right:6px; font-size:12px;}
      .card { background:#0f172a; border:1px solid rgba(148,163,184,0.15); border-radius:14px; padding:16px; }
      .soft { box-shadow:0 6px 24px rgba(0,0,0,0.25); }
      .stButton>button {
          background:#3b82f6; color:white; border:0; padding:0.6rem 1rem;
          border-radius:10px; transition:0.15s;
      }
      .stButton>button:hover { transform: translateY(-1px); filter:brightness(1.05); }
      .metric-card { background:#0f172a; border-radius:14px; padding:14px; border:1px solid rgba(148,163,184,0.12); }
      .section-title { margin-top:0; }
      textarea, .stTextInput>div>div>input { color:#e5e7eb !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📊 AI-Assisted Data & Document Insights</h1>", unsafe_allow_html=True)
st.caption("Smarter summaries • Actionable insights • Clean visuals")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def infer_types(df: pd.DataFrame):
    types = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types

def clean_text_for_summary(text: str) -> str:
    """
    Clean document text while preserving numbering, bullets, and paragraphs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if re.match(r"^(\d+[\.\)]|[ivxlcdmIVXLCDM]+[\.\)]|\-|\*)\s*", line):
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(re.sub(r"[ \t]{2,}", " ", line))
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{2,}", "\n\n", cleaned_text)
    return cleaned_text.strip()

def sumy_lexrank(text: str, sentences_count: int = 8) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return "\n".join([str(s) for s in summary])

def summarize_document(text: str, target_sentences: int = 8) -> str:
    cleaned = clean_text_for_summary(text)
    try:
        words = len(cleaned.split())
        if words < 60:
            return "⚠️ Document too short to summarize."
        ratio = min(0.25, max(0.03, target_sentences / max(120, words/2)))
        gsum = gensim_summarize(cleaned, ratio=ratio)
        if gsum and len(gsum.split()) > 20:
            gsum = re.sub(r"(?<=[.!?])\s+", "\n", gsum)
            gsum = re.sub(r"(\n\d+[\.\)]\s+)", r"\n\1", gsum)
            gsum = re.sub(r"(\n[\-\*]\s+)", r"\n\1", gsum)
            return gsum.strip()
    except Exception:
        pass
    try:
        summary = sumy_lexrank(cleaned, sentences_count=target_sentences)
        summary = re.sub(r"(?<=[.!?])\s+", "\n", summary)
        summary = re.sub(r"(\n\d+[\.\)]\s+)", r"\n\1", summary)
        summary = re.sub(r"(\n[\-\*]\s+)", r"\n\1", summary)
        return summary.strip()
    except Exception as e:
        return f"⚠️ Could not summarize: {e}"

def extract_keywords(text: str, top_k: int = 10):
    text = text.lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text)
    stop = set("""the of to and a in for on with is that as at by from this be are or it an
        we you they he she i was were has have had not but can will would could should
        into over under after before between about than more most other such any each
        which who whom whose where when why how also there their its our your
        however therefore meanwhile moreover nevertheless per within without using
        based including across among through against new study results data""".split())
    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [w for w, _ in items]

# Dataset helpers, anomaly detection, visualization remain exactly as in your previous version

# ---------------------------
# Sidebar: choose mode
# ---------------------------
mode = st.sidebar.radio("Choose what to analyze:", ["📂 Dataset", "📑 Document"])

# ---------------------------
# MODE 1: Dataset
# ---------------------------
if mode == "📂 Dataset":
    # Keep all dataset logic, cleaning, visualization, anomaly detection
    # Exactly as in your current code
    pass  # placeholder for your full dataset mode code

# ---------------------------
# MODE 2: Document
# ---------------------------
if mode == "📑 Document":
    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    st.subheader("📑 Upload a document (PDF, DOCX, or TXT)")
    doc_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    st.markdown("</div>", unsafe_allow_html=True)

    if not doc_file:
        st.info("Upload a document to get started.")
        st.stop()

    # Extract text
    text = ""
    if doc_file.name.lower().endswith(".pdf"):
        with fitz.open(stream=doc_file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        doc_file.seek(0)
    elif doc_file.name.lower().endswith(".docx"):
        text = docx2txt.process(doc_file)
    else:
        stringio = StringIO(doc_file.getvalue().decode("utf-8", errors="ignore"))
        text = stringio.read()

    st.subheader("📝 Content Preview")
    preview = text[:4000] + ("..." if len(text) > 4000 else "")
    st.text_area("Extracted text:", preview, height=220)

    st.subheader("🧠 Summarize & Insights")
    c1, c2 = st.columns([2,1])
    with c1:
        sent_target = st.slider("Target summary length (sentences)", 4, 16, 8)
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_document(text, target_sentences=sent_target)
            st.markdown("#### ✨ Summary")
            st.text_area("Summary", summary, height=220)
    with c2:
        st.markdown("**Keyword Hints**")
        kws = extract_keywords(text, top_k=12)
        if kws:
            st.markdown(" ".join([f'<span class="chip">{k}</span>' for k in kws]), unsafe_allow_html=True)
        else:
            st.caption("No strong keywords detected.")

    st.markdown("#### 💡 Actionable Notes")
    notes = []
    lowered = text.lower()
    if any(tag in lowered for tag in ["abstract", "introduction", "summary"]):
        notes.append("Document appears academic/structured — consider skimming Abstract/Conclusion first.")
    if "conclusion" in lowered or "results" in lowered or "findings" in lowered:
        notes.append("Key takeaways likely live under **Conclusion/Results** sections.")
    if len(text.split()) > 3000:
        notes.append("This is a long document — try a shorter summary first, then refine with more sentences.")
    if not notes:
        notes.append("Use Keyword Hints to jump to relevant sections or search within the document (Ctrl/Cmd + F).")
    st.markdown("\n".join([f"- {n}" for n in notes]))
