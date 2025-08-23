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
    .card { background:#0f172a; border:1px solid rgba(148,163,184,0.15); border-radius:14px; padding:16px; margin:8px 0; }
    .soft { box-shadow:0 6px 24px rgba(0,0,0,0.25); }
    .stButton>button { background:#3b82f6; color:white; border:0; padding:0.6rem 1rem; border-radius:10px; transition:0.15s; }
    .stButton>button:hover { transform: translateY(-1px); filter:brightness(1.05); }
    .metric-card { background:#0f172a; border-radius:14px; padding:14px; border:1px solid rgba(148,163,184,0.12); text-align:center; }
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
    """Load CSV or Excel file with caching."""
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine="openpyxl")

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
    """Remove unwanted characters, page numbers, hyphenation."""
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"-\s*\n", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    return text.strip()

def sumy_lexrank(text: str, sentences_count: int = 8) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return "\n".join([str(s) for s in summary])

def summarize_document(text: str, target_sentences: int = 8) -> str:
    """Prefer gensim TextRank then fallback to Sumy LexRank."""
    cleaned = clean_text_for_summary(text)
    try:
        words = len(cleaned.split())
        if words < 60:
            return "⚠️ Document too short to summarize."
        ratio = min(0.25, max(0.03, target_sentences / max(120, words/2)))
        gsum = gensim_summarize(cleaned, ratio=ratio)
        if gsum and len(gsum.split()) > 20:
            return gsum
    except Exception:
        pass
    try:
        return sumy_lexrank(cleaned, sentences_count=target_sentences)
    except Exception as e:
        return f"⚠️ Could not summarize: {e}"

def extract_keywords(text: str, top_k: int = 10):
    text = text.lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text)
    stop = set("""
        the of to and a in for on with is that as at by from this be are or it an
        we you they he she i was were has have had not but can will would could should
        into over under after before between about than more most other such any each
        which who whom whose where when why how also there their its our your
        however therefore meanwhile moreover nevertheless per within without using
        based including across among through against new study results data
    """.split())
    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [w for w, _ in items]

# ---------------------------
# Sidebar: choose mode
# ---------------------------
mode = st.sidebar.radio("Choose what to analyze:", ["📂 Dataset", "📑 Document"])

# ===========================
# MODE 1: Dataset
# ===========================
if mode == "📂 Dataset":
    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded_file:
        st.info("Upload a dataset to get started.")
        st.stop()

    df = load_file(uploaded_file)
    types = infer_types(df)

    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    st.subheader("👀 Quick Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card"><h3>Rows</h3><h2>{df.shape[0]:,}</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3>Columns</h3><h2>{df.shape[1]}</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><h3>Missing Cells</h3><h2>{int(df.isna().sum().sum()):,}</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><h3>Duplicate Rows</h3><h2>{int(df.duplicated().sum()):,}</h2></div>', unsafe_allow_html=True)

    # Data cleaning
    with st.expander("🧹 Data Cleaning"):
        drop_dups = st.checkbox("Drop duplicate rows", value=True)
        na_strategy = st.selectbox("Fill NA for numeric columns with:", ["None", "Mean", "Median"])
        df_work = df.copy()
        if drop_dups:
            df_work = df_work.drop_duplicates()
        if na_strategy != "None":
            num_cols = [c for c, t in types.items() if t == "numeric"]
            if na_cols := num_cols:
                if na_strategy == "Mean":
                    df_work[na_cols] = df_work[na_cols].fillna(df_work[na_cols].mean())
                else:
                    df_work[na_cols] = df_work[na_cols].fillna(df_work[na_cols].median())
        st.success("✅ Cleaning settings applied to working copy.")

    # Charts & plots
    st.subheader("📈 Quick Explorations")
    if numeric_cols := [c for c, t in types.items() if t == "numeric"]:
        chart_col = st.selectbox("Select numeric column for histogram/boxplot", numeric_cols)
        if chart_col:
            fig, ax = plt.subplots()
            ax.hist(df_work[chart_col].dropna(), bins=30, edgecolor="black", color="#3b82f6")
            ax.set_title(f"Histogram • {chart_col}")
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            ax2.boxplot(df_work[chart_col].dropna())
            ax2.set_title(f"Boxplot • {chart_col}")
            st.pyplot(fig2)
    else:
        st.info("No numeric columns for plotting.")

    # Anomaly Detection
    st.subheader("🚨 Anomaly Detection")
    if numeric_cols:
        features = st.multiselect("Select numeric features", numeric_cols, default=numeric_cols[:3])
        if features and len(df_work) > 5:
            X = df_work[features].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            contamination = st.slider("Anomaly Sensitivity", 0.01, 0.15, 0.03)
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X_scaled)
            outliers = X[preds == -1]
            st.write(f"Flagged anomalies: **{len(outliers)}**")
            if len(outliers):
                st.dataframe(outliers.head(), use_container_width=True)
                st.download_button("Download anomalies CSV",
                                   outliers.to_csv(index=False).encode("utf-8"),
                                   "anomalies.csv", "text/csv")
        else:
            st.info("Not enough numeric data for anomaly detection.")
    else:
        st.info("No numeric columns for anomaly detection.")

# ===========================
# MODE 2: Document
# ===========================
if mode == "📑 Document":
    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    doc_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","txt"])
    st.markdown("</div>", unsafe_allow_html=True)

    if not doc_file:
        st.info("Upload a document to get started.")
        st.stop()

    # Extract text
    text = ""
    if doc_file.name.lower().endswith(".pdf"):
        with fitz.open(stream=doc_file.read(), filetype="pdf") as pdf:
            for page in pdf: text += page.get_text()
    elif doc_file.name.lower().endswith(".docx"):
        text = docx2txt.process(doc_file)
    else:
        stringio = StringIO(doc_file.getvalue().decode("utf-8", errors="ignore"))
        text = stringio.read()

    # Preview
    st.subheader("📝 Content Preview")
    preview = "\n".join([line.strip() for line in text.splitlines() if line.strip()][:40])
    st.text_area("Extracted text:", preview, height=250)

    # Summary
    st.subheader("🧠 Summarize & Insights")
    sent_target = st.slider("Target summary length (sentences)", 4, 16, 8)
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_document(text, target_sentences=sent_target)
            # Clean display
            summary_clean = "\n".join([s.strip() for s in summary.split("\n") if s.strip()])
            st.text_area("✨ Summary", summary_clean, height=300)

    # Keywords
    kws = extract_keywords(text, top_k=12)
    if kws:
        st.markdown(" ".join([f'<span class="chip">{k}</span>' for k in kws]), unsafe_allow_html=True)
