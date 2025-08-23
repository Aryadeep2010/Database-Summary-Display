# application.py
from io import BytesIO, StringIO
import re
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

# ---- documents ----
import docx2txt
import fitz  # PyMuPDF

# ---- summarization ----
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

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

st.markdown(
    """
    <style>
      .stApp { background: #0b1222; color: #e5e7eb; }
      h1,h2,h3 { color: #93c5fd; }
      .accent { color:#93c5fd; }
      .muted { color:#94a3b8; }
      .chip { display:inline-block; padding:4px 10px; border-radius:999px; background:rgba(148,163,184,0.15); margin-right:6px; font-size:12px;}
      .card { background:#0f172a; border:1px solid rgba(148,163,184,0.15); border-radius:14px; padding:16px; margin-bottom:12px; }
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
    """,
    unsafe_allow_html=True,
)

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
    cleaned = clean_text_for_summary(text)
    if len(cleaned.split()) < 60:
        return "⚠️ Document too short to summarize."
    try:
        return sumy_lexrank(cleaned, sentences_count=target_sentences)
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

# ---------------------------
# Sidebar: choose mode
# ---------------------------
mode = st.sidebar.radio("Choose what to analyze:", ["📂 Dataset", "📑 Document"])

# ---------------------------
# Dataset Mode
# ---------------------------
if mode == "📂 Dataset":
    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    uploaded = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded:
        st.info("Upload a dataset to get started.")
        st.stop()

    df = load_file(uploaded)
    types = infer_types(df)

    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    st.subheader("👀 Quick Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Cells", int(df.isna().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    # Data cleaning
    with st.expander("🧹 Data Cleaning"):
        drop_dups = st.checkbox("Drop duplicate rows", value=True)
        na_strategy = st.selectbox("Fill NA for numeric columns with:", ["None", "Mean", "Median"])
        df_work = df.copy()
        if drop_dups:
            df_work = df_work.drop_duplicates()
        if na_strategy != "None":
            num_cols = [c for c, t in types.items() if t == "numeric"]
            if num_cols:
                if na_strategy == "Mean":
                    df_work[num_cols] = df_work[num_cols].fillna(df_work[num_cols].mean())
                else:
                    df_work[num_cols] = df_work[num_cols].fillna(df_work[num_cols].median())
        st.success("✅ Cleaning applied to working copy.")

    # Charts
    st.subheader("📈 Explorations")
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Histogram / Boxplot", "Correlation", "Time Series"])
    with chart_tab1:
        num_cols = [c for c, t in types.items() if t == "numeric"]
        if num_cols:
            sel_col = st.selectbox("Select numeric column", num_cols)
            fig, ax = plt.subplots()
            ax.hist(df_work[sel_col].dropna(), bins=30, edgecolor="black")
            ax.set_title(f"Histogram • {sel_col}")
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            ax2.boxplot(df_work[sel_col].dropna())
            ax2.set_title(f"Boxplot • {sel_col}")
            st.pyplot(fig2)
        else:
            st.info("No numeric columns found.")

    with chart_tab2:
        num_cols = [c for c, t in types.items() if t == "numeric"]
        if len(num_cols) >= 2:
            corr = df_work[num_cols].corr(numeric_only=True)
            fig, ax = plt.subplots()
            cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=45)
            ax.set_yticks(range(len(num_cols)))
            ax.set_yticklabels(num_cols)
            fig.colorbar(cax)
            st.pyplot(fig)
        else:
            st.info("Need at least two numeric columns for correlation heatmap.")

    with chart_tab3:
        datetime_cols = [c for c, t in types.items() if t == "datetime"]
        num_cols = [c for c, t in types.items() if t == "numeric"]
        if datetime_cols and num_cols:
            dt_col = st.selectbox("Datetime column", datetime_cols)
            y_col = st.selectbox("Numeric column to plot", num_cols)
            tmp = df_work[[dt_col, y_col]].dropna().sort_values(dt_col).groupby(dt_col)[y_col].mean()
            st.line_chart(tmp)
        else:
            st.info("Need at least one datetime and numeric column for time series.")

    # Anomaly detection
    st.subheader("🚨 Anomaly Detection")
    num_cols = [c for c, t in types.items() if t == "numeric"]
    if num_cols:
        features = st.multiselect("Select numeric features", num_cols, default=num_cols[: min(3, len(num_cols))])
        if features:
            X = df_work[features].dropna()
            if len(X) > 5:
                contamination = st.slider("Anomaly Sensitivity", 0.01, 0.15, 0.03)
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                model = IsolationForest(contamination=contamination, random_state=42)
                preds = model.fit_predict(Xs)
                outliers = X[preds == -1]
                st.write(f"Flagged anomalies: **{len(outliers)}**")
                if len(outliers):
                    st.dataframe(outliers.head(), use_container_width=True)
                    st.download_button("⬇️ Download anomalies CSV",
                                       outliers.to_csv().encode("utf-8"),
                                       "anomalies.csv", "text/csv")
            else:
                st.info("Need at least a few rows to detect anomalies.")
    else:
        st.info("No numeric columns available for anomaly detection.")

    # Export cleaned dataset
    st.subheader("⬇️ Export (Working Copy)")
    st.download_button("Download cleaned dataset (CSV)",
                       df_work.to_csv(index=False).encode("utf-8"),
                       "cleaned_dataset.csv", "text/csv")
    buf = BytesIO()
    df_work.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    st.download_button("Download cleaned dataset (XLSX)",
                       buf,
                       file_name="cleaned_dataset.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------
# Document Mode
# ---------------------------
if mode == "📑 Document":
    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    st.subheader("📑 Upload a document (PDF, DOCX, TXT)")
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
            st.markdown(" ".join([f'<span class="chip">{k}</span>' for k in kws]),
                        unsafe_allow_html=True)
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
