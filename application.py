# application.py
from io import BytesIO  # add this at the top with imports
import os
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore

# ---- for documents ----
from io import StringIO
import docx2txt
import fitz  # PyMuPDF

# Offline summarization using sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# ---------------------------
# Streamlit Config + CSS
# ---------------------------
st.set_page_config(page_title="AI-Assisted Data Insights", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #0f172a; color: #e2e8f0; }
      h1,h2,h3 { color: #60a5fa; }
      .stButton>button {
          background:#3b82f6; color:white; border:0; padding:0.6rem 1rem;
          border-radius:10px; transition:0.2s;
      }
      .stButton>button:hover { transform: translateY(-1px); }
      .card {
          background:#111827; border-radius:14px; padding:16px; margin:8px 0;
          box-shadow:0 6px 18px rgba(0,0,0,0.25);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>📊 AI-Assisted Data Insights Dashboard</h1>", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_file(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

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

def summarize_text(text, sentences_count=6):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return "\n".join([str(sentence) for sentence in summary])

# ---------------------------
# Sidebar: choose mode
# ---------------------------
mode = st.sidebar.radio("Choose what to analyze:", ["📂 Dataset", "📑 Document"])

# ===========================
# MODE 1: Dataset
# ===========================
if mode == "📂 Dataset":
    uploaded = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])
    if not uploaded:
        st.info("Please upload a dataset to get started.")
        st.stop()

    df = load_file(uploaded)

    st.subheader("👀 Data Preview")
    st.dataframe(df.head())

    st.subheader("📋 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", len(df))
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing cells", int(df.isna().sum().sum()))
    col4.metric("Duplicate rows", int(df.duplicated().sum()))

    types = infer_types(df)
    st.write("Detected column types:", types)

    with st.expander("🧹 Data Cleaning Options"):
        drop_dups = st.checkbox("Drop duplicate rows", value=True)
        na_strategy = st.selectbox("Fill NA for numeric columns with:", ["None", "Mean", "Median"])
        if drop_dups:
            df = df.drop_duplicates()
        if na_strategy != "None":
            num_cols = [c for c, t in types.items() if t == "numeric"]
            if num_cols:
                if na_strategy == "Mean":
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                else:
                    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    st.subheader("📈 Quick Charts")
    chart_col = st.selectbox("Select a column for visualization", df.columns)
    ctype = types[chart_col]

    if ctype == "numeric":
        fig, ax = plt.subplots()
        ax.hist(df[chart_col].dropna(), bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram • {chart_col}")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.boxplot(df[chart_col].dropna(), vert=True)
        ax2.set_title(f"Boxplot • {chart_col}")
        st.pyplot(fig2)

    elif ctype == "categorical":
        vc = df[chart_col].astype(str).value_counts().head(15)
        st.bar_chart(vc)

    elif ctype == "datetime":
        num_cols = [c for c, t in types.items() if t == "numeric"]
        if num_cols:
            ycol = st.selectbox("Numeric column to plot over time", num_cols)
            tmp = df[[chart_col, ycol]].dropna().sort_values(chart_col).groupby(chart_col)[ycol].mean()
            st.line_chart(tmp)

    st.subheader("🚨 Anomaly Detection")
    num_cols = [c for c, t in types.items() if t == "numeric"]
    if num_cols:
        features = st.multiselect("Select numeric features for anomaly detection", num_cols, default=num_cols[:3])
        if features:
            X = df[features].dropna()
            contamination = st.slider("Anomaly Sensitivity", 0.01, 0.1, 0.03)
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X)
            outliers = X[preds == -1]
            st.write(f"Flagged anomalies: {len(outliers)}")
            st.dataframe(outliers.head())
            st.download_button("⬇️ Download anomalies CSV",
                               outliers.to_csv().encode("utf-8"),
                               "anomalies.csv", "text/csv")
    else:
        st.info("No numeric columns available for anomaly detection.")

st.subheader("⬇️ Export")
# CSV Export
st.download_button("Download cleaned dataset (CSV)",
                   df.to_csv(index=False).encode("utf-8"),
                   "cleaned_dataset.csv", "text/csv")

# Excel Export (fixed with BytesIO)
towrite = BytesIO()
df.to_excel(towrite, index=False, engine="openpyxl")
towrite.seek(0)
st.download_button("Download cleaned dataset (XLSX)",
                   data=towrite,
                   file_name="cleaned_dataset.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ===========================
# MODE 2: Document (Offline, Safe)
# ===========================
else:
    st.subheader("📑 Upload a document (PDF, DOCX, or TXT)")
    doc_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

    if not doc_file:
        st.info("Please upload a document to get started.")
        st.stop()

    # Extract text
    text = ""
    if doc_file.name.lower().endswith(".pdf"):
        with fitz.open(stream=doc_file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        doc_file.seek(0)  # reset pointer just in case
    elif doc_file.name.lower().endswith(".docx"):
        text = docx2txt.process(doc_file)
    else:  # txt
        stringio = StringIO(doc_file.getvalue().decode("utf-8", errors="ignore"))
        text = stringio.read()

    st.subheader("📝 Document Content Preview")
    preview = text[:4000] + ("..." if len(text) > 4000 else "")
    st.text_area("Extracted text:", preview, height=220)

    st.subheader("🤖 AI-Free Summary")
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            try:
                if len(text.split('.')) < 5:
                    summary = "⚠️ Document too short to summarize."
                else:
                    summary = summarize_text(text, sentences_count=8)
            except Exception as e:
                summary = f"⚠️ Could not summarize: {e}"

        st.text_area("Summary:", summary, height=200)

