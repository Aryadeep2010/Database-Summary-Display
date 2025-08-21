# application.py
import os
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore

# ---- NEW: for documents ----
from io import StringIO
import docx2txt
import fitz  # PyMuPDF
from openai import OpenAI

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

# ---- NEW: OpenAI client (works with env var or Streamlit secrets) ----
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=api_key) if api_key else None  # Chat Completions supported indefinitely.  # noqa
# Ref: openai-python usage & chat completions. :contentReference[oaicite:2]{index=2}

def _split_text(text: str, max_chars: int = 9000):
    """Simple chunker to avoid token limits; splits by paragraph."""
    if len(text) <= max_chars:
        return [text]
    parts, cur, cur_len = [], [], 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if cur_len + len(p) + 2 > max_chars:
            parts.append("\n\n".join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        parts.append("\n\n".join(cur))
    return parts

def summarize_with_openai(text: str, model: str = "gpt-4o-mini"):
    """Map-reduce style summary for long docs."""
    if not client:
        return "⚠️ No API key found. Set OPENAI_API_KEY in environment or Streamlit Secrets."
    chunks = _split_text(text)
    partial = []
    for idx, ch in enumerate(chunks, 1):
        r = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You summarize documents into clear bullet points."},
                {"role": "user", "content": f"Summarize this part {idx}/{len(chunks)} in 6-10 bullets:\n\n{ch}"},
            ],
        )
        partial.append(r.choices[0].message.content.strip())
    if len(partial) == 1:
        return partial[0]
    combined = "\n\n".join(partial)
    final = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Create a concise, non-repetitive summary."},
            {"role": "user", "content": f"Combine these partial summaries into one final brief:\n\n{combined}"},
        ],
    )
    return final.choices[0].message.content.strip()

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
            model = IsolationForest(contamination=0.03, random_state=42)
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
    st.download_button("Download cleaned dataset (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "cleaned_dataset.csv", "text/csv")

# ===========================
# MODE 2: Document
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
    elif doc_file.name.lower().endswith(".docx"):
        text = docx2txt.process(doc_file)
    else:
        stringio = StringIO(doc_file.getvalue().decode("utf-8", errors="ignore"))
        text = stringio.read()

    st.subheader("📝 Document Content Preview")
    preview = text[:4000] + ("..." if len(text) > 4000 else "")
    st.text_area("Extracted text:", preview, height=220)

    st.subheader("🤖 AI-Generated Summary")
    if st.button("Generate Summary"):
        with st.spinner("Summarizing with GPT…"):
            summary = summarize_with_openai(text)
        st.markdown(f"<div class='card'>{summary}</div>", unsafe_allow_html=True)
