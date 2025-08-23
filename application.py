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
st.set_page_config(page_title="AI-Assisted Data Insights", layout="wide")

st.markdown(
    """
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
    # Remove page numbers/headers/footers & excessive whitespace
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)          # lone numbers (page nos)
    text = re.sub(r"-\s*\n", "", text)                    # hyphenation across lines
    text = re.sub(r"\n{2,}", "\n\n", text)                # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)                # extra spaces
    # Keep only sensible characters
    text = re.sub(r"[^\S\r\n]+", " ", text)
    return text.strip()

def sumy_lexrank(text: str, sentences_count: int = 8) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return "\n".join([str(s) for s in summary])

def summarize_document(text: str, target_sentences: int = 8) -> str:
    """
    Prefer gensim TextRank (abstractive-like extractive) then fallback to Sumy LexRank.
    """
    cleaned = clean_text_for_summary(text)
    # Heuristic: try gensim with ratio based on length
    try:
        words = len(cleaned.split())
        if words < 60:
            return "⚠️ Document too short to summarize."
        # ratio maps roughly to number of sentences; adjust slightly
        ratio = min(0.25, max(0.03, target_sentences / max(120, words/2)))
        gsum = gensim_summarize(cleaned, ratio=ratio)
        if gsum and len(gsum.split()) > 20:
            return gsum
    except Exception:
        pass
    # Fallback to LexRank
    try:
        return sumy_lexrank(cleaned, sentences_count=target_sentences)
    except Exception as e:
        return f"⚠️ Could not summarize: {e}"

def extract_keywords(text: str, top_k: int = 10):
    # Simple frequency-based keywords (stopword-light)
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

def dataset_quick_insights(df: pd.DataFrame) -> dict:
    insights = {}
    n_rows, n_cols = df.shape
    insights["shape"] = (n_rows, n_cols)
    insights["missing_cells"] = int(df.isna().sum().sum())
    insights["duplicate_rows"] = int(df.duplicated().sum())

    # Data quality score (simple heuristic)
    total_cells = max(1, n_rows * n_cols)
    missing_rate = insights["missing_cells"] / total_cells
    dup_rate = insights["duplicate_rows"] / max(1, n_rows)
    quality = max(0, 100 - (missing_rate * 60 + dup_rate * 40) * 100)
    insights["quality_score"] = round(quality, 1)

    # Column types
    types = infer_types(df)
    insights["types"] = types

    # Numeric stats
    num_cols = [c for c, t in types.items() if t == "numeric"]
    if num_cols:
        desc = df[num_cols].describe().T
        insights["numeric_summary"] = desc

        # Correlations (top pairs by abs corr)
        corr = df[num_cols].corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_vals = []
        for i, a in enumerate(corr.columns):
            for j, b in enumerate(corr.columns):
                if i < j:
                    corr_vals.append((a, b, corr.iloc[i, j]))
        corr_vals.sort(key=lambda x: abs(x[2]), reverse=True)
        insights["top_correlations"] = corr_vals[:5]

        # Outliers (rough): z-score > 3
        z = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0))
        outlier_rows = int((z > 3).any(axis=1).sum())
        insights["potential_outliers"] = outlier_rows
    else:
        insights["numeric_summary"] = None
        insights["top_correlations"] = []
        insights["potential_outliers"] = 0

    # Categorical top levels
    cat_cols = [c for c, t in types.items() if t == "categorical"]
    top_cats = {}
    for c in cat_cols[:5]:
        vc = df[c].astype(str).value_counts(dropna=False).head(5)
        top_cats[c] = vc
    insights["top_categories"] = top_cats

    return insights

def dataset_narrative(ins: dict) -> str:
    n_rows, n_cols = ins["shape"]
    lines = []
    lines.append(f"• Your dataset has **{n_rows:,} rows** and **{n_cols} columns**.")
    lines.append(f"• Data quality score: **{ins['quality_score']} / 100** "
                 f"(missing cells: {ins['missing_cells']:,}, duplicate rows: {ins['duplicate_rows']:,}).")

    if ins["numeric_summary"] is not None:
        lines.append("• Numeric columns detected: " +
                     ", ".join([c for c, t in ins["types"].items() if t == "numeric"]) + ".")
        if ins["top_correlations"]:
            best = ins["top_correlations"][0]
            lines.append(f"• Strongest correlation: **{best[0]} ↔ {best[1]}** (r = {best[2]:.2f}).")
        if ins["potential_outliers"] > 0:
            lines.append(f"• Potential outliers flagged in ~**{ins['potential_outliers']:,}** rows (|z| > 3).")
    else:
        lines.append("• No numeric columns found; consider encoding or checking your data types.")

    if ins["top_categories"]:
        for c, vc in list(ins["top_categories"].items())[:2]:
            top_label = vc.index[0]
            top_count = int(vc.iloc[0])
            lines.append(f"• In **{c}**, most frequent value is **{top_label}** ({top_count:,} rows).")

    lines.append("• Tip: Use **Data Cleaning** to drop duplicates and fill numeric NAs before modeling.")
    return "\n".join(lines)

def plot_correlation_heatmap(df: pd.DataFrame, num_cols):
    if len(num_cols) < 2:
        st.info("Need at least two numeric columns for a correlation heatmap.")
        return
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots()
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(cax)
    st.pyplot(fig)

# ---------------------------
# Sidebar: choose mode
# ---------------------------
mode = st.sidebar.radio("Choose what to analyze:", ["📂 Dataset", "📑 Document"])

# ===========================
# MODE 1: Dataset
# ===========================
if mode == "📂 Dataset":
    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    uploaded = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])
    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded:
        st.info("Upload a dataset to get started.")
        st.stop()

    df = load_file(uploaded)

    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    st.subheader("👀 Quick Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    ins = dataset_quick_insights(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card"><h3 class="section-title">Rows</h3><h2>{ins["shape"][0]:,}</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3 class="section-title">Columns</h3><h2>{ins["shape"][1]}</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><h3 class="section-title">Missing Cells</h3><h2>{ins["missing_cells"]:,}</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><h3 class="section-title">Quality Score</h3><h2>{ins["quality_score"]}</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="card soft">', unsafe_allow_html=True)
    st.subheader("🧭 Actionable Summary")
    st.markdown(dataset_narrative(ins))
    st.markdown("</div>", unsafe_allow_html=True)

    types = ins["types"]

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
        st.success("✅ Cleaning settings applied to working copy.")

    st.subheader("📈 Explorations")
    tab_hist, tab_corr, tab_time = st.tabs(["Histogram / Boxplot", "Correlation", "Time Series"])
    with tab_hist:
        chart_col = st.selectbox("Select a numeric column", [c for c, t in types.items() if t == "numeric"])
        if chart_col:
            fig, ax = plt.subplots()
            ax.hist(df_work[chart_col].dropna(), bins=30, edgecolor="black")
            ax.set_title(f"Histogram • {chart_col}")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(df_work[chart_col].dropna(), vert=True)
            ax2.set_title(f"Boxplot • {chart_col}")
            st.pyplot(fig2)

    with tab_corr:
        num_cols = [c for c, t in types.items() if t == "numeric"]
        if num_cols:
            plot_correlation_heatmap(df_work, num_cols)
        else:
            st.info("No numeric columns available.")

    with tab_time:
        datetime_cols = [c for c, t in types.items() if t == "datetime"]
        if datetime_cols:
            tcol = st.selectbox("Datetime column", datetime_cols)
            num_cols_time = [c for c, t in types.items() if t == "numeric"]
            if num_cols_time:
                ycol = st.selectbox("Numeric column to plot", num_cols_time)
                tmp = (df_work[[tcol, ycol]]
                       .dropna()
                       .sort_values(tcol)
                       .groupby(tcol)[ycol].mean())
                st.line_chart(tmp)
            else:
                st.info("No numeric columns to plot.")
        else:
            st.info("No datetime columns found.")

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

    st.subheader("⬇️ Export (Working Copy)")
    st.download_button("Download cleaned dataset (CSV)",
                       df_work.to_csv(index=False).encode("utf-8"),
                       "cleaned_dataset.csv", "text/csv")

    towrite = BytesIO()
    df_work.to_excel(towrite, index=False, engine="openpyxl")
    towrite.seek(0)
    st.download_button("Download cleaned dataset (XLSX)",
                       data=towrite,
                       file_name="cleaned_dataset.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===========================
# MODE 2: Document (Offline, Safe)
# ===========================
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
    else:  # txt
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
