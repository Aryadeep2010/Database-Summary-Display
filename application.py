# application.py
import os
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore

# ---- For documents ----
from io import StringIO
import docx2txt
import fitz  # PyMuPDF

# ---- Summarizer (Sumy) ----
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


# ==============================
# Helper: Document Summarization
# ==============================
def summarize_text(text, sentence_count=6):
    if not text.strip():
        return "⚠️ No text found to summarize."

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    summary_sentences = [str(sentence) for sentence in summarizer(parser.document, sentence_count)]

    if not summary_sentences:
        return "⚠️ Could not generate a useful summary."

    # Format nicely with bullet points
    summary = "\n\n".join(f"• {sentence}" for sentence in summary_sentences)
    return summary


# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="📊 Smart Analyzer", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #FF5733;'>📊 Smart File Analyzer</h1>
    <p style='text-align: center; font-size:18px;'>
    Upload a <b>Document</b> (PDF/DOCX/TXT) or a <b>Spreadsheet</b> (CSV/XLSX) 
    and get instant summaries, insights, and analysis.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---- Mode Selector ----
mode = st.radio("🔍 Choose Mode", ["📑 Document", "📊 Spreadsheet"])

# ---- File Upload ----
uploaded_file = st.file_uploader("📂 Upload your file", type=["pdf", "docx", "txt", "csv", "xlsx"])


if uploaded_file is not None:

    # ==============================
    # Handle Document Mode
    # ==============================
    if mode == "📑 Document":
        st.markdown("### 📑 Document Analysis")

        file_type = uploaded_file.name.split(".")[-1].lower()
        text = ""

        if file_type == "pdf":
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf_doc:
                text += page.get_text("text")

        elif file_type == "docx":
            text = docx2txt.process(uploaded_file)

        elif file_type == "txt":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()

        if text.strip():
            summary = summarize_text(text, sentence_count=7)
            st.markdown("#### 📌 Clean Summary")
            st.success(summary)
        else:
            st.warning("⚠️ Could not extract any text from this document.")

    # ==============================
    # Handle Spreadsheet Mode
    # ==============================
    elif mode == "📊 Spreadsheet":
        st.markdown("### 📊 Spreadsheet Analysis")

        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Show preview
        st.markdown("#### 🗂️ Data Preview")
        st.dataframe(df.head())

        # ===================
        # Summary & Insights
        # ===================
        st.markdown("#### 📌 Key Insights")
        insights = []
        insights.append(f"• Dataset Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
        insights.append(f"• Columns: {', '.join(df.columns)}")

        # Missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            insights.append(f"• Missing Values: **{missing}** found in dataset")

        # Numeric columns insights
        if not df.select_dtypes(include="number").empty:
            num_stats = df.describe().transpose()
            insights.append("• Basic Statistics (Numeric Columns):")
            st.write(num_stats)

        st.info("\n\n".join(insights))

        # ===================
        # Outlier Detection
        # ===================
        st.markdown("#### 🚨 Outlier Detection")
        if not df.select_dtypes(include="number").empty:
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(df.select_dtypes(include="number").fillna(0))
            outliers = df[preds == -1]
            st.write(f"Detected **{len(outliers)}** potential outliers.")
            st.dataframe(outliers.head())
        else:
            st.warning("⚠️ No numeric columns found for anomaly detection.")

        # ===================
        # Visualization
        # ===================
        st.markdown("#### 📈 Quick Visualization")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col_to_plot = st.selectbox("Choose a column to visualize:", num_cols)
            plt.figure(figsize=(6, 4))
            df[col_to_plot].hist(bins=20, color="#FF5733", edgecolor="black")
            plt.title(f"Distribution of {col_to_plot}", fontsize=14, color="#333")
            plt.xlabel(col_to_plot)
            plt.ylabel("Frequency")
            st.pyplot(plt)
        else:
            st.info("No numeric columns to visualize.")
