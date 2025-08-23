# application.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# For documents
from io import StringIO
import docx2txt
import fitz  # PyMuPDF

# For summarization (Sumy)
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Ensure nltk tokenizers are available
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)


# =============== Helper Functions ===============

def clean_text(text: str) -> str:
    """Basic cleanup to remove unwanted newlines, extra spaces, and very short lines."""
    lines = text.splitlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 3]
    return " ".join(lines)


def summarize_text(text: str, sentence_count: int = 6) -> str:
    """Summarize text using Sumy's LexRank algorithm."""
    try:
        text = clean_text(text)
        if len(text.split()) < 50:  # too short
            return "⚠️ Document too short to summarize."

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentence_count)

        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"⚠️ Could not summarize: {e}"


def detect_anomalies(df: pd.DataFrame):
    """Detect anomalies in numeric columns of a DataFrame using IsolationForest."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return None

    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(numeric_df)

    anomalies = df[preds == -1]
    return anomalies


# =============== Streamlit App ===============

st.set_page_config(page_title="Smart Analyzer", layout="wide")

st.title("📊 Smart Data & Document Analyzer")
st.markdown(
    """
    This app helps you analyze spreadsheets and summarize documents **offline & safely**.  
    Choose a mode below to get started.
    """
)

mode = st.radio("Choose a mode:", ["📈 Spreadsheet Analysis", "📑 Document Summarization"])


# ===========================
# MODE 1: Spreadsheet Analysis
# ===========================
if mode == "📈 Spreadsheet Analysis":
    st.subheader("📂 Upload a spreadsheet (CSV or XLSX)")
    file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if file is not None:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            st.subheader("🔍 Data Preview")
            st.dataframe(df.head())

            st.subheader("📊 Basic Statistics")
            st.write(df.describe(include="all"))

            # Detect anomalies
            anomalies = detect_anomalies(df)
            if anomalies is not None and not anomalies.empty:
                st.subheader("🚨 Anomaly Detection")
                st.write("Unusual rows detected using Isolation Forest:")
                st.dataframe(anomalies)
            else:
                st.info("No significant anomalies detected in numeric data.")

            # Visualization
            st.subheader("📈 Visualization")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select a numeric column to visualize:", numeric_cols)
                fig, ax = plt.subplots()
                df[col].plot(kind="hist", bins=30, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available for visualization.")

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload a CSV or Excel file to begin.")


# ===========================
# MODE 2: Document Summarization
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
    preview = text[:2000] + ("..." if len(text) > 2000 else "")
    st.text_area("Extracted text:", preview, height=220)

    # Summary controls
    st.subheader("🤖 Smart Summary")
    summary_length = st.radio(
        "Choose summary length:",
        ["Short", "Medium", "Long"],
        index=1,
        horizontal=True,
    )
    if summary_length == "Short":
        sentences = 4
    elif summary_length == "Medium":
        sentences = 8
    else:
        sentences = 12

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(text, sentence_count=sentences)
        st.text_area("Summary:", summary, height=200)
