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
def summarize_text(text, sentence_count=5):
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
st.set_page_config(page_title="📊 Smart File Summarizer", layout="wide")

st.title("📊 Smart File Summarizer")
st.markdown("Upload a **document** (PDF/DOCX/TXT) or a **spreadsheet** (CSV/XLSX) to get an instant, clean summary.")


# ---- File Upload ----
uploaded_file = st.file_uploader("📂 Upload your file", type=["pdf", "docx", "txt", "csv", "xlsx"])


if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # ==============================
    # Handle Document Mode
    # ==============================
    if file_type in ["pdf", "docx", "txt"]:
        st.markdown("### 📑 Document Mode")

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
            st.markdown("#### 📌 Summary:")
            st.markdown(summary)
        else:
            st.warning("⚠️ Could not extract any text from this document.")

    # ==============================
    # Handle Spreadsheet Mode
    # ==============================
    elif file_type in ["csv", "xlsx"]:
        st.markdown("### 📊 Spreadsheet Mode")

        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.markdown("#### 🗂️ Data Preview")
        st.dataframe(df.head())

        # Basic statistics
        st.markdown("#### 📌 Key Insights:")
        buffer = []
        buffer.append(f"• Shape of dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        buffer.append(f"• Columns: {', '.join(df.columns)}")

        # Numeric columns insights
        if not df.select_dtypes(include="number").empty:
            buffer.append("• Numeric column statistics:")
            desc = df.describe().transpose()
            buffer.append(desc.to_string())

        # Outlier detection (Isolation Forest)
        if not df.select_dtypes(include="number").empty:
            st.markdown("#### 🚨 Outlier Detection")
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(df.select_dtypes(include="number").fillna(0))
            outliers = df[preds == -1]
            st.write(f"Detected {len(outliers)} outliers.")
            st.dataframe(outliers.head())

        # Display summary
        st.markdown("\n\n".join(buffer))

        # Plot numeric columns
        st.markdown("#### 📈 Quick Visualization")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col_to_plot = st.selectbox("Choose a column to visualize:", num_cols)
            plt.figure(figsize=(6, 4))
            df[col_to_plot].hist(bins=20)
            plt.title(f"Distribution of {col_to_plot}")
            st.pyplot(plt)
        else:
            st.info("No numeric columns to visualize.")

