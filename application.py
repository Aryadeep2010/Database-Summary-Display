import streamlit as st
import pandas as pd
import PyPDF2
import docx
import re
from collections import Counter

# -------------------------
# Simple Text Summarizer
# -------------------------
def summarize_text(text, num_sentences=5):
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Tokenize words
    words = re.findall(r'\w+', text.lower())

    # Word frequency
    word_freq = Counter(words)

    # Score sentences
    sentence_scores = {}
    for sent in sentences:
        for word in re.findall(r'\w+', sent.lower()):
            if word in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

    # Pick top sentences
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = " ".join(ranked_sentences[:num_sentences])
    return summary if summary else "⚠️ Could not generate summary."


# -------------------------
# File Handlers
# -------------------------
def handle_csv_xlsx(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def handle_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def handle_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error reading Word file: {e}")
        return ""


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Document & Data Analyzer", page_icon="📊", layout="wide")
st.title("📊 Document & Data Analyzer")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(("csv", "xlsx")):
        df = handle_csv_xlsx(uploaded_file)
        if df is not None:
            st.subheader("📄 Data Preview")
            st.dataframe(df.head())

            if st.button("Generate Summary"):
                summary = summarize_text(df.to_string())
                st.subheader("📝 AI-Generated Analysis")
                st.write(summary)

    elif uploaded_file.name.endswith("pdf"):
        text = handle_pdf(uploaded_file)
        if text:
            st.subheader("📄 Extracted PDF Text")
            st.text_area("Content", text, height=200)

            if st.button("Generate Summary"):
                summary = summarize_text(text)
                st.subheader("📝 AI-Generated Analysis")
                st.write(summary)

    elif uploaded_file.name.endswith("docx"):
        text = handle_docx(uploaded_file)
        if text:
            st.subheader("📄 Extracted Word Text")
            st.text_area("Content", text, height=200)

            if st.button("Generate Summary"):
                summary = summarize_text(text)
                st.subheader("📝 AI-Generated Analysis")
                st.write(summary)
