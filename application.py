import streamlit as st # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore
from io import StringIO
import docx2txt
import fitz  # PyMuPDF for PDFs

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="AI-Assisted Data Insights", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        color: #ff4b4b;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    .stMetric {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>📊 AI-Assisted Data Insights Dashboard</h1>", unsafe_allow_html=True)

# ---------------------------
# File Upload
# ---------------------------
@st.cache_data
def load_file(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def infer_types(df: pd.DataFrame):
    """Infer column data types: numeric, categorical, datetime"""
    types = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types

# ---------------------------
# Sidebar Choice
# ---------------------------
mode = st.sidebar.radio("Choose what to analyze:", ["📂 Dataset", "📑 Document"])

# ---------------------------
# Dataset Mode
# ---------------------------
if mode == "📂 Dataset":
    uploaded = st.file_uploader("📂 Upload CSV or Excel file", type=["csv","xlsx"])
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

    # ---------------------------
    # Data Cleaning
    # ---------------------------
    with st.expander("🧹 Data Cleaning Options"):
        drop_dups = st.checkbox("Drop duplicate rows", value=True)
        na_strategy = st.selectbox("Fill NA for numeric columns with:", ["None","Mean","Median"])
        
        if drop_dups:
            df = df.drop_duplicates()
        if na_strategy != "None":
            num_cols = [c for c,t in types.items() if t=="numeric"]
            if num_cols:
                if na_strategy == "Mean":
                    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                else:
                    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ---------------------------
    # Quick Charts
    # ---------------------------
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
        num_cols = [c for c,t in types.items() if t=="numeric"]
        if num_cols:
            ycol = st.selectbox("Numeric column to plot over time", num_cols)
            tmp = df[[chart_col, ycol]].dropna()
            tmp = tmp.sort_values(chart_col)
            tmp = tmp.groupby(chart_col)[ycol].mean()
            st.line_chart(tmp)

    # ---------------------------
    # Anomaly Detection
    # ---------------------------
    st.subheader("🚨 Anomaly Detection")
    num_cols = [c for c,t in types.items() if t=="numeric"]
    if num_cols:
        features = st.multiselect("Select numeric features for anomaly detection", num_cols, default=num_cols[:3])
        if features:
            X = df[features].dropna()
            model = IsolationForest(contamination=0.03, random_state=42)
            preds = model.fit_predict(X)
            outliers = X[preds == -1]
            st.write(f"Flagged anomalies: {len(outliers)}")
            st.dataframe(outliers.head())
            st.download_button("⬇️ Download anomalies CSV", outliers.to_csv().encode("utf-8"),
                            "anomalies.csv", "text/csv")
    else:
        st.info("No numeric columns available for anomaly detection.")

    # ---------------------------
    # Download Cleaned Dataset
    # ---------------------------
    st.subheader("⬇️ Export")
    st.download_button("Download cleaned dataset (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    "cleaned_dataset.csv",
                    "text/csv")

# ---------------------------
# Document Mode
# ---------------------------
else:
    doc_file = st.file_uploader("📑 Upload a document (PDF, DOCX, TXT)", type=["pdf","docx","txt"])
    if not doc_file:
        st.info("Please upload a document to summarize.")
        st.stop()

    text = ""
    if doc_file.name.endswith(".pdf"):
        with fitz.open(stream=doc_file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
    elif doc_file.name.endswith(".docx"):
        text = docx2txt.process(doc_file)
    else:
        stringio = StringIO(doc_file.getvalue().decode("utf-8"))
        text = stringio.read()

    st.subheader("📄 Document Content Preview")
    st.text_area("Extracted text:", text[:1500] + "..." if len(text) > 1500 else text, height=200)

    # 🔹 Placeholder for AI summary
    st.subheader("🤖 AI-Generated Summary")
    st.info("Here you would integrate OpenAI or HuggingFace API to summarize the extracted text.")
    # Example: summary = openai.ChatCompletion.create(...)
