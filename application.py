import streamlit as st # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="AI-Assisted Data Insights", layout="wide")
st.title("📊 AI-Assisted Data Insights Dashboard")

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

uploaded = st.file_uploader("📂 Upload CSV or Excel file", type=["csv","xlsx"])
if not uploaded:
    st.info("Please upload a dataset to get started.")
    st.stop()

df = load_file(uploaded)

# ---------------------------
# Preview & Summary
# ---------------------------
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
    # Try to pick a numeric target to plot over time
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
# AI-Like Report (Offline)
# ---------------------------
st.subheader("📝 Generate Report (Offline AI Summary)")

if st.button("Generate Summary Report"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("report.pdf")
    story = []

    # Add title
    story.append(Paragraph("Dataset Summary Report", styles['Title']))
    story.append(Spacer(1, 20))

    # Dataset info
    summary_text = f"""
    This dataset contains <b>{len(df)}</b> rows and <b>{df.shape[1]}</b> columns.<br/>
    Missing values detected: <b>{df.isna().sum().sum()}</b><br/>
    Duplicate rows: <b>{df.duplicated().sum()}</b><br/>
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # Column descriptions
    for col, t in types.items():
        desc = f"Column <b>{col}</b> is of type <b>{t}</b>."
        if t == "numeric":
            desc += f" Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}, Min: {df[col].min()}, Max: {df[col].max()}"
        elif t == "categorical":
            desc += f" Unique categories: {df[col].nunique()}"
        story.append(Paragraph(desc, styles['Normal']))
        story.append(Spacer(1, 10))

    # Build PDF
    doc.build(story)

    # Let user download
    with open("report.pdf", "rb") as f:
        st.download_button("⬇️ Download Report (PDF)", f, "report.pdf", "application/pdf")
