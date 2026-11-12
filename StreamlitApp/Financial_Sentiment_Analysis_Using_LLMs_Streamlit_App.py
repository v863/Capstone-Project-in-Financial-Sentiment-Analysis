import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis Suite",
    page_icon="📈",
    layout="wide"
)

# Custom CSS Styling for Proper Cards & UI Components
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid #dee2e6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc !important;
        color: white !important;
    }
    /* Metric & Container Card Styling */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e3e6f0;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        color: #212529;
        font-weight: 700;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e3e6f0;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Data Acquisition & Caching
@st.cache_resource
def load_and_preprocess_data():
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    csv_path = os.path.join(path, "all-data.csv")
    
    df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
    df = df.drop_duplicates().reset_index(drop=True)
    df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
    
    # NLTK Downloads & Preprocessing
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {
        'not', 'no', 'nor', 'up', 'down', 'above', 'below', 'under', 'over', 'more', 'less'}

    def preprocess_text(text):
        text = str(text).lower()
        tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return ' '.join(tokens)

    df['Cleaned_Headline'] = df['News Headline'].apply(preprocess_text)
    return df

try:
    df = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error loading dataset from KaggleHub: {e}")
    st.stop()

# Sidebar Configuration
st.sidebar.title("⚙️ Engine Controls")
model_choice = st.sidebar.selectbox(
    "Select LLM / Architecture Engine",
    ["Groq (LLaMA-3 / Qwen / Mixtral)", "Fine-Tuned QLoRA Transformer", "Google GenAI (Gemini)", "NVIDIA Nemotron", "FinBERT (Baseline)"]
)

prompt_mode = st.sidebar.radio("Prompting Paradigm", ["Zero-Shot", "Few-Shot"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Use the tabs below to navigate through Exploratory Data Analysis, Benchmark Performances, and Live Inference.")

# Main Layout Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 EDA & Overview", 
    "🏆 Model Performance", 
    "🔍 Live Headline Predictor"
])

# ================= TAB 1: EDA & Overview =================
with tab1:
    st.markdown("### 📊 Financial PhraseBank Dataset Explorer")
    st.markdown("Analyze dataset structure, class balance, and text length distributions evaluated across multi-model architectures.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Samples</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        pos_count = len(df[df['Sentiment'] == 'positive'])
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Positive Sentiment</div>
                <div class="metric-value">{pos_count:,} ({pos_count/len(df)*100:.1f}%)</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        neu_count = len(df[df['Sentiment'] == 'neutral'])
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Neutral Sentiment</div>
                <div class="metric-value">{neu_count:,} ({neu_count/len(df)*100:.1f}%)</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><h4>Sentiment Distribution</h4>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Sentiment', palette='viridis', order=['neutral', 'positive', 'negative'], ax=ax)
        ax.set_title("Class Frequency Bar Chart", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sentiment Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="card"><h4>Raw Dataset Sample</h4>', unsafe_allow_html=True)
        st.dataframe(df[['Sentiment', 'News Headline']].head(8), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: Model Performance =================
with tab2:
    st.markdown("### 🏆 Multi-Model Benchmark Comparison")
    st.markdown("Performance benchmarks comparing **Groq LLMs**, **Fine-Tuned QLoRA**, **Google GenAI**, **Nemotron**, and **FinBERT**.")

    # Benchmark Summary Table
    benchmark_data = {
        "Model / Engine": ["Groq (gpt-oss-120b)", "Groq (qwen-2.5-32b)", "QLoRA Fine-Tuned (Qwen 0.5B)", "Google GenAI (Gemini Flash)", "NVIDIA Nemotron", "FinBERT (Baseline)"],
        "Paradigm": ["Few-Shot", "Few-Shot", "Domain Fine-Tune", "Few-Shot", "Zero-Shot", "Fine-Tuned Transformer"],
        "Accuracy": [0.78, 0.55, 0.82, 0.86, 0.84, 0.85],
        "Precision (Weighted)": [0.82, 0.30, 0.84, 0.87, 0.85, 0.88],
        "Recall (Weighted)": [0.78, 0.55, 0.82, 0.86, 0.84, 0.85],
        "F1-Score (Macro)": [0.77, 0.24, 0.80, 0.85, 0.83, 0.79]
    }
    bench_df = pd.DataFrame(benchmark_data)
    
    st.markdown('<div class="card"><h4>Comparative Performance Metrics</h4>', unsafe_allow_html=True)
    st.dataframe(bench_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card"><h4>Key Findings & Insights</h4>', unsafe_allow_html=True)
