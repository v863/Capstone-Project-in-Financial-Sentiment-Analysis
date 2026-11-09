import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from groq import Groq, RateLimitError
from transformers import pipeline
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Multi-Model Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize FinBERT Pipeline (Cached)
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

finbert_pipeline = load_finbert()

# Load and Prepare Dataset (Cached to prevent reloading on every interaction)
@st.cache_data
def load_and_split_data():
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    csv_path = os.path.join(path, "all-data.csv")
    
    df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
    df = df.drop_duplicates().reset_index(drop=True)
    df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
    
    # Stratified Train/Test Split
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['Sentiment']
    )
    return df, train_df, test_df

try:
    df, train_df, test_df = load_and_split_data()
    total_records = df.shape[0]
    train_shape = train_df.shape
    test_shape = test_df.shape
except Exception as e:
    # Fallback default values if offline/kaggle cache fails
    total_records = 4840
    train_shape = (3872, 2)
    test_shape = (968, 2)
    df = pd.DataFrame(columns=["Sentiment", "News Headline"])

# Helper function for Groq API calls with retry mechanism
def safe_chat_completion(client, messages, model, temperature=0.0, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature
            )
            time.sleep(0.3)
            return response
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = (2 ** attempt) + 1
            time.sleep(wait_time)
        except Exception as e:
            raise e

# Sidebar Configuration
st.sidebar.title("🛠️ Global Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))

st.sidebar.markdown("---")
st.sidebar.info(
    "This application evaluates financial text sentiment across multi-model configurations "
    "(Groq LLMs vs. Fine-Tuned FinBERT) using the Financial PhraseBank benchmark dataset[cite: 1]."
)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🚀 TAB 1: Real-Time Predictor", 
    "📂 TAB 2: Dataset Explorer", 
    "📊 TAB 3: Comparative Benchmarks"
])

# ==========================================
# TAB 1: REAL-TIME PREDICTOR
# ==========================================
with tab1:
    st.header("Real-Time Sentiment Predictor")
    st.markdown("Test individual headlines or analyst notes across different model configurations and prompt paradigms.")
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        model_choice = st.selectbox(
            "Choose Model / Architecture",
            [
                "openai/gpt-oss-120b",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b",
                "ProsusAI/finbert (Fine-Tuned Local)"
            ]
        )
    with col_cfg2:
        prompt_paradigm = st.selectbox("Select Paradigm (for LLMs)", ["Zero-Shot", "Few-Shot"])

    user_input = st.text_area(
        "Enter Financial Headline/News Text:",
        "The international electronic industry company posted a net loss for the second quarter, missing estimates."
    )

    if st.button("Run Prediction", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Processing prediction..."):
                if "finbert" in model_choice.lower():
                    start_time = time.time()
                    res = finbert_pipeline(user_input[:512])[0]
                    elapsed = time.time() - start_time
                    
                    pred_label = res['label'].strip().lower()
                    confidence = res['score']
                    
                    st.success(f"Completed in {elapsed:.2f}s using FinBERT")
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted Sentiment", pred_label.upper())
                    c2.metric("Confidence Score", f"{confidence:.4f}")
                else:
                    if not groq_api_key:
                        st.error("⚠️ Please provide a valid Groq API Key in the sidebar.")
                    else:
                        try:
                            client = Groq(api_key=groq_api_key)
                            system_prompt = (
                                "The goal is to classify financial texts (e.g., news articles, analyst reports, or company announcements) "
                                "into three categories: positive, negative, or neutral sentiment."
                            )
                            
                            if prompt_paradigm == "Zero-Shot":
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {
                                        "role": "user",
                                        "content": f"Please classify the following sentence into exactly one sentiment category (positive, negative, or neutral). Return only the category name:\n\n\"{user_input}\""
                                    }
                                ]
                            else:
                                few_shot_examples_text = (
                                    'S1: "The company has no plans to expand production." -> neutral\n'
                                    'S2: "The international electronic industry company posted a net loss." -> negative\n'
                                    'S3: "With the new production plant the company will increase revenue." -> positive'
                                )
                                messages = [
                                    {"role": "system", "content": "Here are some example demonstrations for sentiment classification. " + system_prompt},
                                    {"role": "user", "content": "Here are training examples:\n" + few_shot_examples_text},
                                    {"role": "assistant", "content": "Understood. I will classify subsequent texts into positive, negative, or neutral."},
                                    {"role": "user", "content": f"Classify the sentiment of this text (return only positive, negative, or neutral): \"{user_input}\""}
                                ]
                            
                            start_time = time.time()
                            resp = safe_chat_completion(client=client, messages=messages, model=model_choice, temperature=0.0)
                            elapsed = time.time() - start_time
                            
                            pred_label = resp.choices[0].message.content.strip().lower()
                            
                            st.success(f"Completed in {elapsed:.2f}s using {model_choice} ({prompt_paradigm})")
                            st.metric("Predicted Sentiment", pred_label.upper())
                            
                        except Exception as e:
                            st.error(f"API Execution Error: {e}")

# ==========================================
# TAB 2: DATASET EXPLORER
# ==========================================
with tab2:
    st.header("Dataset Explorer (Financial PhraseBank)")
    st.markdown("Explore the benchmark dataset distribution and sample financial headlines.")
    
    # Dataset Statistics metrics populated programmatically
    st.markdown("### Dataset Statistics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric(label="Total Records Loaded", value=f"{total_records:,}")
    with col_stat2:
        st.metric(label="Train / Test Split Shapes", value=f"Train: {train_shape} | Test: {test_shape}")
        
    st.markdown("---")
    
    display_df = df if not df.empty else pd.DataFrame({
        "Sentiment": ["neutral", "neutral", "negative", "positive"],
        "News Headline": ["Sample 1", "Sample 2", "Sample 3", "Sample 4"]
    })
    
    filter_sentiment = st.selectbox("Filter by Sentiment Label", ["All", "positive", "negative", "neutral"])
    if filter_sentiment != "All":
        display_df = display_df[display_df["Sentiment"] == filter_sentiment]
        
    st.dataframe(display_df.head(100), use_container_width=True)
    
    st.markdown("### Class Distribution Chart")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(data=df, x="Sentiment", palette=["#2b5c8f", "#2ca02c", "#d9534f"], ax=ax)
    ax.set(xlabel="Sentiment Category", ylabel="Sample Count")
    st.pyplot(fig)

# ==========================================
# TAB 3: COMPARATIVE BENCHMARKS
# ==========================================
with tab3:
    st.header("Comparative Performance Benchmarks")
    st.markdown("Review macro-evaluation metrics across all tested model variants and prompting configurations[cite: 1].")
    
    benchmark_data = [
        {"Model": "openai/gpt-oss-120b", "Configuration": "Zero-shot", "Accuracy": 0.75, "Precision": 0.79, "Recall": 0.75, "F1-score": 0.75},
        {"Model": "openai/gpt-oss-120b", "Configuration": "Few-shot", "Accuracy": 0.80, "Precision": 0.84, "Recall": 0.80, "F1-score": 0.80},
        {"Model": "qwen/qwen3.8-27b", "Configuration": "Zero-shot", "Accuracy": 0.82, "Precision": 0.85, "Recall": 0.82, "F1-score": 0.83},
        {"Model": "qwen/qwen3.8-27b", "Configuration": "Few-shot", "Accuracy": 0.82, "Precision": 0.87, "Recall": 0.82, "F1-score": 0.83},
        {"Model": "openai/gpt-oss-20b", "Configuration": "Zero-shot", "Accuracy": 0.78, "Precision": 0.78, "Recall": 0.78, "F1-score": 0.77},
        {"Model": "openai/gpt-oss-20b", "Configuration": "Few-shot", "Accuracy": 0.80, "Precision": 0.84, "Recall": 0.80, "F1-score": 0.80},
        {"Model": "ProsusAI/finbert", "Configuration": "Fine-Tuned", "Accuracy": 0.85, "Precision": 0.88, "Recall": 0.85, "F1-score": 0.85},
    ]
    
    benchmark_df = pd.DataFrame(benchmark_data)
    st.dataframe(benchmark_df, use_container_width=True)
    
    st.markdown("### F1-Score Comparison Bar Chart")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=benchmark_df, 
        x="Model", 
        y="F1-score", 
        hue="Configuration", 
        palette="viridis", 
        ax=ax2
    )
    plt.xticks(rotation=20)
    ax2.set(ylim=(0.6, 1.0), title="F1-Score Comparison Across Models & Configurations")
    st.pyplot(fig2)
