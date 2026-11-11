import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from groq import Groq, RateLimitError
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Ensure NLTK resources are available
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception:
    pass

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis Using LLMs & ML",
    page_icon="📈",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e2ebfa;
        border-radius: 5px 5px 0px 0px;
        font-weight: 600;
        color: #174EA6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b5c8f !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Header
st.title("📈 Financial Sentiment Analysis: Dynamic ML & LLM Pipeline")
st.markdown("A dynamic evaluation pipeline comparing Statistical Classifiers (TF-IDF + LogReg/SVM/RF), FinBERT, and Groq LLMs on the Financial PhraseBank dataset.")
st.markdown("---")

# Initialize FinBERT Pipeline (Cached)
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

finbert_pipeline = load_finbert()

# Load and Preprocess Dataset & Train TF-IDF Models
@st.cache_data
def load_and_train_ml_models():
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    csv_path = os.path.join(path, "all-data.csv")
    
    df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
    df = df.drop_duplicates().reset_index(drop=True)
    df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {
        'not', 'no', 'nor', 'up', 'down', 'above', 'below', 'under', 'over', 'more', 'less'
    }
    
    def preprocess_text(text):
        words = str(text).lower().split()
        tokens = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(tokens)
    
    df['Cleaned_Headline'] = df['News Headline'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Headline'],
        df['Sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['Sentiment']
    )
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Classical Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": SVC(kernel='linear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
        
    return df, X_test, y_test, X_test_tfidf, vectorizer, trained_models

try:
    df, X_test, y_test, X_test_tfidf, vectorizer, trained_models = load_and_train_ml_models()
    total_records = df.shape[0]
    test_shape = X_test.shape[0]
except Exception as e:
    total_records = 4840
    test_shape = 968
    df = pd.DataFrame(columns=["Sentiment", "News Headline", "Cleaned_Headline"])
    X_test, y_test = pd.Series(), pd.Series()
    trained_models = {}

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
    "This application evaluates financial text sentiment dynamically across TF-IDF ML classifiers, "
    "FinBERT, and Groq LLMs using the Financial PhraseBank benchmark dataset."
)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 EDA & Overview", 
    "🏆 Model Performance & Benchmark", 
    "🔍 Live Headline Predictor"
])

# ==========================================
# TAB 1: EDA & OVERVIEW
# ==========================================
with tab1:
    st.header("Exploratory Data Analysis & Dataset Overview")
    st.markdown("Explore the benchmark dataset distribution, statistics, and sample financial headlines.")
    
    st.markdown("### Dataset Statistics")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric(label="Total Records Loaded", value=f"{total_records:,}")
    with col_stat2:
        st.metric(label="Test Split Shape", value=f"{test_shape}")
    with col_stat3:
        st.metric(label="Classes", value="positive, neutral, negative")
        
    st.markdown("---")
    
    display_df = df if not df.empty else pd.DataFrame({
        "Sentiment": ["neutral", "positive", "negative"],
        "News Headline": ["Sample 1", "Sample 2", "Sample 3"]
    })
    
    filter_sentiment = st.selectbox("Filter by Sentiment Label", ["All", "positive", "negative", "neutral"])
    if filter_sentiment != "All":
        display_df = display_df[display_df["Sentiment"] == filter_sentiment]
        
    st.dataframe(display_df.head(100), use_container_width=True)
    
    st.markdown("### Sentiment Class Distribution")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Sentiment', palette='viridis', order=['neutral', 'positive', 'negative'], ax=ax)
        ax.set_title('Sentiment Distribution in Financial News')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

# ==========================================
# TAB 2: MODEL PERFORMANCE & BENCHMARK
# ==========================================
with tab2:
    st.subheader("Dynamic Model Evaluation & Benchmarking")
    st.markdown("Run live evaluation on test samples using **TF-IDF ML Classifiers** (Logistic Regression, Linear SVM, Random Forest), **FinBERT**, and **Groq LLMs** (`gpt-oss-120b`, `qwen3.8-27b`, `gpt-oss-20b`) to dynamically compute `Accuracy`, `Macro F1-Score`, and `Weighted F1-Score`.")
    
    col_eval_cfg1, col_eval_cfg2 = st.columns(2)
    with col_eval_cfg1:
        eval_model = st.selectbox(
            "Select Model / Classifier",
            [
                "Logistic Regression (TF-IDF)",
                "Linear SVM (TF-IDF)",
                "Random Forest (TF-IDF)",
                "ProsusAI/finbert (Fine-Tuned Local)",
                "gpt-oss-120b",
                "qwen3.8-27b",
                "gpt-oss-20b"
            ],
            key="eval_model_select"
        )
    with col_eval_cfg2:
        eval_paradigm = st.selectbox("Select Paradigm (for LLMs)", ["Zero-Shot", "Few-Shot"], key="eval_paradigm_select")
        
    sample_size = st.slider("Select Number of Test Samples to Evaluate", min_value=10, max_value=200, value=30, step=10)

    if st.button("🚀 Run Dynamic Evaluation", type="primary"):
        is_llm = eval_model in ["gpt-oss-120b", "qwen3.8-27b", "gpt-oss-20b"]
        if is_llm and not groq_api_key:
            st.error("⚠️ Please provide a valid Groq API Key in the sidebar for Groq LLMs.")
        else:
            with st.spinner(f"Evaluating {eval_model} on {sample_size} test samples..."):
                test_subset = X_test.head(sample_size).reset_index(drop=True)
                true_subset = y_test.head(sample_size).reset_index(drop=True)
                
                predictions = []
                start_time = time.time()
                
                if "TF-IDF" in eval_model:
                    model_key = eval_model.split(" (")[0]
                    clf = trained_models[model_key]
                    subset_tfidf = vectorizer.transform(test_subset)
                    predictions = clf.predict(subset_tfidf)
                elif "finbert" in eval_model.lower():
                    for text in test_subset:
                        try:
                            res = finbert_pipeline(text[:512])[0]
                            predictions.append(res['label'].strip().lower())
                        except Exception:
                            predictions.append("neutral")
                else:
                    client = Groq(api_key=groq_api_key)
                    system_prompt = (
                        "The goal is to classify financial texts (e.g., news articles, analyst reports, or company announcements) "
                        "into three categories: positive, negative, or neutral sentiment. Return ONLY the exact category name."
                    )
                    for text in test_subset:
                        try:
                            if eval_paradigm == "Zero-Shot":
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"Classify the sentiment of this text into positive, negative, or neutral:\n\n\"{text}\""}
                                ]
                            else:
                                messages = [
                                    {"role": "system", "content": "Here are some example demonstrations for sentiment classification. " + system_prompt},
                                    {"role": "user", "content": 'S1: "The company has no plans to expand production." -> neutral\nS2: "The international electronic industry company posted a net loss." -> negative\nS3: "With the new production plant the company will increase revenue." -> positive'},
                                    {"role": "assistant", "content": "Understood. I will classify subsequent texts into positive, negative, or neutral."},
                                    {"role": "user", "content": f"Classify the sentiment of this text:\n\n\"{text}\""}
                                ]
                            resp = safe_chat_completion(client=client, messages=messages, model=eval_model, temperature=0.0)
                            pred = resp.choices[0].message.content.strip().lower()
                            if pred not in ['positive', 'negative', 'neutral']:
                                if 'pos' in pred: pred = 'positive'
                                elif 'neg' in pred: pred = 'negative'
                                else: pred = 'neutral'
                            predictions.append(pred)
                        except Exception:
                            predictions.append("neutral")
                            
                elapsed_time = time.time() - start_time
                
                acc = accuracy_score(true_subset, predictions)
                macro_f1 = f1_score(true_subset, predictions, average='macro', zero_division=0)
                weighted_f1 = f1_score(true_subset, predictions, average='weighted', zero_division=0)
                
                st.success(f"Evaluation completed in {elapsed_time:.2f} seconds!")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Accuracy Score", f"{acc:.4f}")
                col_m2.metric("Macro F1-Score", f"{macro_f1:.4f}")
                col_m3.metric("Weighted F1-Score", f"{weighted_f1:.4f}")
                
                st.markdown("### Dynamic Results DataFrame")
                dynamic_res_df = pd.DataFrame([{
                    "Model": f"{eval_model} ({eval_paradigm})" if is_llm else eval_model,
                    "Samples Evaluated": sample_size,
                    "Accuracy": acc,
                    "Macro F1-Score": macro_f1,
                    "Weighted F1-Score": weighted_f1,
                    "Latency (s)": round(elapsed_time, 2)
                }])
                st.dataframe(dynamic_res_df.style.highlight_max(axis=0, subset=['Accuracy', 'Macro F1-Score', 'Weighted F1-Score'], color='#a8df8e'), use_container_width=True)
                
                st.markdown("### Confusion Matrix")
                labels = ['negative', 'neutral', 'positive']
                cm = confusion_matrix(true_subset, predictions, labels=labels)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
                ax_cm.set_xlabel('Predicted Sentiment')
                ax_cm.set_ylabel('True Sentiment')
                st.pyplot(fig_cm)

# ==========================================
# TAB 3: LIVE HEADLINE PREDICTOR
# ==========================================
with tab3:
    st.header("Live Headline Predictor")
    st.markdown("Test individual headlines or analyst notes in real-time across statistical models, FinBERT, and Groq LLMs.")
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        model_choice = st.selectbox(
            "Choose Model / Architecture",
            [
                "gpt-oss-120b",
                "qwen3.8-27b",
                "gpt-oss-20b",
                "ProsusAI/finbert (Fine-Tuned Local)",
                "Logistic Regression (TF-IDF)",
                "Linear SVM (TF-IDF)",
                "Random Forest (TF-IDF)"
            ],
            key="live_model_choice"
        )
    with col_cfg2:
        prompt_paradigm = st.selectbox("Select Paradigm (for LLMs)", ["Zero-Shot", "Few-Shot"], key="live_prompt_paradigm")

    user_input = st.text_area(
        "Enter Financial Headline/News Text:",
        "The international electronic industry company posted a net loss for the second quarter, missing estimates."
    )

    if st.button("Run Live Prediction", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Processing prediction..."):
                if "TF-IDF" in model_choice:
                    start_time = time.time()
                    model_key = model_choice.split(" (")[0]
                    clf = trained_models[model_key]
                    cleaned_input = vectorizer.transform([user_input])
                    pred_label = clf.predict(cleaned_input)[0]
                    elapsed = time.time() - start_time
                    
                    st.success(f"Completed in {elapsed:.2f}s using {model_choice}")
                    st.metric("Predicted Sentiment", pred_label.upper())
                    
                elif "finbert" in model_choice.lower():
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
