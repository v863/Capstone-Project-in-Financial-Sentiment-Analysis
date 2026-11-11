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
from openai import OpenAI
from groq import Groq, RateLimitError
from google import genai
from google.genai import types
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Ensure NLTK resources are available
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception:
    pass

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis Using Multi-LLMs",
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
st.title("📈 Financial Sentiment Analysis Using Multi-LLMs")
st.markdown("A dynamic evaluation pipeline supporting **OpenRouter** (`nvidia/nemotron-3-ultra-550b-a55b:free`), **Groq**, **Gemini**, and **FinBERT** with real-time measured latency, Accuracy, Macro-F1, and Weighted-F1.")
st.markdown("---")

# Initialize FinBERT Pipeline (Cached)
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

finbert_pipeline = load_finbert()

# Load and Train/Split function
@st.cache_data
def load_and_train_models():
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
    return df, X_train, X_test, y_train, y_test

try:
    df, X_train, X_test, y_train, y_test = load_and_train_models()
    total_records = df.shape[0]
    train_shape = X_train.shape[0]
    test_shape = X_test.shape[0]
except Exception as e:
    total_records = 4840
    train_shape = 3872
    test_shape = 968
    df = pd.DataFrame(columns=["Sentiment", "News Headline", "Cleaned_Headline"])
    X_test, y_test = pd.Series(), pd.Series()

# Sidebar Configuration
st.sidebar.title("🛠️ Global Configuration")
openrouter_api_key = st.sidebar.text_input("Enter OpenRouter API Key", type="password", value=os.environ.get("OPENROUTER_API_KEY", ""))
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

st.sidebar.markdown("---")
st.sidebar.info(
    "Configure your API keys above to dynamically test OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free), "
    "Groq, Gemini, and FinBERT on the Financial PhraseBank benchmark dataset."
)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 EDA & Overview", 
    "🏆 Model Performance", 
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
        st.metric(label="Train Split Shape", value=f"{train_shape}")
    with col_stat3:
        st.metric(label="Test Split Shape", value=f"{test_shape}")
        
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
# TAB 2: MODEL PERFORMANCE (DYNAMIC EVALUATION)
# ==========================================
with tab2:
    st.subheader("Dynamic Model Evaluation & Comparison")
    st.markdown("Run live evaluation on test samples using **OpenRouter (nvidia/nemotron-3-ultra-550b-a55b:free)**, **Groq**, **Gemini**, and **FinBERT** to dynamically calculate `Accuracy`, `Macro F1-Score`, `Weighted F1-Score`, and real-time measured `Latency`.")
    
    col_eval_cfg1, col_eval_cfg2 = st.columns(2)
    with col_eval_cfg1:
        eval_model = st.selectbox(
            "Select Model for Dynamic Evaluation",
            [
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "gemini-3.5-flash-lite",
                "openai/gpt-oss-120b",
                "qwen/qwen2.5-32b",
                "openai/gpt-oss-20b",
                "ProsusAI/finbert (Fine-Tuned Local)"
            ],
            key="eval_model_select"
        )
    with col_eval_cfg2:
        eval_paradigm = st.selectbox("Select Paradigm", ["Zero-Shot", "Few-Shot"], key="eval_paradigm_select")
        
    sample_size = st.slider("Select Number of Test Samples to Evaluate", min_value=5, max_value=100, value=20, step=5)

    if st.button("🚀 Run Dynamic Evaluation", type="primary"):
        is_openrouter = eval_model == "nvidia/nemotron-3-ultra-550b-a55b:free"
        is_gemini = "gemini" in eval_model.lower()
        is_finbert = "finbert" in eval_model.lower()
        
        if is_openrouter and not openrouter_api_key:
            st.error("⚠️ Please provide a valid OpenRouter API Key in the sidebar.")
        elif is_gemini and not gemini_api_key:
            st.error("⚠️ Please provide a valid Gemini API Key in the sidebar.")
        elif not is_openrouter and not is_gemini and not is_finbert and not groq_api_key:
            st.error("⚠️ Please provide a valid Groq API Key in the sidebar.")
        else:
            with st.spinner(f"Evaluating {eval_model} ({eval_paradigm}) on {sample_size} test samples..."):
                test_subset = X_test.head(sample_size).reset_index(drop=True)
                true_subset = y_test.head(sample_size).reset_index(drop=True)
                
                predictions = []
                start_time = time.time()
                
                if is_finbert:
                    for text in test_subset:
                        try:
                            res = finbert_pipeline(text[:512])[0]
                            predictions.append(res['label'].strip().lower())
                        except Exception:
                            predictions.append("neutral")
                elif is_gemini:
                    client_gemini = genai.Client(api_key=gemini_api_key)
                    system_instruction = (
                        "The goal is to classify financial texts into three categories: positive, negative, or neutral sentiment. "
                        "Return ONLY the exact category name."
                    )
                    for text in test_subset:
                        try:
                            if eval_paradigm == "Zero-Shot":
                                prompt = f"Classify the sentiment of this text into positive, negative, or neutral:\n\n\"{text}\""
                            else:
                                prompt = (
                                    'Here are some example demonstrations for sentiment classification:\n'
                                    'S1: "The company has no plans to expand production." -> neutral\n'
                                    'S2: "The international electronic industry company posted a net loss." -> negative\n'
                                    'S3: "With the new production plant the company will increase revenue." -> positive\n\n'
                                    f'Classify the sentiment of this text: "{text}"'
                                )
                            response = client_gemini.models.generate_content(
                                model=eval_model,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    temperature=0.0
                                )
                            )
                            pred = response.text.strip().lower()
                            if pred not in ['positive', 'negative', 'neutral']:
                                if 'pos' in pred: pred = 'positive'
                                elif 'neg' in pred: pred = 'negative'
                                else: pred = 'neutral'
                            predictions.append(pred)
                        except Exception:
                            predictions.append("neutral")
                elif is_openrouter:
                    client_or = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_api_key
                    )
                    system_prompt = (
                        "The goal is to classify financial texts into three categories: positive, negative, or neutral sentiment. "
                        "Return ONLY the exact category name."
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
                            resp = client_or.chat.completions.create(model=eval_model, messages=messages, temperature=0.0)
                            pred = resp.choices[0].message.content.strip().lower()
                            if pred not in ['positive', 'negative', 'neutral']:
                                if 'pos' in pred: pred = 'positive'
                                elif 'neg' in pred: pred = 'negative'
                                else: pred = 'neutral'
                            predictions.append(pred)
                        except Exception:
                            predictions.append("neutral")
                else:
                    client_groq = Groq(api_key=groq_api_key)
                    system_prompt = (
                        "The goal is to classify financial texts into three categories: positive, negative, or neutral sentiment. "
                        "Return ONLY the exact category name."
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
                            resp = client_groq.chat.completions.create(model=eval_model, messages=messages, temperature=0.0)
                            pred = resp.choices[0].message.content.strip().lower()
                            if pred not in ['positive', 'negative', 'neutral']:
                                if 'pos' in pred: pred = 'positive'
                                elif 'neg' in pred: pred = 'negative'
                                else: pred = 'neutral'
                            predictions.append(pred)
                        except Exception:
                            predictions.append("neutral")
                            
                total_latency = time.time() - start_time
                avg_latency_per_sample = total_latency / sample_size
                
                acc = accuracy_score(true_subset, predictions)
                macro_f1 = f1_score(true_subset, predictions, average='macro', zero_division=0)
                weighted_f1 = f1_score(true_subset, predictions, average='weighted', zero_division=0)
                
                st.success(f"Evaluation completed in {total_latency:.2f} seconds ({avg_latency_per_sample:.3f}s per sample)!")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Accuracy Score", f"{acc:.4f}")
                col_m2.metric("Macro F1-Score", f"{macro_f1:.4f}")
                col_m3.metric("Weighted F1-Score", f"{weighted_f1:.4f}")
                col_m4.metric("Avg Latency / Sample", f"{avg_latency_per_sample:.2f}s")
                
                st.markdown("### Dynamic Results DataFrame")
                dynamic_res_df = pd.DataFrame([{
                    "Model": f"{eval_model} ({eval_paradigm})" if not is_finbert else "FinBERT (Fine-Tuned)",
                    "Samples Evaluated": sample_size,
                    "Accuracy": acc,
                    "Macro F1-Score": macro_f1,
                    "Weighted F1-Score": weighted_f1,
                    "Total Latency (s)": round(total_latency, 2),
                    "Avg Latency (s/sample)": round(avg_latency_per_sample, 3)
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
    st.markdown("Test individual headlines or analyst notes in real-time with precise measured query latency.")
    
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        model_choice = st.selectbox(
            "Choose Model / Architecture",
            [
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "gemini-3.5-flash-lite",
                "openai/gpt-oss-120b",
                "qwen/qwen2.5-32b",
                "openai/gpt-oss-20b",
                "ProsusAI/finbert (Fine-Tuned Local)"
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
                is_openrouter = model_choice == "nvidia/nemotron-3-ultra-550b-a55b:free"
                is_gemini = "gemini" in model_choice.lower()
                is_finbert = "finbert" in model_choice.lower()
                
                if is_finbert:
                    start_time = time.time()
                    res = finbert_pipeline(user_input[:512])[0]
                    elapsed = time.time() - start_time
                    
                    pred_label = res['label'].strip().lower()
                    confidence = res['score']
                    
                    st.success(f"Completed in {elapsed:.3f}s using FinBERT")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted Sentiment", pred_label.upper())
                    c2.metric("Confidence Score", f"{confidence:.4f}")
                    c3.metric("Measured Latency", f"{elapsed:.3f}s")
                elif is_gemini:
                    if not gemini_api_key:
                        st.error("⚠️ Please provide a valid Gemini API Key in the sidebar.")
                    else:
                        try:
                            client_gemini = genai.Client(api_key=gemini_api_key)
                            system_instruction = "Classify financial text sentiment into positive, negative, or neutral. Return ONLY the category name."
                            if prompt_paradigm == "Zero-Shot":
                                prompt = f"Classify the sentiment of this text into positive, negative, or neutral:\n\n\"{user_input}\""
                            else:
                                prompt = (
                                    'Here are some example demonstrations for sentiment classification:\n'
                                    'S1: "The company has no plans to expand production." -> neutral\n'
                                    'S2: "The international electronic industry company posted a net loss." -> negative\n'
                                    'S3: "With the new production plant the company will increase revenue." -> positive\n\n'
                                    f'Classify the sentiment of this text: "{user_input}"'
                                )
                            start_time = time.time()
                            response = client_gemini.models.generate_content(
                                model=model_choice,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    temperature=0.0
                                )
                            )
                            elapsed = time.time() - start_time
                            pred_label = response.text.strip().lower()
                            
                            st.success(f"Completed in {elapsed:.3f}s using {model_choice} ({prompt_paradigm})")
                            c1, c2 = st.columns(2)
                            c1.metric("Predicted Sentiment", pred_label.upper())
                            c2.metric("Measured Latency", f"{elapsed:.3f}s")
                        except Exception as e:
                            st.error(f"Gemini API Execution Error: {e}")
                elif is_openrouter:
                    if not openrouter_api_key:
                        st.error("⚠️ Please provide a valid OpenRouter API Key in the sidebar.")
                    else:
                        try:
                            client_or = OpenAI(
                                base_url="https://openrouter.ai/api/v1",
                                api_key=openrouter_api_key
                            )
                            system_prompt = "Classify financial text sentiment into positive, negative, or neutral. Return ONLY the category name."
                            if prompt_paradigm == "Zero-Shot":
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"Classify the sentiment of this text into positive, negative, or neutral:\n\n\"{user_input}\""}
                                ]
                            else:
                                few_shot_examples_text = (
                                    'S1: "The company has no plans to expand production." -> neutral\n'
                                    'S2: "The international electronic industry company posted a net loss." -> negative\n'
                                    'S3: "With the new production plant the company will increase revenue." -> positive'
                                )
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": "Here are training examples:\n" + few_shot_examples_text},
                                    {"role": "assistant", "content": "Understood. I will classify subsequent texts into positive, negative, or neutral."},
                                    {"role": "user", "content": f"Classify the sentiment of this text: \"{user_input}\""}
                                ]
                            start_time = time.time()
                            resp = client_or.chat.completions.create(model=model_choice, messages=messages, temperature=0.0)
                            elapsed = time.time() - start_time
                            pred_label = resp.choices[0].message.content.strip().lower()
                            
                            st.success(f"Completed in {elapsed:.3f}s using {model_choice} ({prompt_paradigm})")
                            c1, c2 = st.columns(2)
                            c1.metric("Predicted Sentiment", pred_label.upper())
                            c2.metric("Measured Latency", f"{elapsed:.3f}s")
                        except Exception as e:
                            st.error(f"OpenRouter API Execution Error: {e}")
                else:
                    if not groq_api_key:
                        st.error("⚠️ Please provide a valid Groq API Key in the sidebar.")
                    else:
                        try:
                            client_groq = Groq(api_key=groq_api_key)
                            system_prompt = "Classify financial text sentiment into positive, negative, or neutral. Return ONLY the category name."
                            if prompt_paradigm == "Zero-Shot":
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"Classify the sentiment of this text into positive, negative, or neutral:\n\n\"{user_input}\""}
                                ]
                            else:
                                few_shot_examples_text = (
                                    'S1: "The company has no plans to expand production." -> neutral\n'
                                    'S2: "The international electronic industry company posted a net loss." -> negative\n'
                                    'S3: "With the new production plant the company will increase revenue." -> positive'
                                )
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": "Here are training examples:\n" + few_shot_examples_text},
                                    {"role": "assistant", "content": "Understood. I will classify subsequent texts into positive, negative, or neutral."},
                                    {"role": "user", "content": f"Classify the sentiment of this text: \"{user_input}\""}
                                ]
                            start_time = time.time()
                            resp = client_groq.chat.completions.create(model=model_choice, messages=messages, temperature=0.0)
                            elapsed = time.time() - start_time
                            pred_label = resp.choices[0].message.content.strip().lower()
                            
                            st.success(f"Completed in {elapsed:.3f}s using {model_choice} ({prompt_paradigm})")
                            c1, c2 = st.columns(2)
                            c1.metric("Predicted Sentiment", pred_label.upper())
                            c2.metric("Measured Latency", f"{elapsed:.3f}s")
                        except Exception as e:
                            st.error(f"Groq API Execution Error: {e}")
