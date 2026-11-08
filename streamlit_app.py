import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Sentiment Analysis Suite",
    page_icon="📈",
    layout="wide"
)

# --- Initialize NLTK Resources ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# --- Data Loading & Model Training Pipeline ---
@st.cache_resource
def load_and_train_models():
    # 1. Download & Load Dataset from Kaggle Hub
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    csv_path = os.path.join(path, "all-data.csv")
    
    df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
    df = df.drop_duplicates().reset_index(drop=True)
    df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
    
    # 2. Preprocessing setup
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {
        'not', 'no', 'nor', 'up', 'down', 'above', 'below', 'under', 'over', 'more', 'less'
    }
    
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        return ' '.join(tokens)
    
    df['Cleaned_Headline'] = df['News Headline'].apply(preprocess_text)
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Headline'],
        df['Sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['Sentiment']
    )
    
    # 4. TF-IDF Vectorization
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 5. Train Core Classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append({"Model": name, "Accuracy": acc, "Macro F1-Score": f1})
        
    results_df = pd.DataFrame(results)
    
    return df, trained_models, tfidf, results_df, X_test_tfidf, y_test, preprocess_text

# Execute loading and training cached block
with st.spinner("Downloading financial dataset and training models... Please wait."):
    df, trained_models, tfidf, results_df, X_test_tfidf, y_test, preprocess_text = load_and_train_models()

# --- Streamlit Dashboard Layout ---
st.title("📊 Financial News Sentiment Analysis Platform")
st.markdown("A unified platform featuring classical machine learning performance metrics, real-time headline inference, and LLM benchmark summaries.")

tabs = st.tabs(["🚀 Real-Time Predictor", "📈 Exploratory Data Analysis & Models", "🌐 LLM Benchmark Integration"])

# --- TAB 1: REAL-TIME PREDICTOR ---
with tabs[0]:
    st.header("Real-Time Headline Sentiment Inference")
    st.markdown("Type or paste a financial news headline to analyze its market sentiment direction using our trained models.")
    
    user_input = st.text_area("Enter News Headline:", "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period.")
    
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a valid headline text.")
        else:
            cleaned_input = preprocess_text(user_input)
            input_vector = tfidf.transform([cleaned_input])
            
            cols = st.columns(len(trained_models))
            for i, (name, model) in enumerate(trained_models.items()):
                pred = model.predict(input_vector)[0].upper()
                with cols[i]:
                    st.subheader(name)
                    if pred == "POSITIVE":
                        st.success(f"🎯 {pred}")
                    elif pred == "NEGATIVE":
                        st.error(f"📉 {pred}")
                    else:
                        st.info(f"⚖️ {pred}")

# --- TAB 2: EDA & MODEL COMPARISON ---
with tabs[1]:
    st.header("Exploratory Data Analysis & Core Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Sentiment', palette='viridis', order=['neutral', 'positive', 'negative'], ax=ax)
        ax.set_title('Sentiment Distribution in Financial News')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Model Performance Comparison Table")
        st.dataframe(results_df.style.format({"Accuracy": "{:.4f}", "Macro F1-Score": "{:.4f}"}), use_container_width=True)
        
    st.divider()
    st.subheader("Performance Bar Chart (Accuracy vs. Macro F1-Score)")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    x = np.arange(len(results_df['Model']))
    width = 0.35
    rects1 = ax_bar.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', color='#2b5c8f')
    rects2 = ax_bar.bar(x + width/2, results_df['Macro F1-Score'], width, label='Macro F1-Score', color='#38a3a5')
    ax_bar.set_ylabel('Score')
    ax_bar.set_title('Model Performance Comparison')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(results_df['Model'])
    ax_bar.set_ylim(0, 1.0)
    ax_bar.legend()
    st.pyplot(fig_bar)
    
    st.divider()
    st.subheader("Confusion Matrices")
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = ['negative', 'neutral', 'positive']
    for ax, (name, model) in zip(axes_cm, trained_models.items()):
        preds = model.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, preds, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

# --- TAB 3: LLM BENCHMARK INTEGRATION ---
with tabs[2]:
    st.header("Multi-Model LLM & Fine-Tuned Transformer Benchmarks")
    st.markdown("Comparison framework tracking zero-shot/few-shot evaluations against domain-specific models like **FinBERT**[cite: 2, 3].")
    
    llm_benchmark_data = {
        "Model Configuration": [
            "openai/gpt-oss-120b (Zero-shot)", "openai/gpt-oss-120b (Few-shot)",
            "qwen/qwen3.8-27b (Zero-shot)", "qwen/qwen3.8-27b (Few-shot)",
            "openai/gpt-oss-20b (Zero-shot)", "openai/gpt-oss-20b (Few-shot)",
            "ProsusAI/finbert (Fine-Tuned)"
        ],
        "Accuracy": [0.75, 0.80, 0.82, 0.82, 0.78, 0.80, 0.85],
        "Precision": [0.79, 0.84, 0.85, 0.87, 0.78, 0.84, 0.88],
        "Recall": [0.75, 0.80, 0.82, 0.82, 0.78, 0.80, 0.85],
        "F1-score": [0.75, 0.80, 0.83, 0.83, 0.77, 0.80, 0.85]
    }
    
    llm_df = pd.DataFrame(llm_benchmark_data)
    st.dataframe(llm_df, use_container_width=True)
    
    st.info("💡 **Insight:** Domain-specific fine-tuned transformers like FinBERT outperform general-purpose zero-shot LLM configurations due to specialized financial corpus pre-training[cite: 2, 3]. However, few-shot prompting approaches significantly close the performance gap for high-capacity models.")
