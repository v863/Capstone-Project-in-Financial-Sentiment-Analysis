import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

warnings.filterwarnings('ignore')

# Download required NLTK assets quietly
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# --- Load and Cache Dataset ---
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    csv_path = os.path.join(path, "all-data.csv")
    
    # Load dataset with latin-1 encoding and assign column names
    df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
    
    # Clean duplicates and normalize labels
    df = df.drop_duplicates().reset_index(drop=True)
    df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
    return df

with st.spinner("Downloading and loading the Financial News Sentiment dataset..."):
    df = load_data()

# --- Text Preprocessing Setup ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {
    'not', 'no', 'nor', 'up', 'down', 'above', 'below', 'under', 'over', 'more', 'less'
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

# --- Train Models and Cache ---
@st.cache_resource
def train_models(data_df):
    data_df['Cleaned_Headline'] = data_df['News Headline'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data_df['Cleaned_Headline'],
        data_df['Sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=data_df['Sentiment']
    )
    
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42, probability=True),
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
    return trained_models, tfidf, results_df, X_test_tfidf, y_test

with st.spinner("Training Machine Learning models (Logistic Regression, SVM, Random Forest)..."):
    trained_models, tfidf, results_df, X_test_tfidf, y_test = train_models(df)

# --- UI Layout ---
st.title("📊 Financial News Sentiment Analysis App")
st.markdown("""
This app analyzes financial news headlines[cite: 1] using Natural Language Processing (NLP) and Machine Learning models trained on the Financial PhraseBank dataset.
""")

# Sidebar controls
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose a section", [
    "Single Headline Predictor", 
    "Model Performance Comparison", 
    "Exploratory Data Analysis"
])

if app_mode == "Single Headline Predictor":
    st.subheader("Test Custom Financial Headlines")
    user_input = st.text_area("Enter a financial headline:", "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period.")
    
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a valid headline.")
        else:
            cleaned = preprocess_text(user_input)
            vec = tfidf.transform([cleaned])
            
            st.markdown("### Predictions:")
            cols = st.columns(len(trained_models))
            
            for idx, (name, model) in enumerate(trained_models.items()):
                pred = model.predict(vec)[0].upper()
                with cols[idx]:
                    st.metric(label=name, value=pred)

elif app_mode == "Model Performance Comparison":
    st.subheader("Model Evaluation Metrics")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Macro F1-Score']))
    
    # Accuracy & Macro F1 Bar Chart
    st.markdown("### Performance Comparison Bar Chart")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(results_df['Model']))
    width = 0.35

    rects1 = ax.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, results_df['Macro F1-Score'], width, label='Macro F1-Score', color='#38a3a5')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Logistic Regression vs. SVM vs. Random Forest')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.set_ylim(0, 1.0)
    ax.legend()

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    
    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    labels = ['negative', 'neutral', 'positive']
    fig_cm, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, (name, model) in zip(axes, trained_models.items()):
        preds = model.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, preds, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set_title(f"{name} CM")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    st.pyplot(fig_cm)

elif app_mode == "Exploratory Data Analysis":
    st.subheader("Dataset Insights & Distribution")
    st.write(f"Total Cleaned Records: **{df.shape[0]}**")
    
    st.markdown("### Sentiment Class Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x='Sentiment', palette='viridis', order=['neutral', 'positive', 'negative'], ax=ax)
    ax.set_title('Sentiment Distribution in Financial News')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    st.markdown("### Raw Dataset Sample")
    st.dataframe(df.head(10))