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
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings('ignore')

# Download required NLTK resources silently
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_resources()

# Page configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis",
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
st.title("📊 Financial Sentiment Analysis Using Statistical Models")
st.markdown("A production-grade NLP application for classifying financial news headlines into **Positive**, **Neutral**, or **Negative** sentiments using TF-IDF and classical machine learning models.")

# Load and Cache Dataset & Models using st.cache_resource
@st.cache_resource
def load_and_train_models():
    # 1. Download & Load Dataset
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    csv_path = os.path.join(path, "all-data.csv")
    
    df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
    df = df.drop_duplicates().reset_index(drop=True)
    df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
    
    # 2. Text Preprocessing
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
    
    df['Cleaned_Headline'] = df['News Headline'].apply(preprocess_text)
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Headline'],
        df['Sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['Sentiment']
    )
    
    # 4. TF-IDF Vectorizer
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 5. Train Models
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
    
    return df, trained_models, tfidf, results_df, X_test, y_test, X_test_tfidf

# Execute loading/training pipeline
with st.spinner("⏳ Loading financial news dataset and training statistical models (TF-IDF + LR/SVM/RF)..."):
    df, trained_models, tfidf, results_df, X_test, y_test, X_test_tfidf = load_and_train_models()

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Model:",
    list(trained_models.keys()),
    index=0
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset Source:** Kaggle - Sentiment Analysis for Financial News\n\n"
    "**Algorithms Used:**\n"
    "- Logistic Regression\n"
    "- Support Vector Machine (Linear SVM)\n"
    "- Random Forest Classifier"
)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 EDA & Overview", 
    "🏆 Model Performance", 
    "🌳 Interpretability (XAI)", 
    "🔍 Live Headline Predictor"
])

with tab1:
    st.subheader("Dataset Exploratory Data Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Dataset Preview:**")
        st.dataframe(df[['Sentiment', 'News Headline']].head(10), use_container_width=True)
        st.write(f"**Total Cleaned Samples:** {df.shape[0]}")
        
    with col2:
        st.write("**Sentiment Class Distribution:**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Sentiment', palette='viridis', order=['neutral', 'positive', 'negative'], ax=ax)
        ax.set_title('Sentiment Distribution in Financial News')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

with tab2:
    st.subheader("Model Evaluation & Comparison")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Macro F1-Score'], color='#a8df8e'), use_container_width=True)
    
    st.markdown("### Accuracy & Macro F1-Score Comparison")
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
    
    st.markdown("### Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = ['negative', 'neutral', 'positive']
    
    for ax, (name, model) in zip(axes, trained_models.items()):
        preds = model.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, preds, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Model Interpretability & Explainable AI (XAI)")
    
    analysis_type = st.radio("Select Interpretability View:", ["Feature Importance / Top Words", "Decision Tree Visualizations"])
    
    feature_names = tfidf.get_feature_names_out()
    
    if analysis_type == "Feature Importance / Top Words":
        st.markdown("### Top Indicative Financial Words per Model")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # LR Top Words
        lr_model = trained_models["Logistic Regression"]
        lr_coef = lr_model.coef_[2] - lr_model.coef_[0]
        top_lr_idx = np.argsort(np.abs(lr_coef))[-15:]
        axes[0].barh(feature_names[top_lr_idx], lr_coef[top_lr_idx], color='#2b5c8f')
        axes[0].set_title("Logistic Regression: Top 15 Words")
        axes[0].set_xlabel("Coefficient Weight")
        
        # SVM Top Words
        svm_model = trained_models["SVM"]
        svm_coef = svm_model.coef_[2].toarray().flatten() - svm_model.coef_[0].toarray().flatten()
        top_svm_idx = np.argsort(np.abs(svm_coef))[-15:]
        axes[1].barh(feature_names[top_svm_idx], svm_coef[top_svm_idx], color='#38a3a5')
        axes[1].set_title("Linear SVM: Top 15 Words")
        axes[1].set_xlabel("Hyperplane Weight")
        
        # RF Top Words
        rf_model = trained_models["Random Forest"]
        rf_importances = rf_model.feature_importances_
        top_rf_idx = np.argsort(rf_importances)[-15:]
        axes[2].barh(feature_names[top_rf_idx], rf_importances[top_rf_idx], color='#57cc99')
        axes[2].set_title("Random Forest: Top 15 Words")
        axes[2].set_xlabel("Gini Feature Importance")
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.markdown("### Decision Trees (Random Forest & Surrogate Trees for Linear Models)")
        with st.spinner("Generating tree visualizations..."):
            fig, axes = plt.subplots(3, 1, figsize=(20, 24))
            class_names = ['negative', 'neutral', 'positive']
            
            # 1. RF Tree
            rf_single_tree = trained_models["Random Forest"].estimators_[0]
            plot_tree(rf_single_tree, max_depth=3, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10, ax=axes[0])
            axes[0].set_title("1. Random Forest — Actual Internal Decision Tree (Estimator #0, Depth=3)", fontsize=14, fontweight='bold')
            
            # 2. LR Surrogate
            lr_preds_train = trained_models["Logistic Regression"].predict(tfidf.transform(df['Cleaned_Headline']))
            lr_surr = DecisionTreeClassifier(max_depth=3, random_state=42).fit(tfidf.transform(df['Cleaned_Headline']), lr_preds_train)
            plot_tree(lr_surr, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10, ax=axes[1])
            axes[1].set_title("2. Logistic Regression — Surrogate Decision Tree", fontsize=14, fontweight='bold')
            
            # 3. SVM Surrogate
            svm_preds_train = trained_models["SVM"].predict(tfidf.transform(df['Cleaned_Headline']))
            svm_surr = DecisionTreeClassifier(max_depth=3, random_state=42).fit(tfidf.transform(df['Cleaned_Headline']), svm_preds_train)
            plot_tree(svm_surr, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10, ax=axes[2])
            axes[2].set_title("3. Linear SVM — Surrogate Decision Tree", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)

with tab4:
    st.subheader("🔍 Live Financial Headline Sentiment Predictor")
    
    default_text = "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007."
    user_headline = st.text_area("Enter financial news headline to analyze:", value=default_text, height=100)
    
    if st.button("Predict Sentiment", type="primary"):
        if user_headline.strip() == "":
            st.warning("Please enter a valid news headline.")
        else:
            # Preprocess
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'up', 'down', 'above', 'below', 'under', 'over', 'more', 'less'}
            
            cleaned = user_headline.lower()
            cleaned = re.sub(r'http\S+|www\S+', '', cleaned)
            cleaned = re.sub(r'[^a-z\s]', '', cleaned)
            tokens = [lemmatizer.lemmatize(w) for w in cleaned.split() if w not in stop_words]
            cleaned_text = ' '.join(tokens)
            
            vec = tfidf.transform([cleaned_text])
            
            st.markdown("### Prediction Results:")
            res_cols = st.columns(len(trained_models))
            
            for idx, (name, model) in enumerate(trained_models.items()):
                pred = model.predict(vec)[0].upper()
                with res_cols[idx]:
                    st.markdown(f"**{name}**")
                    if pred == "POSITIVE":
                        st.success(f"🟢 {pred}")
                    elif pred == "NEGATIVE":
                        st.error(f"🔴 {pred}")
                    else:
                        st.warning(f"🟡 {pred}")
