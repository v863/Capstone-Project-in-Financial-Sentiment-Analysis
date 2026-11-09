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
from sklearn.tree import DecisionTreeClassifier, plot_tree

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b5c8f !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & PREPROCESSING (CACHED)
# ==========================================
@st.cache_resource(show_spinner=False)
def download_and_load_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
        csv_path = os.path.join(path, "all-data.csv")
        
        df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
        df = df.drop_duplicates().reset_index(drop=True)
        df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
        return df
    except Exception as e:
        # Fallback synthetic dataset if Kaggle download fails offline
        data = {
            'Sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral'],
            'News Headline': [
                "Company reports record quarterly profit surge of 45 percent.",
                "Shares plunge following disappointing revenue guidance and heavy losses.",
                "The annual general meeting took place in Helsinki on Tuesday.",
                "New partnership agreement expected to expand international market reach.",
                "Regulatory investigation launched into accounting practices.",
                "According to the firm, operations remain stable throughout the quarter."
            ]
        }
        return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def preprocess_and_train(_df):
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

    cleaned_headlines = _df['News Headline'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_headlines,
        _df['Sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=_df['Sentiment']
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

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
    
    return tfidf, trained_models, results_df, X_test, y_test, preprocess_text

# Load data and models with progress spinner
with st.spinner("📥 Loading Financial News Dataset and Training Statistical Models... Please wait."):
    df = download_and_load_data()
    tfidf, trained_models, results_df, X_test, y_test, preprocess_text = preprocess_and_train(df)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📊 Financial Sentiment Hub")
st.sidebar.markdown("Statistical & Machine Learning models trained on financial news headlines.")

st.sidebar.info(
    "**Models Implemented:**\n"
    "- Logistic Regression\n"
    "- Support Vector Machine (SVM)\n"
    "- Random Forest Classifier\n\n"
    "**Features:** TF-IDF n-grams (1,2) with custom financial stop-word retention."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
selected_tab = st.sidebar.radio(
    "Go to:",
    ["Dataset Explorer", "Model Performance & Confusion Matrices", "Decision Tree Visualizations", "Interactive Prediction"]
)

# ==========================================
# MAIN TITLE
# ==========================================
st.title("📈 Financial Sentiment Analysis Using Statistical Models")
st.markdown("Analyze financial news sentiment using robust TF-IDF feature extraction and supervised classification algorithms.")
st.markdown("---")

# ==========================================
# TAB 1: DATASET EXPLORER
# ==========================================
if selected_tab == "Dataset Explorer":
    st.header("📂 Dataset Explorer & Distribution")
    st.markdown("Explore the raw financial news headlines and the class distribution of sentiments (`positive`, `neutral`, `negative`).")
    
    col1, col2, col3 = st.st.columns(3) if hasattr(st, "columns") else st.columns(3) # safe check
    with col1:
        st.metric("Total Clean Records", f"{len(df):,}")
    with col2:
        st.metric("Classes", "Positive, Neutral, Negative")
    with col3:
        st.metric("Train-Test Split", "80% / 20% Stratified")

    st.subheader("🔍 Sample Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📊 Sentiment Distribution Chart")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(data=df, x='Sentiment', palette='viridis', order=['neutral', 'positive', 'negative'], ax=ax)
    ax.set_title("Sentiment Class Distribution in Financial News", fontsize=12, fontweight='bold')
    ax.set_xlabel("Sentiment Category", fontsize=10)
    ax.set_ylabel("Headline Count", fontsize=10)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# TAB 2: MODEL PERFORMANCE & CONFUSION MATRICES
# ==========================================
elif selected_tab == "Model Performance & Confusion Matrices":
    st.header("🏆 Model Performance & Evaluation")
    st.markdown("Compare the Accuracy and Macro F1-Scores across all three classifiers, alongside detailed confusion matrices.")

    st.subheader("📊 Performance Metrics Comparison")
    st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Macro F1-Score'], color='#d4edda'), use_container_width=True)

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(results_df['Model']))
    width = 0.35

    rects1 = ax.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, results_df['Macro F1-Score'], width, label='Macro F1-Score', color='#38a3a5')

    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Model Performance: Logistic Regression vs. SVM vs. Random Forest', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📉 Test Set Confusion Matrices")
    
    # Pre-transform test data
    X_test_cleaned = X_test.apply(preprocess_text)
    X_test_tfidf = tfidf.transform(X_test_cleaned)
    labels = ['negative', 'neutral', 'positive']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, model) in zip(axes, trained_models.items()):
        preds = model.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, preds, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set_title(f"{name} CM", fontsize=11, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("Actual Label", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# TAB 3: DECISION TREE VISUALIZATIONS
# ==========================================
elif selected_tab == "Decision Tree Visualizations":
    st.header("🌳 Decision Tree & Surrogate Model Visualizations")
    st.markdown("""
    Explore structural tree representations:
    1. **Random Forest:** An actual internal Decision Tree estimator extracted from the forest.
    2. **Logistic Regression & SVM:** **Surrogate Decision Trees** trained to mimic and explain the decision boundaries of linear models (Explainable AI technique).
    """)

    feature_names = tfidf.get_feature_names_out()
    class_names = ['negative', 'neutral', 'positive']

    # Generate trees
    X_train_cleaned = df['News Headline'].apply(preprocess_text) # simplified for visualization demo
    # We use sample transform for speed
    X_train_sub, _, y_train_sub, _ = train_test_split(df['News Headline'].apply(preprocess_text), df['Sentiment'], train_size=min(1000, len(df)), random_state=42)
    X_train_sub_tfidf = tfidf.transform(X_train_sub)

    rf_single_tree = trained_models["Random Forest"].estimators_[0]
    
    lr_preds = trained_models["Logistic Regression"].predict(X_train_sub_tfidf)
    lr_surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
    lr_surrogate.fit(X_train_sub_tfidf, lr_preds)

    svm_preds = trained_models["SVM"].predict(X_train_sub_tfidf)
    svm_surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
    svm_surrogate.fit(X_train_sub_tfidf, svm_preds)

    model_choice = st.selectbox(
        "Select Model Tree to Inspect:",
        ["Random Forest (Actual Tree)", "Logistic Regression (Surrogate Tree)", "SVM (Surrogate Tree)"]
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    if "Random Forest" in model_choice:
        plot_tree(rf_single_tree, max_depth=3, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10, ax=ax)
        ax.set_title("Random Forest — Internal Decision Tree Estimator (#0, Max Depth 3)", fontsize=13, fontweight='bold')
    elif "Logistic Regression" in model_choice:
        plot_tree(lr_surrogate, max_depth=3, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10, ax=ax)
        ax.set_title("Logistic Regression — Surrogate Decision Tree (Explaining Linear Weights)", fontsize=13, fontweight='bold')
    else:
        plot_tree(svm_surrogate, max_depth=3, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10, ax=ax)
        ax.set_title("SVM — Surrogate Decision Tree (Explaining Hyperplane Boundaries)", fontsize=13, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# TAB 4: INTERACTIVE PREDICTION
# ==========================================
elif selected_tab == "Interactive Prediction":
    st.header("⚡ Live Financial Sentiment Predictor")
    st.markdown("Type or select a financial headline below to test how each statistical model classifies its market sentiment.")

    preset_headlines = [
        "Select a preset or type your own...",
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period.",
        "The company announced it will lay off 500 employees following a severe drop in Q3 revenue.",
        "The board of directors approved the annual report during Tuesday's meeting.",
        "Net sales increased by 12% driven by robust demand in European markets."
    ]

    choice = st.selectbox("Quick Presets:", preset_headlines)
    
    default_text = "" if choice == preset_headlines[0] else choice
    user_input = st.text_area("Financial Headline / News Text:", value=default_text, height=100)

    if st.button("🔮 Predict Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Please enter or select a valid financial headline.")
        else:
            cleaned_input = preprocess_text(user_input)
            vec = tfidf.transform([cleaned_input])

            st.subheader("🎯 Prediction Results")
            cols = st.columns(3)

            for i, (name, model) in enumerate(trained_models.items()):
                pred = model.predict(vec)[0].upper()
                
                # Color badges
                if pred == "POSITIVE":
                    badge_color = "#28a745"
                elif pred == "NEGATIVE":
                    badge_color = "#dc3545"
                else:
                    badge_color = "#ffc107"

                with cols[i]:
                    st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
                            <h4 style="color: #2b5c8f; margin-bottom: 10px;">{name}</h4>
                            <span style="background-color: {badge_color}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px;">
                                {pred}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📝 Preprocessed Features (TF-IDF Tokens)")
            st.code(cleaned_input if cleaned_input else "(No informative tokens remaining after stop-word removal)", language="text")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Financial Sentiment Analysis App • Built with Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)
