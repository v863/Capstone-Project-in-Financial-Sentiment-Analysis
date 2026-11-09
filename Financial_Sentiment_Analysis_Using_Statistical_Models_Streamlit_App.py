import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import kagglehub

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Financial News Sentiment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI, metric cards, and badge styling
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        margin-top: 5px;
    }
    .badge-positive {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-negative {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. NLTK SETUP & DATA LOADING CACHE
# ==========================================
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_resources()

@st.cache_data
def load_default_dataset():
    try:
        path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
        csv_path = os.path.join(path, "all-data.csv")
        df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=['Sentiment', 'News Headline'])
        return df
    except Exception as e:
        # Fallback sample dataset if kagglehub network issue occurs
        data = {
            'Sentiment': ['neutral', 'neutral', 'negative', 'positive', 'positive'],
            'News Headline': [
                'According to Gran, the company has no plans to move production.',
                'Technopolis plans to develop in stages an area of over 100,000 square meters.',
                'The international electronic industry company ELCOTEQ recorded a negative quarterly result.',
                'With the new production plant the company would increase its capacity by 50 percent.',
                'According to the company updated strategy for the coming years, net sales will grow.'
            ]
        }
        return pd.DataFrame(data)

# ==========================================
# 3. TEXT PREPROCESSING PIPELINE
# ==========================================
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

# ==========================================
# 4. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
st.sidebar.title("🛠️ App Configuration")
st.sidebar.markdown("---")

# Dataset Source Selection
data_source = st.sidebar.radio("📁 Select Dataset Source", ["Default Financial News Dataset", "Upload Custom CSV"])

df = None
if data_source == "Default Financial News Dataset":
    with st.spinner("Loading financial news dataset..."):
        df = load_default_dataset()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain sentiment and headline columns)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            # Standardize column names if needed
            if len(df.columns) >= 2:
                df.columns = ['Sentiment', 'News Headline'] + list(df.columns[2:])
        except Exception as ex:
            st.sidebar.error(f"Error loading file: {ex}")
    else:
        st.info("Please upload a CSV file or switch to the default dataset in the sidebar.")
        st.stop()

# Clean Data
df = df.dropna().drop_duplicates().reset_index(drop=True)
df['Sentiment'] = df['Sentiment'].str.strip().str.lower()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Hyperparameters")
max_features = st.sidebar.slider("TF-IDF Max Features", min_value=1000, max_value=10000, value=5000, step=500)
ngram_max = st.sidebar.selectbox("TF-IDF N-gram Range", [1, 2], index=1, format_func=lambda x: f"(1, {x})")

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox("🤖 Primary Inference Model", ["Logistic Regression", "Support Vector Machine (SVM)", "Random Forest"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 About")
st.sidebar.markdown("Production-grade NLP pipeline built with Streamlit, Scikit-Learn, and NLTK for high-performance financial sentiment analysis.")

# ==========================================
# 5. MODEL TRAINING & PIPELINE CACHING
# ==========================================
@st.cache_resource
def train_models(df_cache, max_feat, ngram_val):
    df_cache['Cleaned_Headline'] = df_cache['News Headline'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_cache['Cleaned_Headline'],
        df_cache['Sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df_cache['Sentiment']
    )
    
    tfidf = TfidfVectorizer(ngram_range=(1, ngram_val), max_features=max_feat)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    results = []
    
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append({"Model": name, "Accuracy": acc, "Macro F1-Score": f1})
        
    return trained_models, tfidf, X_test_tfidf, y_test, results, X_train_tfidf, y_train

trained_models, tfidf, X_test_tfidf, y_test, results, X_train_tfidf, y_train = train_models(df, max_features, ngram_max)
results_df = pd.DataFrame(results)

# ==========================================
# 6. MAIN DASHBOARD TABS
# ==========================================
st.title("📊 Financial News Sentiment Intelligence Dashboard")
st.markdown("Analyze market sentiment, inspect model performance metrics, visualize feature importances, and run real-time inference.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Data Analysis (EDA)", 
    "🤖 Model Training & Evaluation", 
    "📈 Feature Importance & Decision Trees", 
    "🔮 Live Sentiment Predictor"
])

# ------------------------------------------
# TAB 1: EDA
# ------------------------------------------
with tab1:
    st.subheader("Dataset Overview & Sample Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">Total Headlines</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['Sentiment'].nunique()}</div>
            <div class="metric-label">Sentiment Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Features / Columns</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Class Distribution")
    col_chart, col_table = st.columns([1.2, 1])
    
    with col_chart:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df, x='Sentiment', palette='viridis', order=['neutral', 'positive', 'negative'], ax=ax)
        ax.set_title("Sentiment Class Frequencies")
        ax.set_xlabel("Sentiment Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
    with col_table:
        st.markdown("#### Sample Headlines")
        st.dataframe(df[['Sentiment', 'News Headline']].head(8), use_container_width=True)

# ------------------------------------------
# TAB 2: Model Training & Evaluation
# ------------------------------------------
with tab2:
    st.subheader("Comparative Performance Metrics")
    st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Macro F1-Score'], color='#dcfce7'), use_container_width=True)
    
    st.markdown("### Performance Comparison Bar Chart")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(results_df['Model']))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, results_df['Macro F1-Score'], width, label='Macro F1-Score', color='#38a3a5')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.set_ylim(0, 1.0)
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("### Confusion Matrices")
    labels = ['negative', 'neutral', 'positive']
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 4.5))
    
    for ax, (name, model) in zip(axes_cm, trained_models.items()):
        preds = model.predict(X_test_tfidf)
        cm = confusion_matrix(y_test, preds, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    st.pyplot(fig_cm)

# ------------------------------------------
# TAB 3: Feature Importance & Decision Trees
# ------------------------------------------
with tab3:
    st.subheader("Top Predictor Words & Surrogate Decision Trees")
    
    feature_names = tfidf.get_feature_names_out()
    
    st.markdown("### Top 15 Feature Weights / Coefficients")
    fig_feat, axes_feat = plt.subplots(1, 3, figsize=(18, 5))
    
    # Logistic Regression
    lr_coef = trained_models["Logistic Regression"].coef_[2] - trained_models["Logistic Regression"].coef_[0]
    top_lr_idx = np.argsort(np.abs(lr_coef))[-15:]
    axes_feat[0].barh(feature_names[top_lr_idx], lr_coef[top_lr_idx], color='#2b5c8f')
    axes_feat[0].set_title("Logistic Regression Top Words")
    
    # SVM
    svm_coef = trained_models["Support Vector Machine (SVM)"].coef_[2].toarray().flatten() - trained_models["Support Vector Machine (SVM)"].coef_[0].toarray().flatten()
    top_svm_idx = np.argsort(np.abs(svm_coef))[-15:]
    axes_feat[1].barh(feature_names[top_svm_idx], svm_coef[top_svm_idx], color='#38a3a5')
    axes_feat[1].set_title("Linear SVM Top Words")
    
    # Random Forest
    rf_importances = trained_models["Random Forest"].feature_importances_
    top_rf_idx = np.argsort(rf_importances)[-15:]
    axes_feat[2].barh(feature_names[top_rf_idx], rf_importances[top_rf_idx], color='#57cc99')
    axes_feat[2].set_title("Random Forest Top Split Words")
    
    st.pyplot(fig_feat)
    
    st.markdown("### Decision Tree & Surrogate Rules (Max Depth = 3)")
    tree_choice = st.selectbox("Select Model Tree Visualization", ["Random Forest (Actual Tree #0)", "Logistic Regression (Surrogate Tree)", "Support Vector Machine (Surrogate Tree)"])
    
    fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
    if "Random Forest" in tree_choice:
        tree_obj = trained_models["Random Forest"].estimators_[0]
    elif "Logistic Regression" in tree_choice:
        lr_preds = trained_models["Logistic Regression"].predict(X_train_tfidf)
        tree_obj = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train_tfidf, lr_preds)
    else:
        svm_preds = trained_models["Support Vector Machine (SVM)"].predict(X_train_tfidf)
        tree_obj = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train_tfidf, svm_preds)
        
    plot_tree(tree_obj, max_depth=3, feature_names=feature_names, class_names=['negative', 'neutral', 'positive'], filled=True, rounded=True, fontsize=10, ax=ax_tree)
    st.pyplot(fig_tree)

# ------------------------------------------
# TAB 4: Live Sentiment Predictor
# ------------------------------------------
with tab4:
    st.subheader("🔮 Real-Time Financial Headline Sentiment Analyzer")
    st.markdown("Type or paste any financial news headline below to instantly evaluate its market sentiment across all models.")
    
    default_text = "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007."
    user_headline = st.text_area("Financial News Headline:", value=default_text, height=100)
    
    if st.button("Evaluate Sentiment", type="primary"):
        if not user_headline.strip():
            st.warning("Please enter a valid headline.")
        else:
            cleaned_input = preprocess_text(user_headline)
            input_vector = tfidf.transform([cleaned_input])
            
            st.markdown("### Prediction Results")
            cols = st.columns(3)
            
            for idx, (name, model) in enumerate(trained_models.items()):
                pred = model.predict(input_vector)[0].upper()
                
                # Retrieve probability if available
                prob_str = ""
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_vector)[0]
                    max_prob = np.max(probs) * 100
                    prob_str = f" ({max_prob:.1f}% Confidence)"
                
                badge_class = "badge-neutral"
                if pred == "POSITIVE":
                    badge_class = "badge-positive"
                elif pred == "NEGATIVE":
                    badge_class = "badge-negative"
                    
                with cols[idx]:
                    st.markdown(f"**{name}**")
                    st.markdown(f'<div class="{badge_class}">{pred}{prob_str}</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
            st.markdown("---")
            st.markdown("#### Preprocessed Token Representation:")
            st.code(cleaned_input if cleaned_input else "(All tokens removed or filtered out)", language="text")
