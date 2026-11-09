import os
import kagglehub
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Sentiment Analyzer & Gemini Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Phrasebank Sentiment Analysis with Google GenAI & KaggleHub")
st.markdown(
    "This application loads the Financial Phrasebank dataset via **kagglehub**, "
    "allows you to explore its distribution, and leverages **Google GenAI** (`google-genai`) "
    "for zero-shot and few-shot financial text sentiment classification."
)

# ------------------------------------------------------------------
# Sidebar Configuration & API Key Setup
# ------------------------------------------------------------------
st.sidebar.header("Configuration")

# API Key input (checks environment variable first, then fallback to sidebar text input)
env_api_key = os.environ.get("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=env_api_key,
    type="password",
    help="Enter your Google Gemini API key or set it as an environment variable."
)

if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input

selected_model = st.sidebar.selectbox(
    "Select Gemini Model",
    ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-lite-latest"],
    index=0
)

# ------------------------------------------------------------------
# Dataset Loading via KaggleHub
# ------------------------------------------------------------------
@st.cache_data
def load_financial_phrasebank():
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    
    # Locate the CSV file in the downloaded directory
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                break
        if csv_file:
            break
            
    if not csv_file:
        raise FileNotFoundError("Could not locate the CSV file in the downloaded kagglehub directory.")
        
    df = pd.read_csv(csv_file, encoding="latin-1", header=None, names=["sentiment", "text"])
    
    # Clean and normalize
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df["text"] = df["text"].str.strip()
    df = df.dropna().drop_duplicates(subset=["sentiment", "text"]).reset_index(drop=True)
    return df

with st.spinner("Downloading and loading dataset via kagglehub..."):
    try:
        df = load_financial_phrasebank()
        st.sidebar.success(f"Dataset loaded successfully! Total records: {len(df)}")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

# ------------------------------------------------------------------
# Main UI Layout (Tabs)
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Live Classifier", "📊 Dataset Explorer", "⚙️ Few-Shot Evaluation"])

with tab1:
    st.header("Interactive Financial Sentiment Classifier")
    st.markdown("Test the Gemini model on custom financial sentences.")
    
    user_input = st.text_area(
        "Enter financial news sentence:",
        value="Operating profit margins improved significantly due to lower operational costs and strong quarterly revenue growth."
    )
    
    classification_mode = st.radio("Classification Mode", ["Zero-Shot", "Few-Shot (with Examples)"])
    
    if st.button("Classify Sentiment", type="primary"):
        if not api_key_input:
            st.error("Please provide a valid Gemini API Key in the sidebar.")
        else:
            try:
                client = genai.Client(api_key=api_key_input)
                
                if classification_mode == "Zero-Shot":
                    prompt = (
                        "You are an expert financial sentiment classification system.\n"
                        "Classify the following financial sentence into exactly one category: positive, negative, or neutral.\n"
                        "Return ONLY the lowercase label.\n\n"
                        f"Sentence: {user_input}"
                    )
                else:
                    # Construct few-shot prompt using samples from the dataset
                    few_shot_samples = df.groupby("sentiment", group_keys=False).sample(n=2, random_state=42)
                    examples_str = ""
                    for _, row in few_shot_samples.iterrows():
                        examples_str += f"Sentence: {row['text']}\nSentiment: {row['sentiment']}\n\n"
                        
                    prompt = (
                        "You are an expert financial sentiment classification system.\n"
                        "Classify each financial news sentence into exactly one category: positive, negative, or neutral.\n\n"
                        f"Here are some examples:\n{examples_str}"
                        f"Now classify this sentence:\nSentence: {user_input}\nSentiment:"
                    )
                
                with st.spinner("Analyzing sentiment with Gemini..."):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0,
                            max_output_tokens=20
                        )
                    )
                    
                prediction = response.text.strip().lower()
                
                st.subheader("Result")
                if "positive" in prediction:
                    st.success(f"**Predicted Sentiment:** Positive 🟢")
                elif "negative" in prediction:
                    st.error(f"**Predicted Sentiment:** Negative 🔴")
                else:
                    st.info(f"**Predicted Sentiment:** Neutral 🔵")
                    
                st.caption(f"Raw model response: `{response.text.strip()}`")
                
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

with tab2:
    st.header("Dataset Overview & Visualizations")
    st.write("Preview of the cleaned Financial PhraseBank dataset:")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        st.bar_chart(sentiment_counts)
        
    with col2:
        st.subheader("Sentence Word Count Statistics")
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))
        st.bar_chart(df.groupby("sentiment")["word_count"].mean())

with tab3:
    st.header("Batch Evaluation & Few-Shot Testing")
    st.markdown("Run a quick sample batch test against a subset of validation/test data.")
    
    sample_size = st.slider("Select sample size for test evaluation", min_value=5, max_value=50, value=10)
    
    if st.button("Run Batch Evaluation"):
        if not api_key_input:
            st.error("Please provide a valid Gemini API Key in the sidebar.")
        else:
            client = genai.Client(api_key=api_key_input)
            test_subset = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in test_subset.iterrows():
                prompt = (
                    "Classify this financial sentence as positive, negative, or neutral. "
                    f"Return only the lowercase label.\nSentence: {row['text']}"
                )
                try:
                    resp = client.models.generate_content(
                        model=selected_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0, max_output_tokens=10)
                    )
                    pred = resp.text.strip().lower()
                except Exception:
                    pred = "error"
                    
                results.append({
                    "True Label": row["sentiment"],
                    "Predicted": pred,
                    "Text": row["text"]
                })
                progress_bar.progress((idx + 1) / sample_size)
                
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            correct = (res_df["True Label"] == res_df["Predicted"]).sum()
            st.metric(label="Sample Accuracy", value=f"{(correct / sample_size) * 100:.2f}%")
