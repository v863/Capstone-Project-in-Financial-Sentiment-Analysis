import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from groq import Groq, RateLimitError
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize Groq Client & FinBERT Pipeline
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

finbert_pipeline = load_finbert()

# Sidebar Configuration
st.sidebar.title("🛠️ Configuration & Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))

model_choice = st.sidebar.selectbox(
    "Choose Groq Model",
    [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "ProsusAI/finbert (Fine-Tuned Local)"
    ]
)

prompt_paradigm = st.sidebar.radio("Select Evaluation Paradigm", ["Zero-Shot", "Few-Shot"])

st.sidebar.markdown("---")
st.sidebar.info(
    "This app compares open-weight LLMs via Groq API and FinBERT "
    "on financial phrasebank sentiment analysis."
)

# Helper function for Groq API call with retry mechanism
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

# Main Application Layout
st.title("📊 Financial Sentiment Analysis: Multi-Model Comparison Dashboard")
st.markdown("""
This dashboard allows you to run real-time sentiment classification on financial text 
using **Groq-hosted LLMs** (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`) 
and domain-specific **FinBERT**[cite: 1].
""")

# Input section
st.subheader("✍️ Test Financial Headline")
user_input = st.text_area(
    "Enter a news headline or financial report excerpt:",
    "The company reported a substantial increase in quarterly operating profits, beating analyst expectations."
)

if st.button("Run Sentiment Analysis", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            if "finbert" in model_choice.lower():
                # FinBERT Inference
                start_time = time.time()
                res = finbert_pipeline(user_input[:512])[0]
                elapsed = time.time() - start_time
                
                pred_label = res['label'].strip().lower()
                confidence = res['score']
                
                st.success(f"Analysis Complete (in {elapsed:.2f}s)")
                col1, col2 = st.columns(2)
                col1.metric("Predicted Sentiment", pred_label.upper())
                col2.metric("Confidence Score", f"{confidence:.4f}")
                
            else:
                # Groq LLM Inference
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
                            # Few-Shot setup
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
                        
                        st.success(f"Analysis Complete (in {elapsed:.2f}s using {model_choice})")
                        st.metric("Predicted Sentiment", pred_label.upper())
                        
                    except Exception as e:
                        st.error(f"An error occurred during API execution: {e}")

st.markdown("---")

# Benchmark Summary Section
st.subheader("📋 Benchmark Summary Reference Table")
st.markdown("Performance results evaluated on the **Financial PhraseBank** benchmark dataset[cite: 1]:")

summary_data = [
    {"Model": "openai/gpt-oss-120b", "Configuration": "Zero-shot", "Accuracy": 0.75, "Precision": 0.79, "Recall": 0.75, "F1-score": 0.75},
    {"Model": "openai/gpt-oss-120b", "Configuration": "Few-shot", "Accuracy": 0.80, "Precision": 0.84, "Recall": 0.80, "F1-score": 0.80},
    {"Model": "qwen/qwen3.8-27b", "Configuration": "Zero-shot", "Accuracy": 0.82, "Precision": 0.85, "Recall": 0.82, "F1-score": 0.83},
    {"Model": "qwen/qwen3.8-27b", "Configuration": "Few-shot", "Accuracy": 0.82, "Precision": 0.87, "Recall": 0.82, "F1-score": 0.83},
    {"Model": "openai/gpt-oss-20b", "Configuration": "Zero-shot", "Accuracy": 0.78, "Precision": 0.78, "Recall": 0.78, "F1-score": 0.77},
    {"Model": "openai/gpt-oss-20b", "Configuration": "Few-shot", "Accuracy": 0.80, "Precision": 0.84, "Recall": 0.80, "F1-score": 0.80},
    {"Model": "ProsusAI/finbert", "Configuration": "Fine-Tuned", "Accuracy": 0.85, "Precision": 0.88, "Recall": 0.85, "F1-score": 0.85},
]

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)
