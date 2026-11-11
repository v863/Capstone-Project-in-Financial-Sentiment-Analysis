import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# App Header
st.title("📈 Financial Sentiment Analysis using Deep Learning")
st.markdown("""
This application compares domain-specific Transformer models (**FinBERT**) against traditional deep learning architectures (**BERT / Bi-LSTM / NN**) for analyzing financial news, earnings reports, and market sentiment.
""")

# Sidebar Configuration
st.sidebar.header("Model Configuration")
selected_model = st.sidebar.selectbox(
    "Choose Deep Learning Model",
    ("FinBERT (Fine-tuned)", "BERT / Bi-LSTM / NN (Train split tuned)")
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Performance Summary:**\n"
    "- **FinBERT:** 0.86 F1-Score\n"
    "- **BERT/Bi-LSTM/NN:** 0.82 F1-Score"
)

# Main Dashboard Layout
tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📊 Model Comparison Matrix", "ℹ️ About Models"])

with tab1:
    st.subheader("Test Financial Text Sentiment")
    
    # Text input area
    sample_text = st.text_area(
        "Enter financial news headline or corporate statement:",
        "Net income rose 35% year-over-year driven by strong cloud infrastructure adoption, beating consensus estimates."
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if not sample_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner(f"Running inference using {selected_model}..."):
                # Simulated prediction outputs for demonstration purposes
                if "FinBERT" in selected_model:
                    sentiment = "Positive"
                    confidence = 0.94
                    scores = {"Positive": 0.94, "Neutral": 0.04, "Negative": 0.02}
                else:
                    sentiment = "Positive"
                    confidence = 0.82
                    scores = {"Positive": 0.82, "Neutral": 0.12, "Negative": 0.06}
            
            # Display Results
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Sentiment", value=sentiment)
            with col2:
                st.metric(label="Model Confidence", value=f"{confidence*100:.1f}%")
                
            st.markdown("### Class Probabilities")
            st.bar_chart(pd.DataFrame(list(scores.values()), index=list(scores.keys()), columns=["Probability"]))

with tab2:
    st.subheader("Benchmark Evaluation Metrics")
    st.markdown("Comparison across standard financial benchmark datasets (e.g., Financial PhraseBank):")
    
    metrics_data = {
        "Model Architecture": ["FinBERT", "BERT / Bi-LSTM / NN"],
        "Category": ["Deep Learning", "Deep Learning"],
        "Training Strategy": ["Fine-tuned", "Train split tuned"],
        "Precision": [0.86, 0.82],
        "Recall": [0.86, 0.81],
        "F1-Score": [0.86, 0.82]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)

with tab3:
    st.subheader("Architecture Details")
    st.markdown("""
    - **FinBERT**: A specialized language model based on BERT, pre-trained on a massive corpus of financial text (earnings calls, analyst reports, financial news). Fine-tuning on domain data captures subtle market contexts effectively.
    - **BERT / Bi-LSTM / NN**: Combines contextual embeddings with recurrent layers or standard dense neural networks to capture sequential text dependencies within financial disclosures.
    """)
