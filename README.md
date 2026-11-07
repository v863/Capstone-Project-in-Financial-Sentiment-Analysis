# Financial Sentiment Analysis: Multi-Model Comparison (Groq LLMs)

## 📌 Project Overview
This project provides an extended, multi-model evaluation pipeline for financial sentiment analysis using the benchmark **Financial PhraseBank** dataset. It systematically compares three distinct model configurations accessed via the Groq API alongside a fine-tuned domain-specific transformer:

1. **`openai/gpt-oss-120b`**: High-capacity open-weight baseline model evaluated via Zero-Shot and Few-Shot prompting.
2. **`qwen/qwen3.8-27b`**: Mid-scale efficient transformer architecture evaluated via Zero-Shot and Few-Shot prompting.
3. **`openai/gpt-oss-20b`**: Lightweight efficient parameter model evaluated via Zero-Shot and Few-Shot prompting.
4. **`ProsusAI/finbert`**: Domain-specific pre-trained and fine-tuned financial transformer model as a baseline.

---

## 🛠️ Repository Structure & Notebook Cells
The pipeline is designed to run sequentially (typically in a Jupyter/Google Colab environment):
- **Cell 0:** Environment Setup & Dependency Installation (`groq`, `scikit-learn`, `seaborn`, `kagglehub`, etc.).
- **Cell 1:** Import libraries, configure warnings, and initialize the Groq client.
- **Cell 2:** Download and load the Financial PhraseBank dataset from Kaggle Hub, clean duplicates, and normalize sentiment labels.
- **Cell 3 & 4:** Perform stratified train/test split ($80\%$ train, $20\%$ test) and visualize class distributions.
- **Cell 5 & 6:** Initialize evaluation harness and execute Zero-Shot and Few-Shot sentiment classification across the Groq LLM suite.
- **Cell 7:** Evaluate the domain-specific `ProsusAI/finbert` model.
- **Cell 8 & 9:** Generate comparative performance summary tables, export predictions to CSV, and plot confusion matrix heatmaps.
- **Cell 10:** Comprehensive conclusions and limitations discussion.

---

## 📊 Performance Summary
Below is a summary of the comparative metrics achieved across models on the test split:

| Model Configuration | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| **openai/gpt-oss-120b** (Zero-shot) | $0.75$ | $0.79$ | $0.75$ | $0.75$ |
| **openai/gpt-oss-120b** (Few-shot) | $0.80$ | $0.84$ | $0.80$ | $0.80$ |
| **qwen/qwen3.8-27b** (Zero-shot) | $0.82$ | $0.85$ | $0.82$ | $0.83$ |
| **qwen/qwen3.8-27b** (Few-shot) | $0.82$ | $0.87$ | $0.82$ | $0.83$ |
| **openai/gpt-oss-20b** (Zero-shot) | $0.78$ | $0.78$ | $0.78$ | $0.77$ |
| **openai/gpt-oss-20b** (Few-shot) | $0.80$ | $0.84$ | $0.80$ | $0.80$ |
| **ProsusAI/finbert** (Fine-Tuned) | $0.85$ | $0.88$ | $0.85$ | $0.85$ |

---

## 🚀 Getting Started

### Prerequisites
Make sure you have your Groq API key set up in your environment or Google Colab user data:
```python
from google.colab import userdata
groq_api_key = userdata.get('GROQ_API_KEY')
```

### Installation
Install required packages using pip:
```bash
pip install groq scikit-learn pandas numpy matplotlib seaborn kagglehub transformers torch
```

### Output Files
- `multimodel_and_finbert_sentiment_predictions.csv`: Contains the ground truth labels, news headlines, and model predictions for all tested configurations.
