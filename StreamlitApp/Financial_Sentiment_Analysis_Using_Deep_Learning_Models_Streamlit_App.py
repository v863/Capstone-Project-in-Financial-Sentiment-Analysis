import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, f1_score, accuracy_score
import re
from collections import Counter, defaultdict
import random
import io
import csv
from pathlib import Path

st.set_page_config(
    page_title="Financial Sentiment Analysis (Dynamic)",
    page_icon="📈",
    layout="wide"
)

# --- STYLES ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; font-weight: 700; }
    .sub-header { font-size: 1.2rem; color: #555555; }
</style>
""", unsafe_allow_html=True)

# --- REPRODUCIBILITY & DEVICE ---
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")

def tokenize(text, normalize_numbers=False):
    tokens = TOKEN_PATTERN.findall(text.casefold())
    if normalize_numbers:
        tokens = ["<num>" if any(ch.isdigit() for ch in tok) else tok for tok in tokens]
    return tokens

# --- MODEL DEFINITION ---
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128,
                 dropout=0.3, embedding_dropout=0.0, pooling="last"):
        super().__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        features = (4 if pooling == "max_mean" else 2) * hidden_dim
        self.classifier = nn.Linear(features, num_classes)

    def forward(self, token_ids, lengths):
        embedded = self.embedding_dropout(self.embedding(token_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, _) = self.encoder(packed)
        if self.pooling == "last":
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            lengths = lengths.to(outputs.device)
            mask = (torch.arange(outputs.size(1), device=outputs.device)[None, :] < lengths[:, None]).unsqueeze(2)
            maximum = outputs.masked_fill(~mask, float("-inf")).amax(dim=1)
            mean = outputs.masked_fill(~mask, 0).sum(dim=1) / lengths[:, None]
            features = torch.cat((maximum, mean), dim=1)
        return self.classifier(self.dropout(features))

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_preprocess_data():
    DATA_FILE = "all-data.csv"
    SEARCH_PATHS = [Path(DATA_FILE), Path("dataset") / DATA_FILE, Path("/content") / DATA_FILE]
    data_path = next((path for path in SEARCH_PATHS if path.exists()), None)
    
    if data_path is None:
        # Fallback mock data if file doesn't exist yet
        data = [
            ("Operating profit rose to EUR 13.1 mn from EUR 8.7 mn .", "positive"),
            ("The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility .", "negative"),
            ("According to Gran , the company has no plans to move all production to Russia .", "neutral"),
            ("Sales increased by 15% during the third quarter .", "positive"),
            ("The company reported a net loss of 2.3 million euros .", "negative"),
            ("Trading remained stable throughout the morning session .", "neutral"),
            ("Operating loss narrowed down significantly over the fiscal year .", "positive"),
            ("Revenues decreased less than expected by financial analysts .", "negative")
        ] * 100 # Multiply to have enough rows for batching
        df_raw = pd.DataFrame(data, columns=["text", "label"])
    else:
        raw = Path(data_path).read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        records = list(csv.reader(io.StringIO(text, newline="")))
        if records and records[0][0].strip().lower() in ("label", "sentiment"):
            records = records[1:]
        rows = [(r[1].strip(), r[0].strip().lower()) for r in records if len(r) == 2]
        df_raw = pd.DataFrame(rows, columns=["text", "label"])

    # Clean duplicates & conflicts
    keys = df_raw["text"].map(lambda t: tuple(tokenize(t)))
    labels_per_text = df_raw.groupby(keys)["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    clean = df_raw.loc[~keys.map(lambda k: k in conflicting)].copy()
    clean = clean.loc[~keys.duplicated(keep="first")]
    return clean

df = load_and_preprocess_data()
LABELS = sorted(df["label"].unique())
LABEL_IDS = {l: i for i, l in enumerate(LABELS)}
CLASS_COLORS = {"negative": "#d62728", "neutral": "#8c8c8c", "positive": "#2ca02c"}

# Stratified Split function
def stratified_split(rows, validation_fraction=0.15, test_fraction=0.15, seed=SEED):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for label, examples in sorted(by_label.items()):
        examples = examples[:]
        rng.shuffle(examples)
        n_val = max(1, int(len(examples) * validation_fraction))
        n_test = max(1, int(len(examples) * test_fraction))
        validation += examples[:n_val]
        test += examples[n_val:n_val + n_test]
        train += examples[n_val + n_test:]
    for split in (train, validation, test):
        rng.shuffle(split)
    return train, validation, test

rows = list(zip(df["text"], df["label"]))
train_rows, val_rows, test_rows = stratified_split(rows)
SPLITS = {"train": train_rows, "validation": val_rows, "test": test_rows}

# --- VOCAB & DATASET HELPERS ---
def build_vocab(train_rows, normalize_numbers=False, min_frequency=1, max_vocab=30000):
    counts = Counter(tok for text, _ in train_rows for tok in tokenize(text, normalize_numbers))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count < min_frequency or len(vocab) >= max_vocab:
            break
        vocab.setdefault(token, len(vocab))
    return vocab

def encode(text, vocab, max_length=128, normalize_numbers=False):
    ids = [vocab.get(tok, 1) for tok in tokenize(text, normalize_numbers)[:max_length]]
    return torch.tensor(ids or [1], dtype=torch.long)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, rows, vocab, max_length, normalize_numbers):
        self.examples = [(encode(text, vocab, max_length, normalize_numbers), LABEL_IDS[label]) for text, label in rows]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return self.examples[index]

def collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0), lengths, torch.tensor(labels)

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Go to:", [
    "📊 Dataset Overview & EDA",
    "🚀 Dynamic Model Training",
    "🔮 Live Model Inference",
    "📈 Results Evaluation"
])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dynamic Deep Learning App**\nTrain PyTorch BiLSTMs live inside Streamlit!")

# ================= PAGE 1: EDA =================
if page == "📊 Dataset Overview & EDA":
    st.markdown('<p class="main-header">Financial Sentiment Analysis Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Exploratory Data Analysis on cleaned financial text data.</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Clean Sentences", f"{len(df):,}")
    col2.metric("Train Split", f"{len(train_rows):,}")
    col3.metric("Test Split", f"{len(test_rows):,}")

    counts = df["label"].value_counts()
    df_counts = pd.DataFrame({"Count": counts, "Share (%)": (100 * counts / len(df)).round(1)})
    st.dataframe(df_counts, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    counts.plot.bar(ax=ax, color=[CLASS_COLORS[lbl] for lbl in counts.index], rot=0)
    ax.set_title("Class Distribution")
    st.pyplot(fig)

# ================= PAGE 2: TRAINING =================
elif page == "🚀 Dynamic Model Training":
    st.markdown('<p class="main-header">Train BiLSTM Model Live</p>', unsafe_allow_html=True)
    st.markdown("Configure hyperparameters and train a bidirectional LSTM network from scratch.")

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", 1, 15, 5)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        lr = st.selectbox("Learning Rate", [1e-4, 1e-3, 2e-3], index=1)
    with col2:
        normalize_numbers = st.checkbox("Normalize Numbers (<num>) ?", value=True)
        pooling_type = st.selectbox("Pooling Strategy", ["max_mean", "last"], index=0)
        weight_decay = st.selectbox("Weight Decay (AdamW)", [0.0, 0.01, 0.05], index=1)

    if st.button("🔥 Start Live Training", type="primary"):
        vocab = build_vocab(train_rows, normalize_numbers=normalize_numbers)
        train_loader = torch.utils.data.DataLoader(
            TextDataset(train_rows, vocab, 128, normalize_numbers),
            batch_size=batch_size, shuffle=True, collate_fn=collate
        )
        val_loader = torch.utils.data.DataLoader(
            TextDataset(val_rows, vocab, 128, normalize_numbers),
            batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        test_loader = torch.utils.data.DataLoader(
            TextDataset(test_rows, vocab, 128, normalize_numbers),
            batch_size=batch_size, shuffle=False, collate_fn=collate
        )

        model = SentimentModel(
            len(vocab), len(LABELS), embedding_dim=128, hidden_dim=128,
            dropout=0.3, embedding_dropout=0.3, pooling=pooling_type
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        @torch.inference_mode()
        def evaluate_model(loader):
            model.eval()
            loss, targets, preds = 0.0, [], []
            for tokens, lengths, labels in loader:
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                logits = model(tokens, lengths)
                loss += criterion(logits, labels).item() * len(labels)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
            return loss / len(targets), accuracy_score(targets, preds), f1_score(targets, preds, average="macro")

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for tokens, lengths, labels in train_loader:
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(tokens, lengths), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(labels)
            
            train_loss /= len(train_rows)
            val_loss, val_acc, val_f1 = evaluate_model(val_loader)
            
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1})
            progress_bar.progress(epoch / epochs)
            status_text.text(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.3f} | Val Macro-F1: {val_f1:.3f}")

        st.success("Training Completed Successfully!")
        
        # Save model and artifacts into session state for inference use
        st.session_state["trained_model"] = model
        st.session_state["vocab"] = vocab
        st.session_state["normalize_numbers"] = normalize_numbers
        st.session_state["pooling_type"] = pooling_type

        # Plot training curves
        df_hist = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", marker="o")
        ax.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_title("Live Training & Validation Loss Curves")
        ax.legend()
        st.pyplot(fig)

# ================= PAGE 3: INFERENCE =================
elif page == "🔮 Live Model Inference":
    st.markdown('<p class="main-header">Dynamic Model Inference</p>', unsafe_allow_html=True)
    
    if "trained_model" not in st.session_state:
        st.warning("⚠️ No trained model found in session state! Please go to **'🚀 Dynamic Model Training'** and click 'Start Live Training' first.")
    else:
        model = st.session_state["trained_model"]
        vocab = st.session_state["vocab"]
        normalize_numbers = st.session_state["normalize_numbers"]
        
        user_input = st.text_area("Enter a financial news sentence for inference:", "Operating profit increased significantly to 15 million euros .")
        
        if st.button("Predict Sentiment", type="primary"):
            model.eval()
            tokens = tokenize(user_input, normalize_numbers=normalize_numbers)
            tensor_seq = encode(user_input, vocab, max_length=128, normalize_numbers=normalize_numbers).unsqueeze(0).to(DEVICE)
            length = torch.tensor([len(tensor_seq[0])], dtype=torch.long)
            
            with torch.inference_mode():
                logits = model(tensor_seq, length)
                probs = logits.softmax(dim=1).cpu().numpy()[0]
                pred_idx = probs.argmax()
                pred_label = LABELS[pred_idx]
            
            st.success(f"**Predicted Sentiment:** `{pred_label.upper()}`")
            
            prob_df = pd.DataFrame({
                "Sentiment": [l.capitalize() for l in LABELS],
                "Probability": probs
            }).set_index("Sentiment")
            st.bar_chart(prob_df)

# ================= PAGE 4: RESULTS =================
elif page == "📈 Results Evaluation":
    st.markdown('<p class="main-header">Evaluation & Performance Metrics</p>', unsafe_allow_html=True)
    if "trained_model" not in st.session_state:
        st.info("Train a model first under **'🚀 Dynamic Model Training'** to view test evaluation metrics.")
    else:
        model = st.session_state["trained_model"]
        vocab = st.session_state["vocab"]
        normalize_numbers = st.session_state["normalize_numbers"]
        
        test_loader = torch.utils.data.DataLoader(
            TextDataset(test_rows, vocab, 128, normalize_numbers),
            batch_size=32, shuffle=False, collate_fn=collate
        )
        
        model.eval()
        targets, preds = [], []
        with torch.inference_mode():
            for tokens, lengths, labels in test_loader:
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                logits = model(tokens, lengths)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        st.metric("Test Accuracy", f"{accuracy_score(targets, preds):.3f}")
        st.metric("Test Macro-F1", f"{f1_score(targets, preds, average='macro'):.3f}")
        
        st.text("Classification Report:")
        st.text(classification_report(targets, preds, target_names=LABELS))
        
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(targets, preds)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Test Confusion Matrix")
        st.pyplot(fig)
