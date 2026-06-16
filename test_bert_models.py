"""
Evaluate saved BERT models for product and issue classification.
Uses the same filtered dataset and holdout split as train_bert_models.py.
"""

import json
import os
import pickle
import re
import warnings
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "complaints-2021-05-14_08_16.json"
PRODUCT_MODEL_DIR = PROJECT_ROOT / "bert_model_product"
ISSUE_MODEL_DIR = PROJECT_ROOT / "bert_model_issue"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_LEN = 256
BATCH_SIZE = 16
TOP_N_CLASSES = 10
TEXT_COLUMN = "complaint_what_happened"
TARGET_COLUMNS = ["product", "issue"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_data_for_target(df, text_col, target_col):
    df_clean = df[[text_col, target_col]].copy()
    df_clean = df_clean.dropna(subset=[text_col, target_col])
    df_clean = df_clean[df_clean[text_col].str.strip() != ""]
    df_clean = df_clean[df_clean[target_col].str.strip() != ""]
    return df_clean


class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_model_and_tokenizer(model_dir):
    print(f"Loading model from: {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    label_encoder_path = model_dir / "label_encoder.pkl"
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    print("[OK] Model loaded successfully")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    return model, tokenizer, label_encoder


def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_labels = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    return all_preds, all_targets, accuracy, macro_f1, weighted_f1


def load_and_filter_data():
    print("Loading dataset...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    complaints = [
        record["_source"]
        for record in data
        if "_source" in record and isinstance(record["_source"], dict)
    ]
    df = pd.DataFrame(complaints)
    print(f"[OK] Dataset loaded: {len(df):,} records")

    data_filtered = {}
    for target_col in TARGET_COLUMNS:
        df_clean = prepare_data_for_target(df, TEXT_COLUMN, target_col)
        top_classes = df_clean[target_col].value_counts().head(TOP_N_CLASSES).index.tolist()
        df_filtered = df_clean[df_clean[target_col].isin(top_classes)].copy()
        df_filtered["text_processed"] = df_filtered[TEXT_COLUMN].apply(preprocess_text)
        data_filtered[target_col] = df_filtered
        print(f"[OK] Filtered {len(df_filtered):,} records for '{target_col}'")
    return data_filtered


def test_bert_model(model_dir, target_col, df_filtered):
    print(f"\n{'=' * 80}")
    print(f"Testing BERT Model for: {target_col.upper()}")
    print(f"{'=' * 80}")

    model, tokenizer, label_encoder = load_model_and_tokenizer(model_dir)
    df_filtered = df_filtered[df_filtered[target_col].isin(label_encoder.classes_)].copy()
    print(f"\nDataset size: {len(df_filtered):,} samples")

    labels_encoded = label_encoder.transform(df_filtered[target_col].values)
    texts = df_filtered["text_processed"].astype(str).tolist()
    _, X_test, _, y_test = train_test_split(
        texts,
        labels_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_encoded,
    )
    print(f"Holdout set size: {len(X_test):,} samples")

    test_dataset = ComplaintDataset(X_test, y_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nEvaluating model on holdout set...")
    preds, targets, accuracy, macro_f1, weighted_f1 = evaluate_model(model, test_loader)

    print(f"\n{'=' * 80}")
    print(f"RESULTS for {target_col.upper()}")
    print(f"{'=' * 80}")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\n{'=' * 80}")
    print("Classification Report:")
    print(f"{'=' * 80}")
    print(
        classification_report(
            targets,
            preds,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    print("=" * 80)
    print("BERT MODEL EVALUATION SCRIPT")
    print("=" * 80)

    for path in (PRODUCT_MODEL_DIR, ISSUE_MODEL_DIR):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing model directory: {path}")

    data_filtered = load_and_filter_data()
    results = {
        "product": test_bert_model(PRODUCT_MODEL_DIR, "product", data_filtered["product"]),
        "issue": test_bert_model(ISSUE_MODEL_DIR, "issue", data_filtered["issue"]),
    }

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Accuracy':<15} {'Macro F1':<15} {'Weighted F1':<15}")
    print("-" * 80)
    for target_col in TARGET_COLUMNS:
        r = results[target_col]
        print(
            f"{target_col.upper():<20} {r['accuracy']:<15.4f} "
            f"{r['macro_f1']:<15.4f} {r['weighted_f1']:<15.4f}"
        )
    print(f"\n{'=' * 80}")
    print("Evaluation complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
