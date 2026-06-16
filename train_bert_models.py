"""
Full DistilBERT fine-tuning for product and issue classification.
Uses the same filtered dataset and train/holdout split as the TF-IDF models in the notebook.
"""

import json
import os
import pickle
import re
import warnings

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

DATA_FILE = "complaints-2021-05-14_08_16.json"
TEXT_COLUMN = "complaint_what_happened"
TARGET_COLUMNS = ["product", "issue"]
TOP_N_CLASSES = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2

BERT_MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 4
MAX_LEN = 256
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(data_loader), 1)


def evaluate(model, data_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_labels = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    accuracy = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    return preds, targets, accuracy, macro_f1, weighted_f1


def save_bert_artifacts(target_col, model, tokenizer, label_encoder):
    out_dir = f"bert_model_{target_col}"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[OK] Saved BERT -> {out_dir}/")


def load_and_filter_data():
    print("Loading dataset...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    complaints = []
    for record in data:
        if "_source" in record and isinstance(record["_source"], dict):
            complaints.append(record["_source"])

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


def train_and_save_target(target_col, data_filtered, tokenizer):
    print(f"\n{'=' * 80}")
    print(f"TRAINING BERT FOR: {target_col.upper()}")
    print(f"{'=' * 80}")

    df_filtered = data_filtered[target_col].copy()
    print(f"Full dataset: {len(df_filtered):,} samples (same scope as TF-IDF models)")

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df_filtered[target_col].values)
    print(f"Number of classes: {len(label_encoder.classes_)}")

    texts = df_filtered["text_processed"].astype(str).tolist()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        texts,
        labels_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_encoded,
    )
    print(
        f"Train: {len(X_train):,} ({100 * (1 - TEST_SIZE):.0f}%) | "
        f"Holdout: {len(X_holdout):,} ({TEST_SIZE * 100:.0f}%)"
    )

    train_loader = DataLoader(
        ComplaintDataset(X_train, y_train, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    holdout_loader = DataLoader(
        ComplaintDataset(X_holdout, y_holdout, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
    )

    num_labels = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=num_labels)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_macro_f1 = -1.0
    best_state = None
    best_epoch = 0

    print(f"\nTraining for {EPOCHS} epochs on {DEVICE} (batch_size={BATCH_SIZE})...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        _, _, val_acc, val_macro_f1, val_weighted_f1 = evaluate(model, holdout_loader, DEVICE)
        print(
            f"  Epoch {epoch + 1}/{EPOCHS} - Loss: {train_loss:.4f} "
            f"- Holdout Acc: {val_acc:.4f} - Macro F1: {val_macro_f1:.4f} - Weighted F1: {val_weighted_f1:.4f}"
        )
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
        print(f"[OK] Restored best checkpoint from epoch {best_epoch} (macro F1={best_macro_f1:.4f})")

    preds, targets, test_acc, test_macro_f1, test_weighted_f1 = evaluate(model, holdout_loader, DEVICE)
    print(f"\nFinal holdout metrics (best checkpoint):")
    print(f"  Accuracy:    {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Macro F1:    {test_macro_f1:.4f}")
    print(f"  Weighted F1: {test_weighted_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            targets,
            preds,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    save_bert_artifacts(target_col, model, tokenizer, label_encoder)
    return {
        "best_epoch": best_epoch,
        "holdout_accuracy": test_acc,
        "holdout_macro_f1": test_macro_f1,
        "holdout_weighted_f1": test_weighted_f1,
    }


def main():
    print("=" * 80)
    print("BERT FULL TRAIN + SAVE")
    print("=" * 80)
    print(f"Model: {BERT_MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS} | Test size: {TEST_SIZE} | Random state: {RANDOM_STATE}")

    data_filtered = load_and_filter_data()
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    summary = {}
    for target_col in TARGET_COLUMNS:
        summary[target_col] = train_and_save_target(target_col, data_filtered, tokenizer)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for target_col, metrics in summary.items():
        print(
            f"{target_col}: best_epoch={metrics['best_epoch']}, "
            f"acc={metrics['holdout_accuracy']:.4f}, "
            f"macro_f1={metrics['holdout_macro_f1']:.4f} -> bert_model_{target_col}/"
        )
    print("[OK] Done")


if __name__ == "__main__":
    main()
