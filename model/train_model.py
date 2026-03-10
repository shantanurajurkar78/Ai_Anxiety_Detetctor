import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.preprocessing import AnxietyDataset, clean_text, create_synthetic_dataset, encode_labels


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def main():
    data_path = ROOT_DIR / "data" / "dataset.csv"
    model_dir = ROOT_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print("Dataset not found. Creating synthetic dataset...")
        create_synthetic_dataset(str(data_path), n_samples=240, seed=42)

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).apply(clean_text)

    encoded_labels, label2id, id2label = encode_labels(df["label"].tolist())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels,
    )

    model_name = "bert-base-uncased"
    max_length = 128
    batch_size = 8
    epochs = 3

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = AnxietyDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = AnxietyDataset(val_texts, val_labels, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += compute_accuracy(outputs.logits.detach(), labels)

        train_loss /= max(len(train_loader), 1)
        train_acc /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_acc += compute_accuracy(outputs.logits, labels)

        val_loss /= max(len(val_loader), 1)
        val_acc /= max(len(val_loader), 1)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}"
        )

    model_path = model_dir / "anxiety_model.pt"
    meta_path = model_dir / "anxiety_model_meta.json"

    torch.save(model.state_dict(), model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "max_length": max_length,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            indent=2,
        )

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
