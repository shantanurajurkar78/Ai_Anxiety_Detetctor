import json
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.preprocessing import clean_text


class AnxietyPredictor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.model_path = self.base_dir / "model" / "anxiety_model.pt"
        self.meta_path = self.base_dir / "model" / "anxiety_model_meta.json"

        if not self.model_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                "Trained model artifacts not found. Run `python model/train_model.py` first."
            )

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.max_length = int(meta["max_length"])
        self.label2id: Dict[str, int] = {k: int(v) for k, v in meta["label2id"].items()}
        self.id2label: Dict[int, str] = {int(k): v for k, v in meta["id2label"].items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = meta.get("model_name", "bert-base-uncased")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label,
        )

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Input text is empty after cleaning.")

        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

        best_idx = int(torch.argmax(probs).item())
        anxiety_level = self.id2label[best_idx]
        confidence = float(probs[best_idx].item())
        scores = {self.id2label[i]: float(probs[i].item()) for i in range(len(self.id2label))}

        return {
            "anxiety_level": anxiety_level,
            "confidence": round(confidence, 4),
            "scores": scores,
        }
