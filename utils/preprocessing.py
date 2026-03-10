import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def clean_text(text: str) -> str:
    """
    Basic text normalization:
    - Lowercase
    - Remove special characters except sentence punctuation
    - Collapse extra whitespace
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def encode_labels(labels: List[str]) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    encoded = [label2id[label] for label in labels]
    return encoded, label2id, id2label


class AnxietyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_synthetic_dataset(output_csv: str, n_samples: int = 240, seed: int = 42) -> pd.DataFrame:
    """
    Build a balanced synthetic dataset with three classes:
    Low, Moderate, High.
    """
    random.seed(seed)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    low_openers = [
        "I feel calm about",
        "I am confident for",
        "I am relaxed before",
        "I feel prepared for",
        "I feel positive about",
        "I am comfortable with",
    ]
    moderate_openers = [
        "I feel a little worried about",
        "I am somewhat nervous for",
        "I feel pressure for",
        "I am slightly stressed about",
        "I am concerned about",
        "I feel uncertain about",
    ]
    high_openers = [
        "I feel very anxious about",
        "I am extremely stressed about",
        "I cannot stop worrying about",
        "I panic when I think about",
        "I feel overwhelmed by",
        "I feel terrified of",
    ]
    exam_targets = [
        "my final exams",
        "the math test",
        "the entrance exam",
        "tomorrow's paper",
        "my semester assessment",
        "the chemistry exam",
        "the board exam",
    ]
    low_endings = [
        "because I revised consistently.",
        "and I think I can do well.",
        "after practicing many mock tests.",
        "and my study plan is working.",
        "because I understand the topics clearly.",
    ]
    moderate_endings = [
        "but I believe I can handle it.",
        "though I am trying breathing exercises.",
        "and I need better time management.",
        "yet I am still preparing daily.",
        "but I think I can improve with practice.",
    ]
    high_endings = [
        "and I cannot focus on anything else.",
        "and my sleep is badly affected.",
        "and I feel like I will fail.",
        "and my heart races all day.",
        "and I keep thinking about worst outcomes.",
    ]

    samples_per_class = n_samples // 3
    rows = []

    for _ in range(samples_per_class):
        rows.append(
            {
                "text": f"{random.choice(low_openers)} {random.choice(exam_targets)} {random.choice(low_endings)}",
                "label": "Low",
            }
        )
        rows.append(
            {
                "text": f"{random.choice(moderate_openers)} {random.choice(exam_targets)} {random.choice(moderate_endings)}",
                "label": "Moderate",
            }
        )
        rows.append(
            {
                "text": f"{random.choice(high_openers)} {random.choice(exam_targets)} {random.choice(high_endings)}",
                "label": "High",
            }
        )

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
