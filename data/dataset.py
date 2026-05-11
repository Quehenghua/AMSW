from __future__ import annotations

import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms

DETECTION_LABELS = {"Non-Harmful": 0, "Harmful": 1}
TYPE_LABELS = {
    "Non-Harmful": 0,
    "Targeted":    1,
    "Offense":     2,
    "Sexual":      3,
    "Dispirited":  4,
}


VIT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ToxiCNMMDataset(Dataset):
    """
    Dataset for ToxiCN MM.

    Args:
        json_path:           Path to train.json or test.json.
        image_dir:           Directory containing meme images.
        interp_dir:          Directory containing pre-computed interpretation
                             text files (subdirs: visual/, textual/, crossmodal/).
        tokenizer_path:      HuggingFace identifier for the text tokenizer
                             (should match the textual encoder used in AMSW).
        sem_tokenizer_path:  Tokenizer for the semantic encoder (usually the
                             same as textual encoder).
        max_text_len:        Maximum token length for meme inline text.
        max_interp_len:      Maximum token length for interpretation texts.
        task:                "detection" or "type".
        image_transform:     Torchvision transform applied to PIL images.
    """

    def __init__(
        self,
        json_path: str | Path,
        image_dir: str | Path,
        interp_dir: str | Path,
        tokenizer_path: str = "hfl/chinese-roberta-wwm-ext",
        sem_tokenizer_path: str = "hfl/chinese-roberta-wwm-ext",
        max_text_len: int = 128,
        max_interp_len: int = 128,
        task: str = "detection",
        image_transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.interp_dir = Path(interp_dir)
        self.max_text_len = max_text_len
        self.max_interp_len = max_interp_len
        self.task = task
        self.transform = image_transform or VIT_TRANSFORM

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.sem_tokenizer = (
            AutoTokenizer.from_pretrained(sem_tokenizer_path)
            if sem_tokenizer_path != tokenizer_path
            else self.tokenizer
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_interpretation(self, view: str, sample_id: str) -> str:
        path = self.interp_dir / view / f"{sample_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sid = str(sample["id"])

        # ---- Image ----
        img_path = self.image_dir / sample["image"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # ---- Meme inline text ----
        text_enc = self.tokenizer(
            sample["text"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ---- Semantic interpretations ----
        def _tok(text: str) -> dict[str, torch.Tensor]:
            return self.sem_tokenizer(
                text,
                max_length=self.max_interp_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        vi = _tok(self._load_interpretation("visual", sid))
        ti = _tok(self._load_interpretation("textual", sid))
        ci = _tok(self._load_interpretation("crossmodal", sid))

        # ---- Label ----
        if self.task == "detection":
            label = torch.tensor(sample["label"], dtype=torch.long)
        else:
            # For non-harmful samples in the type task, type_label == 0
            type_lbl = sample.get("type_label", 0)
            label = torch.tensor(type_lbl, dtype=torch.long)

        return {
            "pixel_values":         pixel_values,
            "text_input_ids":       text_enc["input_ids"].squeeze(0),
            "text_attention_mask":  text_enc["attention_mask"].squeeze(0),
            "vi_input_ids":         vi["input_ids"].squeeze(0),
            "vi_attention_mask":    vi["attention_mask"].squeeze(0),
            "ti_input_ids":         ti["input_ids"].squeeze(0),
            "ti_attention_mask":    ti["attention_mask"].squeeze(0),
            "ci_input_ids":         ci["input_ids"].squeeze(0),
            "ci_attention_mask":    ci["attention_mask"].squeeze(0),
            "labels":               label,
            "sample_id":            sid,
        }


# ---- DataLoader factory ----
def build_dataloader(
    json_path: str | Path,
    image_dir: str | Path,
    interp_dir: str | Path,
    tokenizer_path: str = "hfl/chinese-roberta-wwm-ext",
    sem_tokenizer_path: str = "hfl/chinese-roberta-wwm-ext",
    max_text_len: int = 128,
    max_interp_len: int = 128,
    task: str = "detection",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:

    dataset = ToxiCNMMDataset(
        json_path=json_path,
        image_dir=image_dir,
        interp_dir=interp_dir,
        tokenizer_path=tokenizer_path,
        sem_tokenizer_path=sem_tokenizer_path,
        max_text_len=max_text_len,
        max_interp_len=max_interp_len,
        task=task,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
