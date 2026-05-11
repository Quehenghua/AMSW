"""
Training Script for AMSW
========================
Usage
-----
  # Task 1 — Harmful Meme Detection
  python train.py --config configs/default.yaml --task detection

  # Task 2 — Harmful Type Identification
  python train.py --config configs/default.yaml --task type

  # Override any config key inline
  python train.py --config configs/default.yaml --task detection \\
      --learning_rate 5e-5 --epochs 20 --batch_size 16

The script:
  1. Loads configuration from a YAML file (with optional CLI overrides).
  2. Instantiates the AMSW model, optimiser, and scheduler.
  3. Runs training for the specified number of epochs.
  4. Evaluates on the test split after every epoch.
  5. Saves the best checkpoint according to macro-F1.
"""

from __future__ import annotations

import argparse
import os
import random
import yaml
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm import tqdm

from models.amsw import AMSW
from data.dataset import build_dataloader
from utils.metrics import evaluate
from utils.logger import get_logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    override_map = {
        "task":          ("train", "task"),
        "epochs":        ("train", "epochs"),
        "batch_size":    ("train", "batch_size"),
        "learning_rate": ("train", "learning_rate"),
        "seed":          ("train", "seed"),
    }
    for arg_key, (section, cfg_key) in override_map.items():
        val = getattr(args, arg_key, None)
        if val is not None:
            cfg[section][cfg_key] = val
    return cfg


def train_epoch(
    model: AMSW,
    loader,
    optimiser: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
    logger,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Training", leave=False):
        inputs = {k: v.to(device) for k, v in batch.items()
                  if isinstance(v, torch.Tensor) and k != "sample_id"}
        outputs = model(**inputs)
        loss = outputs["loss"]

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: AMSW,
    loader,
    device: torch.device,
    task: str,
    verbose: bool = False,
) -> dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        inputs = {k: v.to(device) for k, v in batch.items()
                  if isinstance(v, torch.Tensor) and k != "sample_id"}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        preds = outputs["logits"].argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return evaluate(all_labels, all_preds, task=task, verbose=verbose)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train AMSW")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", choices=["detection", "type"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_args(cfg, args)

    task     = cfg["train"]["task"]
    epochs   = cfg["train"]["epochs"]
    bs       = cfg["train"]["batch_size"]
    lr       = cfg["train"]["learning_rate"]
    wd       = cfg["train"]["weight_decay"]
    warmup   = cfg["train"]["warmup_steps"]
    gc       = cfg["train"]["grad_clip"]
    seed     = cfg["train"]["seed"]
    n_workers = cfg["train"]["num_workers"]

    save_dir = Path(cfg["output"]["save_dir"]) / task
    log_dir  = Path(cfg["output"]["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("amsw.train", log_dir=log_dir)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Task: {task}  |  Epochs: {epochs}")

    # ---- Data ----
    dcfg = cfg["data"]
    train_loader = build_dataloader(
        json_path=dcfg["train_json"],
        image_dir=dcfg["image_dir"],
        interp_dir=dcfg["interp_dir"],
        tokenizer_path=cfg["model"]["textual_encoder_path"],
        sem_tokenizer_path=cfg["model"]["semantic_encoder_path"],
        max_text_len=dcfg["max_text_len"],
        max_interp_len=dcfg["max_interp_len"],
        task=task,
        batch_size=bs,
        num_workers=n_workers,
        shuffle=True,
    )
    test_loader = build_dataloader(
        json_path=dcfg["test_json"],
        image_dir=dcfg["image_dir"],
        interp_dir=dcfg["interp_dir"],
        tokenizer_path=cfg["model"]["textual_encoder_path"],
        sem_tokenizer_path=cfg["model"]["semantic_encoder_path"],
        max_text_len=dcfg["max_text_len"],
        max_interp_len=dcfg["max_interp_len"],
        task=task,
        batch_size=bs,
        num_workers=n_workers,
        shuffle=False,
    )

    # ---- Model ----
    mcfg = cfg["model"]
    model = AMSW(
        visual_encoder_path=mcfg["visual_encoder_path"],
        textual_encoder_path=mcfg["textual_encoder_path"],
        semantic_encoder_path=mcfg["semantic_encoder_path"],
        hidden_size=mcfg["hidden_size"],
        num_labels_detection=mcfg["num_labels_detection"],
        num_labels_type=mcfg["num_labels_type"],
        num_heads=mcfg["num_heads"],
        mlp_hidden=mcfg["mlp_hidden"],
        dropout=mcfg["dropout"],
        task=task,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    # ---- Optimiser & Scheduler ----
    optimiser = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = min(warmup, total_steps - 1) if total_steps > 1 else 1
    warmup_sched = LinearLR(optimiser, start_factor=0.1, end_factor=1.0,
                            total_iters=warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    decay_sched  = CosineAnnealingLR(optimiser, T_max=decay_steps)
    scheduler    = SequentialLR(optimiser, [warmup_sched, decay_sched],
                                milestones=[warmup_steps])

    # ---- Training loop ----
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        train_loss = train_epoch(
            model, train_loader, optimiser, scheduler, device, gc, logger
        )
        logger.info(f"  Train loss: {train_loss:.4f}")

        metrics = eval_epoch(model, test_loader, device, task=task, verbose=True)
        logger.info(f"  Test metrics: {metrics}")

        # Save best checkpoint
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_epoch = epoch
            ckpt_path = save_dir / "best_model.pt"
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "metrics":    metrics,
                "config":     cfg,
            }, ckpt_path)
            logger.info(f"  ✓ New best F1 = {best_f1:.2f}% — checkpoint saved.")

    logger.info(
        f"\nTraining complete. Best macro-F1 = {best_f1:.2f}% at epoch {best_epoch}."
    )


if __name__ == "__main__":
    main()
