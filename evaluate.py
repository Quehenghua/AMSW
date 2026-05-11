from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from models.amsw import AMSW
from data.dataset import build_dataloader
from utils.metrics import evaluate
from utils.logger import get_logger


logger = get_logger("amsw.eval")


def load_checkpoint(path: str, device: torch.device) -> tuple[AMSW, dict]:
    """Load model weights and associated config from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["config"]
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
        task=cfg["train"]["task"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from '{path}' (epoch {ckpt.get('epoch', '?')})")
    logger.info(f"Checkpoint metrics: {ckpt.get('metrics', {})}")
    return model, cfg


@torch.no_grad()
def run_evaluation(
    model: AMSW,
    loader,
    device: torch.device,
    task: str,
    save_predictions: str | None = None,
) -> dict[str, float]:

    all_preds, all_labels, all_weights, all_ids = [], [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        sample_ids = batch.pop("sample_id", None)
        inputs = {k: v.to(device) for k, v in batch.items()
                  if isinstance(v, torch.Tensor)}
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        preds   = outputs["logits"].argmax(dim=-1)
        weights = outputs["weights"]

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_weights.extend(weights.cpu().tolist())
        if sample_ids is not None:
            all_ids.extend(sample_ids)

    metrics = evaluate(all_labels, all_preds, task=task, verbose=True)

    if save_predictions:
        records = []
        for i, (sid, pred, label, w) in enumerate(
            zip(all_ids or range(len(all_preds)), all_preds, all_labels, all_weights)
        ):
            records.append({
                "id":      sid,
                "pred":    pred,
                "label":   label,
                "weights": {"visual": round(w[0], 4),
                            "textual": round(w[1], 4),
                            "crossmodal": round(w[2], 4)},
            })
        out_path = Path(save_predictions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Predictions saved to '{out_path}'")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AMSW checkpoint")
    parser.add_argument("--checkpoint",       required=True, help="Path to .pt checkpoint")
    parser.add_argument("--task",             choices=["detection", "type"],
                        help="Evaluation task (defaults to checkpoint config)")
    parser.add_argument("--test_json",        help="Override test JSON path")
    parser.add_argument("--image_dir",        help="Override image directory")
    parser.add_argument("--interp_dir",       help="Override interpretation directory")
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--save_predictions", help="Path to save prediction JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)

    task   = args.task or cfg["train"]["task"]
    dcfg   = cfg["data"]
    mcfg   = cfg["model"]

    test_loader = build_dataloader(
        json_path=args.test_json or dcfg["test_json"],
        image_dir=args.image_dir or dcfg["image_dir"],
        interp_dir=args.interp_dir or dcfg["interp_dir"],
        tokenizer_path=mcfg["textual_encoder_path"],
        sem_tokenizer_path=mcfg["semantic_encoder_path"],
        max_text_len=dcfg["max_text_len"],
        max_interp_len=dcfg["max_interp_len"],
        task=task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    metrics = run_evaluation(
        model=model,
        loader=test_loader,
        device=device,
        task=task,
        save_predictions=args.save_predictions,
    )
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
