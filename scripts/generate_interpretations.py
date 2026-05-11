from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from models.msi import VISUAL_PROMPT, TEXTUAL_PROMPT, CROSSMODAL_PROMPT


def build_visual_messages(image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": VISUAL_PROMPT},
            ],
        }
    ]


def build_textual_messages(text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Meme text: {text}\n\n{TEXTUAL_PROMPT}"},
            ],
        }
    ]


def build_crossmodal_messages(image: Image.Image, text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": f"Meme text: {text}\n\n{CROSSMODAL_PROMPT}"},
            ],
        }
    ]

def generate_single(
    model,
    processor,
    messages: list[dict],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Run a single inference call and return the generated text."""
    from qwen_vl_utils import process_vision_info  # installed with Qwen2-VL

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Strip the prompt tokens
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MLLM interpretations for ToxiCN MM"
    )
    parser.add_argument("--json_path",      required=True)
    parser.add_argument("--image_dir",      required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--model_name",     default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    for view in ("visual", "textual", "crossmodal"):
        (output_dir / view).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print(f"Loading MLLM: {args.model_name} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()

    with open(args.json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in tqdm(samples, desc="Generating interpretations"):
        sid = str(sample["id"])
        text = sample["text"]
        img_path = image_dir / sample["image"]

        # Check which views still need generation
        needed = []
        for view in ("visual", "textual", "crossmodal"):
            out_path = output_dir / view / f"{sid}.txt"
            if not out_path.exists():
                needed.append(view)

        if not needed:
            continue  # all three already exist — skip

        image = Image.open(img_path).convert("RGB")

        for view in needed:
            out_path = output_dir / view / f"{sid}.txt"
            if view == "visual":
                msgs = build_visual_messages(image)
            elif view == "textual":
                msgs = build_textual_messages(text)
            else:
                msgs = build_crossmodal_messages(image, text)

            interp = generate_single(
                model, processor, msgs, device, args.max_new_tokens
            )
            out_path.write_text(interp, encoding="utf-8")

    print("Done. Interpretations saved to:", output_dir)


if __name__ == "__main__":
    main()
