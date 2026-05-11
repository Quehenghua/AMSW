from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms

from .cir import CIRModule
from .msi import MSIModule, SemanticEncoder
from .saa import SAAModule


class VisualEncoder(nn.Module):

    def __init__(
        self,
        model_name_or_path: str = "google/vit-base-patch16-224-in21k",
        hidden_size: int = 768,
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import ViTModel
        self.vit = ViTModel.from_pretrained(model_name_or_path)
        vit_dim = self.vit.config.hidden_size
        self.proj = (
            nn.Linear(vit_dim, hidden_size)
            if vit_dim != hidden_size
            else nn.Identity()
        )
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        output = self.vit(pixel_values=pixel_values)
        return self.proj(output.last_hidden_state)


class TextualEncoder(nn.Module):

    def __init__(
        self,
        model_name_or_path: str = "hfl/chinese-roberta-wwm-ext",
        hidden_size: int = 768,
        freeze: bool = True,
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        roberta_dim = self.roberta.config.hidden_size
        self.proj = (
            nn.Linear(roberta_dim, hidden_size)
            if roberta_dim != hidden_size
            else nn.Identity()
        )
        if freeze:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.roberta(**kwargs)
        return self.proj(output.last_hidden_state)


class MLPClassifier(nn.Module):
    """Two-layer MLP classifier with layer normalisation and dropout."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AMSW(nn.Module):

    def __init__(
        self,
        visual_encoder_path: str = "google/vit-base-patch16-224-in21k",
        textual_encoder_path: str = "hfl/chinese-roberta-wwm-ext",
        semantic_encoder_path: str = "hfl/chinese-roberta-wwm-ext",
        hidden_size: int = 768,
        num_labels_detection: int = 2,
        num_labels_type: int = 5,
        num_heads: int = 8,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        task: str = "detection",
    ):
        super().__init__()
        assert task in {"detection", "type"}, (
            "task must be 'detection' or 'type'"
        )
        self.task = task
        self.hidden_size = hidden_size

        # Encoders
        self.visual_encoder = VisualEncoder(visual_encoder_path, hidden_size, freeze=True)
        self.textual_encoder = TextualEncoder(textual_encoder_path, hidden_size, freeze=True)

        # CIR
        self.cir = CIRModule(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        # MSI
        sem_encoder = SemanticEncoder(
            model_name_or_path=semantic_encoder_path,
            hidden_size=hidden_size,
            freeze=True,
        )
        self.msi = MSIModule(semantic_encoder=sem_encoder, hidden_size=hidden_size)

        # SAA
        self.saa = SAAModule(
            hidden_size=hidden_size,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )

        # Classifiers for both tasks (both are built; task flag selects which is used)
        num_labels = num_labels_detection if task == "detection" else num_labels_type
        self.classifier = MLPClassifier(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

    def forward(
        self,
        # --- Image inputs ---
        pixel_values: torch.Tensor,
        # --- Meme text inputs ---
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        # --- Pre-computed semantic interpretation inputs ---
        vi_input_ids: torch.Tensor,          # visual interpretation
        vi_attention_mask: torch.Tensor,
        ti_input_ids: torch.Tensor,          # textual interpretation
        ti_attention_mask: torch.Tensor,
        ci_input_ids: torch.Tensor,          # cross-modal interpretation
        ci_attention_mask: torch.Tensor,
        # --- Labels (optional; if provided loss is computed) ---
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        # ---- Step 1: Encode raw modalities ----
        fv = self.visual_encoder(pixel_values)                   # (B, Lv, D)
        ft = self.textual_encoder(text_input_ids, text_attention_mask)  # (B, Lt, D)

        # Derive padding masks for attention (True = ignore)
        text_pad_mask = text_attention_mask == 0                  # (B, Lt)
        # ViT does not use padding; all visual positions are valid
        visual_pad_mask = None

        # ---- Step 2: CIR ----
        hv, ht, hv_att, ht_att = self.cir(
            fv=fv,
            ft=ft,
            visual_mask=visual_pad_mask,
            text_mask=text_pad_mask,
        )

        # ---- Step 3: MSI ----
        zv, zt, zc = self.msi(
            visual_input_ids=vi_input_ids,
            visual_attention_mask=vi_attention_mask,
            textual_input_ids=ti_input_ids,
            textual_attention_mask=ti_attention_mask,
            crossmodal_input_ids=ci_input_ids,
            crossmodal_attention_mask=ci_attention_mask,
        )

        # ---- Step 4: SAA ----
        Z, w = self.saa(
            hv=hv,
            ht=ht,
            hv_att=hv_att,
            ht_att=ht_att,
            zv=zv,
            zt=zt,
            zc=zc,
            visual_mask=visual_pad_mask,
            text_mask=text_pad_mask,
        )

        # ---- Step 5: Classify ----
        logits = self.classifier(Z)                               # (B, num_labels)

        output: dict[str, torch.Tensor] = {"logits": logits, "weights": w}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            output["loss"] = loss_fn(logits, labels)

        return output


    def predict(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        vi_input_ids: torch.Tensor,
        vi_attention_mask: torch.Tensor,
        ti_input_ids: torch.Tensor,
        ti_attention_mask: torch.Tensor,
        ci_input_ids: torch.Tensor,
        ci_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted class indices and branch weights without computing
        a loss.  Useful for inference.

        Returns:
            preds:   (B,)  — predicted label indices
            weights: (B, 3)
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                vi_input_ids=vi_input_ids,
                vi_attention_mask=vi_attention_mask,
                ti_input_ids=ti_input_ids,
                ti_attention_mask=ti_attention_mask,
                ci_input_ids=ci_input_ids,
                ci_attention_mask=ci_attention_mask,
            )
        preds = out["logits"].argmax(dim=-1)
        return preds, out["weights"]
