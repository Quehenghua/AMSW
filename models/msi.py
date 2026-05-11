from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


VISUAL_PROMPT = (
    "请忽略所有文本元素。仅从视觉角度出发，对图像中的主要视觉实体、场景及构图结构进行简明扼要的解读，并考量该图像本身可能传达的任何潜在隐性信息。"
)

TEXTUAL_PROMPT = (
    "对该文本的字面内容、隐含意义及语用意图进行简明阐释。从中文角度分析文本中可能存在的双关、讽刺、文化参考等修辞手法，并考量文本在不同语境下可能传达的多层次含义。"
)

CROSSMODAL_PROMPT = (
    "综合考虑该表情包中的图像与文字，分析二者结合所产生的整体含义；并对它们之间的相互关系，以及这种结合所蕴含的深层寓意，进行简明扼要的解读。"
)


class SemanticEncoder(nn.Module):

    def __init__(
        self,
        model_name_or_path: str = "hfl/chinese-roberta-wwm-ext",
        hidden_size: int = 768,
        max_length: int = 128,
        freeze: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.max_length = max_length

        encoder_dim = self.encoder.config.hidden_size
        # Linear projection to shared space (identity when encoder_dim == hidden_size)
        self.proj = (
            nn.Linear(encoder_dim, hidden_size)
            if encoder_dim != hidden_size
            else nn.Identity()
        )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode_texts(
        self,
        texts: list[str],
        device: torch.device,
    ) -> torch.Tensor:

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad() if not self.training else torch.enable_grad():
            output = self.encoder(**encoding)
        cls = output.last_hidden_state[:, 0, :]  # (N, encoder_dim)
        return self.proj(cls)                     # (N, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        cls = output.last_hidden_state[:, 0, :]
        return self.proj(cls)


# ---------------------------------------------------------------------------
# MSI module
# ---------------------------------------------------------------------------
class MSIModule(nn.Module):

    def __init__(self, semantic_encoder: SemanticEncoder, hidden_size: int = 768):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.hidden_size = hidden_size

    def forward(
        self,
        visual_input_ids: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        textual_input_ids: torch.Tensor,
        textual_attention_mask: torch.Tensor,
        crossmodal_input_ids: torch.Tensor,
        crossmodal_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        zv = self.semantic_encoder(visual_input_ids, visual_attention_mask)
        zt = self.semantic_encoder(textual_input_ids, textual_attention_mask)
        zc = self.semantic_encoder(crossmodal_input_ids, crossmodal_attention_mask)
        return zv, zt, zc
