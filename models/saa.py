from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionPooling(nn.Module):

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        seq: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            seq:          (B, L, D)
            padding_mask: (B, L) boolean; True positions are masked out.

        Returns:
            pooled: (B, D)
        """
        scores = self.W_a(torch.tanh(self.W_h(seq)))  # (B, L, 1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)             # (B, L, 1)
        return (weights * seq).sum(dim=1)              # (B, D)


class SAAModule(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention pooling for unimodal and interaction residual summaries
        self.v_pool = AttentionPooling(hidden_size)     # pools H_v   → sv
        self.t_pool = AttentionPooling(hidden_size)     # pools H_t   → st
        self.va_pool = AttentionPooling(hidden_size)    # pools H_v_att
        self.ta_pool = AttentionPooling(hidden_size)    # pools H_t_att

        # Cross-modal signal projection (concatenation of two summaries → sc)
        self.cross_proj = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # MLP for weight estimation: input s = [sv || st || sc] ∈ ℝ^{3d}
        self.weight_mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 3),  # three branch logits
        )

    def forward(
        self,
        hv: torch.Tensor,
        ht: torch.Tensor,
        hv_att: torch.Tensor,
        ht_att: torch.Tensor,
        zv: torch.Tensor,
        zt: torch.Tensor,
        zc: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # --- Build control signal ---
        sv = self.v_pool(hv, visual_mask)                         # (B, D)
        st = self.t_pool(ht, text_mask)                           # (B, D)

        va_pooled = self.va_pool(hv_att, visual_mask)             # (B, D)
        ta_pooled = self.ta_pool(ht_att, text_mask)               # (B, D)
        sc = self.cross_proj(torch.cat([va_pooled, ta_pooled], dim=-1))  # (B, D)

        s = torch.cat([sv, st, sc], dim=-1)                       # (B, 3D)

        # --- Estimate instance-specific weights ---
        w = F.softmax(self.weight_mlp(s), dim=-1)                 # (B, 3)
        wv, wt, wc = w[:, 0:1], w[:, 1:2], w[:, 2:3]            # (B, 1) each

        # --- Adaptive aggregation ---
        Z = wv * zv + wt * zt + wc * zc                           # (B, D)

        return Z, w
