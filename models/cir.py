import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, Lq, D = query.shape
        Lk = key.shape[1]

        Q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Lq, Lk)

        if key_padding_mask is not None:
            # Expand mask: (B, 1, 1, Lk)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # (B, H, Lq, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out_proj(attended)


class CIRModule(nn.Module):

    def __init__(self, hidden_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.v2t_attn = CrossAttention(hidden_size, num_heads, dropout)  # visual queries text
        self.t2v_attn = CrossAttention(hidden_size, num_heads, dropout)  # text queries visual

        self.v_norm = nn.LayerNorm(hidden_size)
        self.t_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        fv: torch.Tensor,
        ft: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Visual attends to text
        hv_att = self.v2t_attn(query=fv, key=ft, value=ft, key_padding_mask=text_mask)
        # Text attends to visual
        ht_att = self.t2v_attn(query=ft, key=fv, value=fv, key_padding_mask=visual_mask)

        # Residual connection + layer normalisation
        hv = self.v_norm(fv + hv_att)
        ht = self.t_norm(ft + ht_att)

        return hv, ht, hv_att, ht_att
