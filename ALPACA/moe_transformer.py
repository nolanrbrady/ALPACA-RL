"""
Transformer with a Mixture-of-Experts (MoE) block, compatible with the
DecoderOnlyTransformer interface used by train_transformer.py.

Inputs are float feature sequences (B, S, F). A Linear projects to d_model,
followed by a TransformerEncoder stack (causal via mask) and a MoE layer that
routes each token representation to a subset of experts. Dual heads produce
continuous and binary outputs analogous to the baseline model.

PositionalEncoding is defined inline here so ALPACA is self-contained.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sine-cosine positional encoding for sequences.

    Adds position embeddings to inputs of shape (S, B, E).
    """

    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (S, 1, E)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: (S, B, E)
        S = x.size(0)
        x = x + self.pe[:S]
        return self.dropout(x)


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        return self.fc2(F.relu(self.fc1(x)))


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E) -> (B, S, num_experts)
        return F.softmax(self.gate(x), dim=2)


class MoELayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )
        self.gate = GatingNetwork(input_dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, num_experts_per_tok: int):
        """
        Forward through MoE with top-k routing and auxiliary load-balancing stats.

        Returns (out, aux) where:
          - out: (B, S, E) mixed expert outputs
          - aux: dict with keys:
              'load_balancing_loss' (scalar tensor),
              'importance' (per-expert mean gate prob, detached),
              'load' (per-expert fraction selected in top-k, detached)
        """
        # x: (B, S, E)
        gate_probs = self.gate(x)  # (B, S, N)
        # Top-k selection mask
        _, topk_idx = gate_probs.topk(num_experts_per_tok, dim=2, sorted=False)
        mask = torch.zeros_like(gate_probs).scatter(2, topk_idx, 1.0)
        # Keep only selected experts' probabilities and renormalize across selected
        gated = gate_probs * mask
        gated = F.normalize(gated, p=1, dim=2)

        # expert_outputs: (B, S, N, E)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # Weighted sum over experts -> (B, S, E)
        out = torch.einsum("bsn,bsne->bse", gated, expert_outputs)

        # Auxiliary load-balancing terms
        B, S, N = gate_probs.shape
        tokens = max(1, B * S)
        # Per-expert average gate probability (importance)
        importance = gate_probs.sum(dim=(0, 1)) / tokens  # (N,)
        # Per-expert fraction of tokens selected in top-k (load)
        load = mask.sum(dim=(0, 1)) / float(tokens)  # (N,)
        # Encourage balanced routing: minimize n * sum(importance_j * load_j)
        lb_loss = (self.num_experts * (importance * load).sum()).to(out.dtype)
        aux = {
            "load_balancing_loss": lb_loss,
            "importance": importance.detach(),
            "load": load.detach(),
        }
        return out, aux


class TransformerWithMoE(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 768,
        dropout: float = 0.3,
        activation: str = "gelu",
        norm_first: bool = True,
        out_cont_dim: int = 0,
        out_bin_dim: int = 0,
        num_experts: int = 8,
        num_experts_per_tok: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.out_cont_dim = out_cont_dim
        self.out_bin_dim = out_bin_dim
        self.num_experts_per_tok = int(max(1, num_experts_per_tok))

        # Input projection and positional encoding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder (decoder-only behavior via causal mask at call time)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MoE layer operating on Transformer outputs
        self.moe = MoELayer(d_model, dim_feedforward, d_model, num_experts)
        self.norm = nn.LayerNorm(d_model)

        # Stash latest auxiliary stats for training to consume
        self._last_aux: dict | None = None

        # Heads
        self.head_cont = nn.Linear(d_model, out_cont_dim) if out_cont_dim > 0 else None
        self.head_bin = nn.Linear(d_model, out_bin_dim) if out_bin_dim > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # x: (B, S, F)
        hidden = self.input_proj(x)  # (B, S, E)
        hidden = hidden.transpose(0, 1)  # (S, B, E)
        hidden = self.pos_enc(hidden)
        hidden = self.encoder(hidden, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        hidden = hidden.transpose(0, 1)  # (B, S, E)

        # Apply MoE and final norm
        hidden, aux = self.moe(hidden, self.num_experts_per_tok)  # (B, S, E), aux dict
        self._last_aux = aux
        hidden = self.norm(hidden)

        pred_cont = self.head_cont(hidden) if self.head_cont is not None else None
        pred_bin = self.head_bin(hidden) if self.head_bin is not None else None
        return pred_cont, pred_bin

    @torch.no_grad()
    def infer_observations(
        self,
        x: torch.Tensor,
        *,
        cont_idx: list,
        bin_idx: list,
        out_dim: int,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = True,
    ) -> torch.Tensor:
        pred_cont, pred_bin = self.forward(
            x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        B, S, _ = x.shape
        device = x.device
        y_full = torch.zeros((B, S, out_dim), dtype=torch.float32, device=device)
        if pred_cont is not None and len(cont_idx) > 0:
            y_full[..., cont_idx] = pred_cont
        if pred_bin is not None and len(bin_idx) > 0:
            y_full[..., bin_idx] = torch.sigmoid(pred_bin) if apply_sigmoid else pred_bin
        return y_full

    def get_auxiliary_losses(self) -> dict:
        """Return latest auxiliary losses/statistics computed during forward."""
        return self._last_aux or {}
