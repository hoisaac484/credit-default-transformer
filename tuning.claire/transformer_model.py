"""
Temporal Tokenisation Transformer for credit-card default prediction.

Architecture (Section 3 of report):
  [CLS]     : learnable classification token                          → d_model
  Token 0   : demographics  [LIMIT_BAL, SEX, EDU, MARRIAGE, AGE]    → dim 5
  Tokens 1-6: monthly       [PAY_t, BILL_AMT_t, PAY_AMT_t,
                              util_ratio_t, pay_ratio_t]              → dim 5

Each token group has its own linear input projection to d_model.
Tokens 1-6 receive sinusoidal positional encoding (position = month index).
[CLS] and demographics receive no positional encoding.

The sequence (8 × d_model) passes through N transformer blocks using
Pre-Norm (LayerNorm before each sub-layer — more stable than post-norm):
  LayerNorm → MultiHeadSelfAttention → residual
  LayerNorm → FFN                    → residual

The [CLS] token output is fed to the classification head → P(default).

Changes vs v1:
  - Learnable [CLS] token (clean separation of readout from demographics)
  - Pre-norm (Pre-LN) for more stable gradient flow
  - month_dim 3 → 5 (added utilisation ratio and payment ratio features)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Positional Encoding ────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding. Applied to the 6 monthly tokens only."""
    def __init__(self, d_model: int, max_len: int = 10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :d_model // 2]
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ── Scaled Dot-Product Attention ───────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Explicit multi-head self-attention (Vaswani et al. 2017).

    For each head h:
        Q_h = X W_Q_h,   K_h = X W_K_h,   V_h = X W_V_h
        A_h = softmax( Q_h K_h^T / sqrt(d_k) ) V_h

    Heads concatenated → projected back to d_model via W_O.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:  x: (B, seq_len, d_model)
        Returns: out (B, seq_len, d_model), attn_weights (B, H, seq_len, seq_len)
        """
        B, S, _ = x.shape

        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores      = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(context), attn_weights


# ── Feed-Forward Sub-Layer ─────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Dropout → Linear."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block (Pre-Norm) ───────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-Norm encoder block (more stable gradient flow than post-norm):
      x → LayerNorm → MHSA → residual add
        → LayerNorm → FFN  → residual add
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff    = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm: normalise before the sub-layer
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop(attn_out)

        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_weights


# ── Full Temporal Transformer ──────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Temporal tokenisation transformer for tabular credit-card default data.

    Sequence layout (length = 8):
      position 0  : [CLS] token  — learnable, used for final classification
      position 1  : demographics — LIMIT_BAL, SEX, EDU, MARRIAGE, AGE
      positions 2-7: monthly     — (PAY, BILL, AMT, util_ratio, pay_ratio) × 6 months

    Hyperparameters
    ---------------
    d_model   : embedding dimension         (default 64)
    n_heads   : number of attention heads   (default 4)
    n_layers  : number of transformer blocks (default 2)
    d_ff      : FFN inner dimension         (default 128)
    dropout   : dropout probability         (default 0.1)
    demo_dim  : demographic feature count   (default 5)
    month_dim : features per monthly token  (default 5)
    """

    def __init__(
        self,
        d_model:   int   = 64,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        d_ff:      int   = 128,
        dropout:   float = 0.1,
        demo_dim:  int   = 5,
        month_dim: int   = 5,
    ):
        super().__init__()

        # Learnable [CLS] token (initialised to zeros, trained end-to-end)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Input projections — separate weights per token type
        self.demo_proj    = nn.Linear(demo_dim,  d_model)
        self.monthly_proj = nn.Linear(month_dim, d_model)

        # Final LayerNorm after all blocks (Pre-LN convention)
        self.final_norm = nn.LayerNorm(d_model)

        # Positional encoding for monthly tokens (positions 0–5 = Sep→Apr)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=6)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head on [CLS] output
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, demo: torch.Tensor, monthly: torch.Tensor):
        """
        Args:
            demo    : (B, 1, 5)  — demographics token
            monthly : (B, 6, 5)  — six monthly tokens (with engineered features)

        Returns:
            logits   : (B,)          — raw logit; sigmoid → P(default)
            attn_maps: list of (B, n_heads, 8, 8) per layer
        """
        B = demo.size(0)

        # Project each token type to d_model
        e_demo    = self.demo_proj(demo)       # (B, 1, d_model)
        e_monthly = self.monthly_proj(monthly)  # (B, 6, d_model)

        # Add sinusoidal positional encoding to monthly tokens only
        e_monthly = self.pos_enc(e_monthly)    # (B, 6, d_model)

        # Expand [CLS] token to batch
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)

        # Sequence: [CLS] | demo | monthly (length = 8)
        x = torch.cat([cls, e_demo, e_monthly], dim=1)  # (B, 8, d_model)

        # Pass through transformer blocks
        attn_maps = []
        for block in self.blocks:
            x, attn_w = block(x)
            attn_maps.append(attn_w)

        x = self.final_norm(x)

        # Classify from [CLS] position (index 0)
        cls_repr = x[:, 0, :]                           # (B, d_model)
        logits   = self.classifier(cls_repr).squeeze(-1) # (B,)

        return logits, attn_maps


if __name__ == "__main__":
    model = TemporalTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128)
    print(model)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total:,}")

    B = 8
    demo    = torch.randn(B, 1, 5)
    monthly = torch.randn(B, 6, 5)
    logits, attn_maps = model(demo, monthly)
    print(f"logits shape   : {logits.shape}")         # (8,)
    print(f"attn_map shape : {attn_maps[0].shape}")   # (8, 4, 8, 8)
