"""
Temporal Tokenisation Transformer for credit-card default prediction.

Architecture:
  11-token sequence (no CLS token):
    Token 0   : LIMIT_BAL  — Linear(1, d_model)
    Token 1   : SEX        — Embedding(4, d_model)
    Token 2   : EDUCATION  — Embedding(5, d_model)
    Token 3   : MARRIAGE   — Embedding(4, d_model)
    Token 4   : AGE        — Linear(1, d_model)
    Tokens 5-10: monthly   — fusion of Linear(2, d_model) + Embedding(16, d_model)
                             via Linear(2*d_model, d_model)

Monthly tokens receive learned positional encoding (nn.Embedding(6, d_model)).
Static tokens receive no positional encoding.

N transformer blocks (Pre-Norm):
  LayerNorm → MultiHeadSelfAttention → residual
  LayerNorm → FFN                    → residual

Mean pooling over all 11 tokens → classification head → P(default).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Multi-Head Self-Attention ──────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape

        Q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores       = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.W_o(context), attn_weights


# ── Feed-Forward Sub-Layer ─────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Transformer Block (Pre-Norm) ───────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-Norm encoder block."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn     = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ff       = FeedForward(d_model, d_ff, dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x, attn_weights


# ── Token Embedding ────────────────────────────────────────────────────────────

class CreditTokenEmbedding(nn.Module):
    """
    Projects each feature into a d_model-dimensional token.

    Static tokens (5):
      LIMIT_BAL  — Linear(1, d_model)
      SEX        — Embedding(4, d_model)   indices 1-2
      EDUCATION  — Embedding(5, d_model)   indices 1-4 (after remapping 0/5/6→4)
      MARRIAGE   — Embedding(4, d_model)   indices 1-3 (after remapping 0→3)
      AGE        — Linear(1, d_model)

    Monthly tokens (6):
      num_vec = Linear(2, d_model) applied to [BILL_AMT, PAY_AMT]
      pay_vec = Embedding(16, d_model) applied to clamped PAY index
      token   = Linear(2*d_model, d_model)([num_vec; pay_vec])
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.sex_emb       = nn.Embedding(4,  d_model)
        self.education_emb = nn.Embedding(5,  d_model)
        self.marriage_emb  = nn.Embedding(4,  d_model)

        self.limit_bal_proj = nn.Linear(1, d_model)
        self.age_proj       = nn.Linear(1, d_model)

        self.monthly_num_proj = nn.Linear(2,           d_model)
        self.pay_status_emb   = nn.Embedding(16,        d_model)
        self.monthly_fusion   = nn.Linear(2 * d_model,  d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        static_num:  torch.Tensor,  # (B, 2)
        static_cat:  torch.Tensor,  # (B, 3)
        monthly_num: torch.Tensor,  # (B, 6, 2)
        monthly_pay: torch.Tensor,  # (B, 6)
    ) -> torch.Tensor:              # (B, 11, d_model)

        limit_bal_tok = self.limit_bal_proj(static_num[:, 0:1]).unsqueeze(1)   # (B,1,d)
        age_tok       = self.age_proj(static_num[:, 1:2]).unsqueeze(1)         # (B,1,d)
        sex_tok       = self.sex_emb(static_cat[:, 0]).unsqueeze(1)            # (B,1,d)
        edu_tok       = self.education_emb(static_cat[:, 1]).unsqueeze(1)      # (B,1,d)
        mar_tok       = self.marriage_emb(static_cat[:, 2]).unsqueeze(1)       # (B,1,d)

        monthly_num_vec = self.monthly_num_proj(monthly_num)                   # (B,6,d)
        pay_vec         = self.pay_status_emb(monthly_pay)                     # (B,6,d)
        monthly_tokens  = self.monthly_fusion(
            torch.cat([monthly_num_vec, pay_vec], dim=-1)
        )                                                                       # (B,6,d)

        # [LIMIT_BAL, SEX, EDU, MARRIAGE, AGE, M1..M6]
        tokens = torch.cat(
            [limit_bal_tok, sex_tok, edu_tok, mar_tok, age_tok, monthly_tokens], dim=1
        )                                                                       # (B,11,d)
        return self.dropout(tokens)


# ── Full Temporal Transformer ──────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Temporal tokenisation transformer for tabular credit-card default data.

    Sequence layout (length = 11):
      positions 0-4 : static tokens   (LIMIT_BAL, SEX, EDU, MARRIAGE, AGE)
      positions 5-10: monthly tokens  (oldest → most recent)

    Classification via mean pooling over all 11 tokens.
    """

    def __init__(
        self,
        d_model:  int   = 64,
        n_heads:  int   = 4,
        n_layers: int   = 2,
        d_ff:     int   = 128,
        dropout:  float = 0.1,
    ):
        super().__init__()

        self.token_embedding = CreditTokenEmbedding(d_model)
        self.pos_emb = nn.Embedding(6, d_model)   # learned PE for 6 monthly tokens only

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def _apply_pe(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(6, device=x.device)
        static  = x[:, :5, :]                              # no PE
        monthly = x[:, 5:, :] + self.pos_emb(positions)   # learned PE
        return torch.cat([static, monthly], dim=1)

    def forward(
        self,
        static_num:  torch.Tensor,
        static_cat:  torch.Tensor,
        monthly_num: torch.Tensor,
        monthly_pay: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embedding(static_num, static_cat, monthly_num, monthly_pay)
        x = self._apply_pe(x)

        for block in self.blocks:
            x, _ = block(x)

        x = self.final_norm(x)
        logits = self.classifier(x.mean(dim=1)).squeeze(-1)   # mean pool over 11 tokens
        return logits

    def get_attention_maps(
        self,
        static_num:  torch.Tensor,
        static_cat:  torch.Tensor,
        monthly_num: torch.Tensor,
        monthly_pay: torch.Tensor,
    ) -> list:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = self.token_embedding(static_num, static_cat, monthly_num, monthly_pay)
            x = self._apply_pe(x)
            attn_maps = []
            for block in self.blocks:
                x, attn_w = block(x)
                attn_maps.append(attn_w)
        self.train(was_training)
        return attn_maps


if __name__ == "__main__":
    model = TemporalTransformer(d_model=64, n_heads=4, n_layers=2, d_ff=128)
    print(model)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total:,}")

    B = 8
    static_num  = torch.randn(B, 2)
    static_cat  = torch.randint(1, 4, (B, 3))
    monthly_num = torch.randn(B, 6, 2)
    monthly_pay = torch.randint(0, 16, (B, 6))

    logits = model(static_num, static_cat, monthly_num, monthly_pay)
    print(f"logits shape: {logits.shape}")   # (8,)
