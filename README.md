# Finance & AI CW2 — Temporal Transformer for Credit Card Default Prediction

## Setup

```bash
pip install -r requirements.txt
```

## File Structure

| File | Purpose |
|---|---|
| `data_loader.py` | Fetch UCI dataset, clean, split 80/10/10, build `CreditCardDataset` |
| `transformer_model.py` | Full transformer architecture (embedding, MHSA, FFN, classifier head) |
| `train_transformer.py` | Training loop with early stopping; saves checkpoint + metrics |
| `random_forest.py` | RF benchmark with `RandomizedSearchCV` hyperparameter tuning |
| `hyperparameter_search.py` | Grid sweep over transformer hyperparameters |
| `visualise.py` | EDA plots, attention heatmaps, training curves, model comparison |

## Workflow

```bash
# 1. EDA plots (no model needed)
python visualise.py --mode eda

# 2. Train the transformer (default hyperparameters)
python train_transformer.py

# 3. Train with custom hyperparameters
python train_transformer.py --d_model 64 --n_heads 4 --n_layers 2 --epochs 50

# 4. Run hyperparameter search (takes ~30–60 min on CPU)
python hyperparameter_search.py

# 5. Train random forest benchmark
python random_forest.py

# 6. Generate all visualisations
python visualise.py --mode all
```

## Architecture Summary

**Tokenisation (7 tokens per record):**
- Token 0: Demographics — `[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE]`
- Tokens 1–6: Monthly snapshots (Sep→Apr 2005) — `[PAY_t, BILL_AMT_t, PAY_AMT_t]`

**Embedding:**
- Separate linear projections to `d_model` for demographics vs monthly tokens
- Sinusoidal positional encoding added to monthly tokens only (time-meaningful)

**Transformer block (×N layers):**
- Multi-head self-attention: Q=XW_Q, K=XW_K, V=XW_V
- Scaled dot-product: Attention(Q,K,V) = softmax(QK^T / √d_k) V
- Residual + LayerNorm after each sub-layer

**Classification:**
- Token 0 output → Linear(d_model, d_model//2) → ReLU → Linear → scalar logit → sigmoid

**Training:**
- Loss: `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance (~22% default)
- Optimiser: `AdamW` with cosine LR schedule
- Early stopping on validation AUROC

## Outputs

All artefacts are saved to `outputs/`:
- `transformer_best.pt` — best model checkpoint
- `transformer_history.json` — epoch-by-epoch metrics
- `transformer_results.json` — final test results
- `rf_results.json` — random forest test results
- `figures/` — all plots
