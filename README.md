# Finance & AI CW2 — Temporal Transformer for Credit Card Default Prediction

## Setup

```bash
pip install -r requirements.txt
```

## File Structure

| File | Purpose |
|---|---|
| `data_loader.py` | Fetch UCI dataset, clean, 70/15/15 split, build `CreditDataset` for skorch |
| `transformer_model.py` | `CreditTokenEmbedding` + `TemporalTransformer` (11 tokens, mean pooling) |
| `train_transformer.py` | skorch `NeuralNetClassifier` + `RandomizedSearchCV` (20 iter, 3-fold) |
| `random_forest.py` | RF on raw data and tokenised embeddings; both with `class_weight="balanced"` |

## Workflow

```bash
# 1. Train the transformer (RandomizedSearchCV over 20 hyperparameter configs)
python train_transformer.py

# 2. Train random forest benchmarks (raw + tokenised)
python random_forest.py
```

## Architecture Summary

**Tokenisation (11 tokens per record):**
- Token 0: `LIMIT_BAL` — Linear(1, d_model)
- Token 1: `SEX` — Embedding(4, d_model)
- Token 2: `EDUCATION` — Embedding(5, d_model)
- Token 3: `MARRIAGE` — Embedding(4, d_model)
- Token 4: `AGE` — Linear(1, d_model)
- Tokens 5–10: Monthly snapshots (Apr→Sep 2005, oldest→recent)
  - numeric: Linear(2, d_model) on `[BILL_AMT, PAY_AMT]`
  - pay status: Embedding(16, d_model) on clamped `PAY` index
  - fusion: Linear(2×d_model, d_model)

**Positional encoding:**
- Learned `nn.Embedding(6, d_model)` applied to monthly tokens only

**Transformer block (×N layers, Pre-Norm):**
- LayerNorm → Multi-head self-attention → residual
- LayerNorm → FFN (ReLU) → residual

**Classification:**
- Mean pooling over all 11 token outputs → Linear(d_model, d_model//2) → ReLU → Linear → logit

**Training:**
- Loss: `BCEWithLogitsLoss` with `pos_weight` (~3.5×) for class imbalance (~22% default rate)
- Optimiser: `Adam`, `lr` tuned via `RandomizedSearchCV`
- Early stopping on `valid_loss` (patience=5), gradient clipping (max norm=1.0)
- Hyperparameter search: 20 random configs, 3-fold CV, scored on F1 macro

## Results

| Model | ROC-AUC | Default recall @0.5 | Default F1 @0.5 | Default recall @0.3 |
|---|---|---|---|---|
| SLM (best config) | 0.7810 | 0.64 | 0.53 | 0.90 |
| RF raw (`class_weight=balanced`) | 0.7740 | 0.52 | 0.53 | 0.75 |
| RF tokenised | 0.4753 | 0.08 | 0.11 | — |

Best SLM config: `d_model=64, n_heads=4, n_layers=2, d_ff=64, dropout=0.1, lr=0.001`

## Outputs

All artefacts are saved to `output/`:
- `SLM_model_weights.pt` — best model weights
- `SLM_loss.png` — training convergence curve
- `SLM_attention.png` — attention heatmap (layer 1, averaged)
- `SLM_confusion_matrix.png` — confusion matrix at threshold 0.5
- `SLM_confusion_matrix_conservative.png` — confusion matrix at threshold 0.3
- `raw_data_RF_model.joblib` — RF (raw data) best estimator
- `tokenised_data_RF_model.joblib` — RF (tokenised) best estimator
- `RF_raw_confusion_matrix.png` / `RF_raw_confusion_matrix_conservative.png`
- `RF_tokenised_confusion_matrix.png`
- `RF_raw_feature_importance.png` / `RF_tokenised_feature_importance.png`
