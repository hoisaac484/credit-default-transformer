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
| `hyperparameter_search.py` | Baseline grid search over transformer hyperparameters (d_model ∈ {32, 64}) |
| `hyperparameter_search_tuning.py` | Expanded random search over transformer hyperparameters (d_model ∈ {64, 128}) |
| `random_forest.py` | RF benchmark with `RandomizedSearchCV` hyperparameter tuning |
| `visualise.py` | EDA plots, attention heatmaps, training curves, model comparison (3 models) |

## Workflow

```bash
# 1. EDA plots (no model needed)
python visualise.py --mode eda

# 2. Train transformer with default hyperparameters
python train_transformer.py

# 3. Rename default outputs before running tuned version
mv outputs/transformer_results.json outputs/transformer_results_default.json
mv outputs/transformer_history.json outputs/transformer_history_default.json
mv outputs/transformer_best.pt outputs/transformer_best_default.pt

# 4. Run expanded random hyperparameter search (takes ~40 min on CPU)
python hyperparameter_search_tuning.py

# 5. Train transformer with best config from search
python train_transformer.py --d_model 128 --n_heads 8 --n_layers 3 --d_ff 128 --dropout 0.2 --lr 0.001 --warmup_epochs 3 --epochs 60 --patience 12

# 6. Rename tuned outputs
mv outputs/transformer_results.json outputs/transformer_results_tuned.json
mv outputs/transformer_history.json outputs/transformer_history_tuned.json
mv outputs/transformer_best.pt outputs/transformer_best_tuned.pt

# 7. Train random forest benchmark
python random_forest.py

# 8. Generate all visualisations
python visualise.py --mode all
```

## Architecture Summary

**Tokenisation (8 tokens per record):**
- Token 0: `[CLS]` — learnable classification token
- Token 1: Demographics — `[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE]`
- Tokens 2–7: Monthly snapshots (Sep→Apr 2005) — `[PAY_t, BILL_AMT_t, PAY_AMT_t, util_ratio_t, pay_ratio_t]`

**Embedding:**
- Separate linear projections to `d_model` for demographics vs monthly tokens
- Sinusoidal positional encoding added to monthly tokens only (time-meaningful)
- `[CLS]` and demographics receive no positional encoding

**Transformer block (×N layers, Pre-Norm):**
- LayerNorm → Multi-head self-attention: Q=XW_Q, K=XW_K, V=XW_V
- Scaled dot-product: Attention(Q,K,V) = softmax(QK^T / √d_k) V
- Residual + LayerNorm → FFN → residual

**Classification:**
- `[CLS]` token output → Linear(d_model, d_model//2) → ReLU → Dropout → Linear → scalar logit → sigmoid

**Training:**
- Loss: `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance (~22% default)
- Optimiser: `AdamW` with linear warmup + cosine LR annealing
- Early stopping on validation AUROC (patience=10 default, 12 for final tuned run)

## Hyperparameter Tuning

Random search following Bergstra & Bengio (2012) with `seed=42` for reproducibility:

| Hyperparameter | Search Space |
|---|---|
| `d_model` | {64, 128} |
| `n_heads` | {2, 4, 8} |
| `n_layers` | {2, 3} |
| `d_ff` | {128, 256} |
| `dropout` | {0.1, 0.2, 0.3} |
| `lr` | {2e-3, 1e-3, 5e-4} |
| `warmup_epochs` | {3, 5} |

Best config found: `d_model=128, n_heads=8, n_layers=3, d_ff=128, dropout=0.2, lr=0.001`

## Results

| Model | Test AUROC | Test Accuracy (tuned threshold) | Parameters |
|---|---|---|---|
| Default Transformer | 0.7614 | 0.7777 | 69,505 |
| **Tuned Transformer** | **0.7688** | 0.7897 | 307,457 |
| Random Forest | 0.7636 | **0.8150** | 200 trees |

## Outputs

All artefacts are saved to `outputs/`:
- `transformer_best_default.pt` — default model checkpoint
- `transformer_best_tuned.pt` — tuned model checkpoint
- `transformer_history_default.json` — default epoch-by-epoch metrics
- `transformer_history_tuned.json` — tuned epoch-by-epoch metrics
- `transformer_results_default.json` — default test results
- `transformer_results_tuned.json` — tuned test results
- `hparam_search.json` — all hyperparameter search trial results
- `rf_results.json` — random forest test results
- `rf_best.joblib` — saved random forest model
- `figures/` — all plots for report