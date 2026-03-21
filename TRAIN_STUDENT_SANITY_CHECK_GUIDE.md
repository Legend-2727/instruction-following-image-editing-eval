# train_student.py Sanity Checking Guide

## Improvements Made (Phase 4)

The `train_student.py` script has been enhanced for classifier debugging and sanity checking:

### New CLI Arguments

1. **`--overfit_n N`** (default: 0)
   - Train on only the first N samples from the training set
   - Useful for debugging convergence on tiny subsets
   - 0 means use all training samples

2. **`--taxonomy_threshold THRESHOLD`** (default: 0.5)
   - Threshold for predicting taxonomy (error) labels
   - Values < 0.5 predict more labels (higher recall)
   - Values > 0.5 predict fewer labels (higher precision)

3. **`--epochs`** (already existed, default: 3)
   - Now available for smoke-test configuration

### New Debug Stats in metrics.json

The script now saves three additional statistics in `metrics.json`:

```json
{
  "mean_taxonomy_logit": <float>,               // Mean raw logit across all samples
  "mean_taxonomy_sigmoid": <float>,             // Mean sigmoid probability (0-1)
  "avg_predicted_labels_per_sample": <float>    // Average taxonomy labels predicted per sample
}
```

These stats help assess:
- **Calibration**: Is the model predicting extreme logits? (healthy: ≈ 0; pathological: >> 0 or << 0)
- **Sigmoid balance**: Are probabilities centered correctly? (healthy: ≈ 0.5; pathological: too high or too low)
- **Label frequency**: How many error types per sample? (depends on data, but consistency indicates stability)

---

## Exact Commands for Testing

### 1. Overfit Test (with `--overfit_n 8`)

Train on only the first 8 training samples to verify gradient flow and convergence:

```bash
cd /home/user/ML-Project\(42_46\)/recovered_repo/xlingual_repo

python scripts/train_student.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --translations_csv data/sample/translations.csv \
  --out runs/test_overfit_n8 \
  --epochs 1 \
  --batch_size 2 \
  --overfit_n 8 \
  --seed 42
```

**Expected output:**
- Log message: `INFO | Overfitting on first 8 train samples.`
- Files saved:
  - `runs/test_overfit_n8/student_model.pt`
  - `runs/test_overfit_n8/metrics.json` (with debug stats)
  - `runs/test_overfit_n8/predictions.jsonl`
  - `runs/test_overfit_n8/config_snapshot.json`

**Verification:**
```bash
cat runs/test_overfit_n8/metrics.json | python -m json.tool
```

Sample output:
```json
{
  "train_loss": 1.84,
  "val_loss": 2.58,
  "adherence_accuracy": 0.25,
  "adherence_macro_f1": 0.133,
  "taxonomy_micro_f1": 0.333,
  "taxonomy_macro_f1": 0.036,
  "taxonomy_mAP": 0.667,
  "mean_taxonomy_logit": -0.725,
  "mean_taxonomy_sigmoid": 0.336,
  "avg_predicted_labels_per_sample": 1.0
}
```

---

### 2. Smoke Ablation: English Only

Test text encoding with English translations only:

```bash
python scripts/train_student.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --translations_csv data/sample/translations.csv \
  --out runs/test_text_english_only \
  --text_mode english_only \
  --epochs 1 \
  --batch_size 2 \
  --seed 42 \
  --limit 12
```

**Key argument:** `--text_mode english_only`

**Expected output:**
- Log: `INFO | Samples: total=12 train=9 val=3`
- Training completes in ~30-60 seconds
- Files saved to `runs/test_text_english_only/`

**Check debug stats:**
```bash
cat runs/test_text_english_only/metrics.json | python -c "import sys, json; m=json.load(sys.stdin); print('mean_taxonomy_sigmoid:', m['mean_taxonomy_sigmoid'])"
```

---

### 3. Smoke Ablation: Original Language Only

Test text encoding with original-language instructions only:

```bash
python scripts/train_student.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --translations_csv data/sample/translations.csv \
  --out runs/test_text_original_only \
  --text_mode original_only \
  --epochs 1 \
  --batch_size 2 \
  --seed 42 \
  --limit 12
```

**Key argument:** `--text_mode original_only`

**Expected output:**
- Same structure as english_only test
- Files saved to `runs/test_text_original_only/`
- Should show slightly different debug stats due to different text input

---

### 4. Smoke Ablation: Both Text Modes

Test with concatenated English + original language text:

```bash
python scripts/train_student.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --translations_csv data/sample/translations.csv \
  --out runs/test_text_both \
  --text_mode both \
  --epochs 1 \
  --batch_size 2 \
  --seed 42 \
  --limit 12
```

**Key argument:** `--text_mode both`

**Expected output:**
- Same structure as previous ablations
- Files saved to `runs/test_text_both/`
- Text inputs will include concatenation: `[original] [SEP] [english]`

---

## Comparative Analysis Script

Compare debug stats across the three text-mode ablations:

```bash
cd /home/user/ML-Project\(42_46\)/recovered_repo/xlingual_repo

python << 'EOF'
import json
import os

results = {}
for text_mode in ['english_only', 'original_only', 'both']:
    metrics_file = f'runs/test_text_{text_mode}/metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            m = json.load(f)
        results[text_mode] = {
            'adh_acc': m.get('adherence_accuracy', 0),
            'tax_logit': m.get('mean_taxonomy_logit', 0),
            'tax_sigmoid': m.get('mean_taxonomy_sigmoid', 0),
            'avg_labels': m.get('avg_predicted_labels_per_sample', 0),
        }

print("\n=== Text Mode Ablation Comparison ===\n")
for mode in ['english_only', 'original_only', 'both']:
    if mode in results:
        r = results[mode]
        print(f"{mode:20s} | adh_acc={r['adh_acc']:.3f} | tax_logit={r['tax_logit']:.3f} | " +
              f"sigmoid={r['tax_sigmoid']:.3f} | avg_labels={r['avg_labels']:.1f}")
EOF
```

---

## Threshold Debugging (Optional)

To test different taxonomy prediction thresholds:

```bash
# Conservative threshold (predict fewer labels)
python scripts/train_student.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --out runs/thresh_0p7 \
  --taxonomy_threshold 0.7 \
  --limit 12 --epochs 1

# Aggressive threshold (predict more labels)
python scripts/train_student.py \
  --metadata data/sample/metadata.jsonl \
  --labels data/annotations/labels.jsonl \
  --out runs/thresh_0p3 \
  --taxonomy_threshold 0.3 \
  --limit 12 --epochs 1

# Compare label counts
python << 'EOF'
import json
for thresh in [0.3, 0.7]:
    with open(f'runs/thresh_0p{int(thresh*10)}/metrics.json') as f:
        m = json.load(f)
    print(f"Threshold {thresh}: avg_labels_per_sample = {m['avg_predicted_labels_per_sample']:.2f}")
EOF
```

---

## What to Check in These Tests

### 1. **Code Runs Without Error**
   - No crashes or import errors
   - All required files load correctly
   - Torch/GPU initialization works

### 2. **Overfit Test Convergence**
   - `train_loss` should decrease across epochs
   - If `--overfit_n` is very small (8 samples), overfitting is expected
   - `val_loss` may increase if overfitting is happening (normal)

### 3. **Debug Stats Sanity**
   - `mean_taxonomy_logit` should be roughly centered (-1 to +1 is healthy; >2 or <-2 suggests saturation)
   - `mean_taxonomy_sigmoid` should be between 0-1 (typically 0.2-0.8 is healthy)
   - `avg_predicted_labels_per_sample` should be > 0 (if == 0, threshold might be too high)

### 4. **Text Mode Differences**
   - `--text_mode both` may show slightly different debug stats than single-language modes
   - Ablations help isolate which text representation works best

### 5. **File Structure**
   - All expected outputs are created: `student_model.pt`, `metrics.json`, `predictions.jsonl`, `config_snapshot.json`
   - Config snapshot should correctly record the command-line arguments used

---

## When to Use Each Test

| Test | Use Case |
|------|----------|
| `--overfit_n 8` | Verify gradient flow, check for NaNs, quick convergence test |
| `--text_mode english_only` | Ablate original language (baseline for multilingual comparison) |
| `--text_mode original_only` | Ablate English translation (test low-resource path) |
| `--text_mode both` | Full multilingual setup (expected best case) |
| `--taxonomy_threshold X` | Fine-tune precision/recall tradeoff post-training |

---

## Next Steps

After sanity checking passes:

1. **Scale up to full dataset**: Remove `--limit` or set to full size
2. **Increase epochs**: Set `--epochs 10-20` for real training
3. **Tune hyperparameters**: Adjust `--lr`, `--batch_size`, `--weight_decay`
4. **Run full ablation suite**: Document all three text modes on full data
5. **Generate paper artifacts**: Export comparison tables and plots from runs

---

## Code Changes Summary

### Files Modified
- `scripts/train_student.py`

### Changes Made
1. Added `--overfit_n` argument in `parse_args()`
2. Added `--taxonomy_threshold` argument in `parse_args()`
3. Modified `evaluate()` function to:
   - Accept `taxonomy_threshold` parameter
   - Collect raw logits (`y_logits_err`)
   - Compute `mean_taxonomy_logit` and `mean_taxonomy_sigmoid`
   - Compute `avg_predicted_labels_per_sample`
   - Include all three stats in returned metrics dict
4. Modified `main()` function to:
   - Apply `--overfit_n` filtering to training records
   - Pass `taxonomy_threshold` to `evaluate()` calls

### Lines Changed
- ~20 total lines (minimal, focused changes)
- No model architecture changes
- No breaking changes to existing functionality

---
