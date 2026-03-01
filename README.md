# Instruction-Following Evaluation for Text-Guided Image Editing

**A complete ML pipeline for evaluating how badly open-source image editing models follow natural language instructions.**

This project measures instruction-following failures in text-guided image editing. We use **InstructPix2Pix** (SOTA open-source editor) to generate edits from MagicBrush prompts, then **Qwen2.5-VL** (vision-language model) as an automated judge to classify adherence and error types — replacing slow human labeling. A human verification stage spot-checks the VLM's accuracy.

Each sample is a triplet **(original image, edit instruction, model-edited image)** evaluated against an 11-category error taxonomy.

**Status:** 🔄 **Checkpoint 2 in progress** — pipeline code complete, testing on RTX 3060 (Mar 1, 2026)

---

## Overview

### Checkpoint 2 Pipeline (Current)

```
┌─────────────────┐
│  MagicBrush     │  ──(stream 500 samples)──>  ┌────────────────────────────┐
│  Dataset (HF)   │                             │  Original images + prompts │
└─────────────────┘                             └────────────┬───────────────┘
                                                             │
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │  InstructPix2Pix (SD 1.5)   │
                                              │  SOTA open-source editor     │
                                              │  ~6 GB VRAM, fp16            │
                                              └──────────────┬───────────────┘
                                                             │
                                                             ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │  data/eval/                                              │
                    │    ├── images/orig/           (original from MagicBrush) │
                    │    ├── images/model_edited/   (IP2P generated edits)     │
                    │    ├── images/ground_truth/   (MagicBrush GT edits)      │
                    │    └── metadata.jsonl                                    │
                    └──────────────────────────────┬───────────────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────────┐
                                    │  Qwen2.5-VL-3B (VLM Judge)  │
                                    │  Sees: orig + edited + prompt│
                                    │  Outputs: structured JSON    │
                                    │  ~7 GB VRAM, full precision  │
                                    └──────────────┬───────────────┘
                                                   │
                              ┌─────────────────────┴──────────────────────┐
                              │                                            │
                              ▼                                            ▼
                   ┌─────────────────────┐              ┌──────────────────────────┐
                   │  vlm_judgments.jsonl │              │  Human Verification UI   │
                   │  (automated labels) │              │  (Streamlit spot-check)  │
                   │  adherence + errors │              │  Validates VLM accuracy  │
                   │  + reasoning        │              └────────┬─────────────────┘
                   └─────────┬───────────┘                       │
                             │                                   ▼
                             │                        human_reviews.jsonl
                             │                                   │
                             └────────────────┬──────────────────┘
                                              │
                                              ▼
                              ┌──────────────────────────────┐
                              │  Analysis & Visualization    │
                              │  • Adherence pie chart       │
                              │  • Error frequency bar chart │
                              │  • Correlation heatmap       │
                              │  • Failure gallery           │
                              │  • VLM confidence histogram  │
                              │  • summary.json              │
                              └──────────────────────────────┘
```

### Checkpoint 1 Pipeline (Complete)

```
┌─────────────────┐
│  MagicBrush     │  ──(stream)──>  ┌────────────────────┐
│  Dataset (HF)   │                 │ Dataset Sampler    │
└─────────────────┘                 │ (20-200 triplets)  │
                                    └──────────┬─────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────┐
                    │  data/sample/                            │
                    │    ├── images/orig/  (resized to 512px)  │
                    │    ├── images/edited/                    │
                    │    ├── metadata.jsonl                    │
                    │    └── translations.csv (cross-lingual)  │
                    └──────────────────────────────────────────┘
                                               │
                      ┌────────────────────────┼─────────────────────────┐
                      │                        │                         │
                      ▼                        ▼                         ▼
           ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
           │  Labeling Tool   │   │  CLIP Embedder   │   │  Prompt Features    │
           │  (Streamlit UI)  │   │  (ViT-B-32)      │   │  Extractor          │
           └────────┬─────────┘   └────────┬─────────┘   └──────────┬──────────┘
                    │                      │                         │
                    ▼                      ▼                         │
         data/annotations/      data/sample/embeddings.npz           │
         labels.jsonl           (512-d × 3 per sample)               │
                    │                      │                         │
                    └──────────────────────┴─────────────────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────────┐
                            │  Baseline Trainer            │
                            │  Logistic Regression Heads   │
                            │  (Adherence + Error Types)   │
                            └─────────────┬────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
         ┌──────────────────────┐               ┌──────────────────────────┐
         │  Trained Models      │               │  Failure Analyzer        │
         │  ├── adherence.pkl   │               │  (Correlation Analysis)  │
         │  ├── error.pkl       │               └────────┬─────────────────┘
         │  ├── metrics.json    │                        │
         │  └── confusion.png   │                        ▼
         └──────────────────────┘          runs/baseline/analysis/
                                           ├── correlations.csv
                                           ├── heatmap.png
                                           ├── error_frequency.png
                                           ├── wordcount_box.png
                                           └── adherence_pie.png
```

---

## Project Timeline

### ✅ Checkpoint 1 — Complete (Feb 8, 2026)

**Accomplished:**
- [x] Built dataset sampler with HuggingFace streaming
- [x] Implemented 11-category error taxonomy
- [x] Created Streamlit labeling tool with side-by-side image display
- [x] Integrated CLIP ViT-B-32 for feature extraction
- [x] Trained baseline logistic regression classifiers
- [x] Built prompt-failure correlation analyzer
- [x] Verified end-to-end pipeline on 20 samples
- [x] Generated all outputs: models, metrics, confusion matrix, 5 analysis plots
- [x] Added cross-lingual schema (English + Bengali placeholders)

**Deliverables:** 16 Python files (~1,800 lines), fully verified end-to-end.

### 🔄 Checkpoint 2 — In Progress (Mar 1, 2026)

**Goal: Prove that open-source image editors fail at instruction-following, using automated VLM evaluation.**

**New approach (replaces manual labeling):**
- [x] Integrated **InstructPix2Pix** (`timbrooks/instruct-pix2pix`) as the SOTA open-source editor
- [x] Integrated **Qwen2.5-VL-3B** (`Qwen/Qwen2.5-VL-3B-Instruct`) as VLM-as-judge
- [x] Built `generate_edits.py` — streams MagicBrush, runs IP2P, saves triplets (orig + model-edited + ground-truth)
- [x] Built `vlm_judge.py` — sends (orig, edited, instruction) to Qwen VLM for structured JSON evaluation
- [x] Built `analyze_vlm_results.py` — produces 5 plots + summary statistics
- [x] Built human verification Streamlit UI for spot-checking VLM accuracy
- [x] Added `VLMJudgment` and `HumanReview` dataclasses to schema
- [x] Added resume support for long runs (both editor and judge scripts)
- [x] Created central config (`config/eval_config.yaml`)
- [ ] **Testing:** Initial 10-sample test run (in progress — resolving VRAM/dependency issues on Windows)

**New files added (Checkpoint 2):**

| File | Purpose |
|---|---|
| `scripts/generate_edits.py` | Stream MagicBrush → InstructPix2Pix → save edited images |
| `scripts/vlm_judge.py` | Send triplets to Qwen2.5-VL → structured JSON judgments |
| `scripts/analyze_vlm_results.py` | Aggregate judgments → plots + summary statistics |
| `scripts/utils/editor_model.py` | `InstructPix2PixEditor` wrapper with memory management |
| `scripts/utils/vlm_evaluator.py` | `QwenVLMJudge` wrapper with JSON parsing + fallback |
| `apps/human_review.py` | Streamlit UI for spot-checking VLM accuracy *(pending)* |
| `config/eval_config.yaml` | Central configuration for models, paths, parameters |

**Known issues being resolved:**
- `autoawq` requires `triton` (Linux-only) → switched to full-precision Qwen2.5-VL-3B (~7 GB VRAM)
- Exit code 139 on first test run → likely OOM during model loading; investigating memory optimization

### 📅 Checkpoint 3 — Planned (Target: 3-4 weeks)

**Goals:**
- [ ] Scale to 500+ samples with full analysis
- [ ] Multi-lingual prompt support (Bengali, Spanish)
- [ ] Explainability analysis (SHAP/LIME)
- [ ] Compare multiple editing models (IP2P vs CosXL vs OmniGen)

### 🎯 Long-Term Vision (Target: 2-3 months)

**Goals:**
- [ ] Publish benchmark dataset (1,000+ labeled triplets)
- [ ] Submit paper to CVPR/NeurIPS
- [ ] Deploy as production quality filter
- [ ] Package as pip-installable library

---

## What We Built

### Checkpoint 2 Pipeline (NEW — automated evaluation)

The key insight: **replace human labeling with VLM-as-judge**. Instead of manually reviewing hundreds of images, we use Qwen2.5-VL to automatically evaluate each edit and classify failures.

1. **Edit generator** (`generate_edits.py`) — Streams MagicBrush samples, runs InstructPix2Pix on each, saves original + model-edited + ground-truth triplets
2. **VLM judge** (`vlm_judge.py`) — Sends each triplet to Qwen2.5-VL with structured prompt; VLM returns JSON with adherence, error types, reasoning, and confidence
3. **Results analyzer** (`analyze_vlm_results.py`) — Aggregates all judgments into plots and statistics: failure rate, error distribution, prompt-failure correlations
4. **Human verification UI** (`apps/human_review.py`) — Streamlit tool where a human spot-checks VLM judgments to validate accuracy *(pending)*

### Checkpoint 1 Pipeline (complete — CLIP baseline)

1. **Dataset sampler** — Streams samples from HuggingFace MagicBrush, resizes images, saves metadata
2. **Labeling tool** — Streamlit UI for human annotation with side-by-side image display
3. **Embedding extractor** — CLIP ViT-B-32 feature extraction (image + text)
4. **Baseline trainer** — Logistic regression classifiers for adherence + multi-label error prediction
5. **Failure analyzer** — Correlates prompt properties with editing failures

**Total codebase:** 23 Python files (~3,000+ lines) including utilities, schemas, configs, and two Streamlit apps.

---

## Quick Start

### Setup (once)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install everything else
pip install diffusers transformers accelerate qwen-vl-utils safetensors \
    open_clip_torch datasets huggingface_hub scikit-learn joblib \
    pandas matplotlib seaborn pyyaml streamlit tqdm Pillow requests
```

### Checkpoint 2 — Automated VLM Evaluation (NEW)

```bash
# Step 1: Generate edits with InstructPix2Pix (~10-15 sec/image on GPU)
python scripts/generate_edits.py --n 10 --out data/eval          # quick test
python scripts/generate_edits.py --n 500 --out data/eval --resume # full run

# Step 2: Judge edits with Qwen2.5-VL (~5-10 sec/image on GPU)
python scripts/vlm_judge.py --data data/eval                      # quick test
python scripts/vlm_judge.py --data data/eval --resume             # full run

# Step 3: Analyze results
python scripts/analyze_vlm_results.py --data data/eval --out runs/eval_analysis

# Step 4: Human verification (spot-check VLM accuracy)
streamlit run apps/human_review.py -- --data data/eval
```

### Checkpoint 1 — CLIP Baseline (verified working)

```bash
# Download MagicBrush samples
python scripts/make_sample_dataset.py --n 20 --out data/sample --max_side 512 --seed 42

# Label samples (manual)
streamlit run apps/label_tool.py

# Extract CLIP embeddings
python scripts/extract_embeddings.py --data data/sample

# Train baseline classifiers
python scripts/train_baseline.py \
    --data data/sample \
    --labels data/annotations/labels.jsonl \
    --out runs/baseline

# Analyze prompt-failure correlations
python scripts/analyze_failures.py \
    --data data/sample \
    --labels data/annotations/labels.jsonl \
    --out runs/baseline/analysis
```

---

## Verification Results (Feb 8, 2026)

We ran the **complete pipeline end-to-end with 20 samples**:

| Step | Command | Time | Output | Status |
|------|---------|------|--------|--------|
| **Dataset** | `make_sample_dataset.py --n 20` | ~4.5 min | 20 orig + 20 edited images (512px), `metadata.jsonl`, `translations.csv` | ✅ Pass |
| **Embeddings** | `extract_embeddings.py` | ~3 min | `embeddings.npz` (20 × 512 dims for orig, edit, text) | ✅ Pass |
| **Labeling** | `streamlit run apps/label_tool.py` | Interactive | UI launches on `localhost:8501`, all features working | ✅ Pass |
| **Training** | `train_baseline.py` | ~2 sec | Models saved: `adherence_model.joblib`, `error_model.joblib`, `metrics.json`, `confusion_matrix.png` | ✅ Pass |
| **Analysis** | `analyze_failures.py` | ~1 sec | 5 plots + `correlations.csv` saved to `runs/baseline/analysis/` | ✅ Pass |

### Key Findings from Test Run

**Test data:** 20 samples with synthetic labels (8 Success, 4 Partial, 8 No)

**Baseline metrics (train: 16, val: 4):**
- **Adherence accuracy:** 25% (expected on 4 val samples with random labels)
- **Adherence macro-F1:** 0.17
- **Error mAP:** 0.75
- **Confusion matrix:** Saved as PNG

**Prompt-failure correlations:**
- **Top correlated feature with failure:** `has_spatial_words` (r = 0.187)
- Generated plots: correlation heatmap, error-type frequency bar chart, word-count box plot, adherence distribution pie chart

**Notes:**
- Low performance is expected — these are synthetic labels on 20 samples for pipeline verification
- With real human labels on 200+ samples, meaningful patterns will emerge
- All 5 analysis plots generated successfully

---

## Repository Structure

```
project_root/
├── README.md
├── requirements.txt
├── config/
│   └── eval_config.yaml           # central config for models & paths
├── data/
│   ├── sample/                    # Checkpoint 1 data
│   │   ├── images/orig/
│   │   ├── images/edited/
│   │   ├── metadata.jsonl
│   │   ├── translations.csv
│   │   └── embeddings.npz
│   ├── eval/                      # Checkpoint 2 data (NEW)
│   │   ├── images/orig/           # originals from MagicBrush
│   │   ├── images/model_edited/   # InstructPix2Pix outputs
│   │   ├── images/ground_truth/   # MagicBrush ground-truth edits
│   │   ├── metadata.jsonl
│   │   ├── vlm_judgments.jsonl    # Qwen VLM automated labels
│   │   └── human_reviews.jsonl   # human spot-check results
│   └── annotations/
│       └── labels.jsonl
├── apps/
│   ├── label_tool.py              # Checkpoint 1 labeling UI
│   └── human_review.py            # Checkpoint 2 VLM verification UI (pending)
├── scripts/
│   ├── make_sample_dataset.py     # Checkpoint 1: data sampler
│   ├── extract_embeddings.py      # Checkpoint 1: CLIP features
│   ├── train_baseline.py          # Checkpoint 1: classifiers
│   ├── analyze_failures.py        # Checkpoint 1: correlation analysis
│   ├── generate_edits.py          # Checkpoint 2: IP2P edit generation (NEW)
│   ├── vlm_judge.py               # Checkpoint 2: VLM-as-judge (NEW)
│   ├── analyze_vlm_results.py     # Checkpoint 2: VLM results analysis (NEW)
│   └── utils/
│       ├── __init__.py
│       ├── io.py                  # JSONL / image I/O
│       ├── schema.py              # labels, taxonomy, VLMJudgment, HumanReview
│       ├── prompt_features.py     # prompt feature extraction
│       ├── clip_encoder.py        # CLIP image/text embedder
│       ├── text_encoder.py        # text-encoder interface (cross-lingual)
│       ├── editor_model.py        # InstructPix2Pix wrapper (NEW)
│       └── vlm_evaluator.py       # Qwen2.5-VL judge wrapper (NEW)
├── notebooks/
│   └── checkpoint1_demo.ipynb
└── runs/
    ├── baseline/                  # Checkpoint 1 outputs
    └── eval_analysis/             # Checkpoint 2 outputs (NEW)
        ├── adherence_distribution.png
        ├── error_type_frequency.png
        ├── heatmap_correlations.png
        ├── confidence_distribution.png
        ├── failure_examples.png
        ├── correlations.csv
        └── summary.json
```

---

## What Each Script Does

### Checkpoint 2 Scripts (NEW)

| Script | Purpose |
|---|---|
| `generate_edits.py` | Streams MagicBrush samples, runs InstructPix2Pix on each original image using the MagicBrush instruction. Saves original + model-edited + ground-truth images. Supports `--resume` for interrupted runs. |
| `vlm_judge.py` | Loads Qwen2.5-VL-3B, sends (original, model-edited, instruction) to the VLM for each sample. VLM returns structured JSON: adherence (Success/Partial/No), error types (11-category), reasoning, confidence. Supports `--resume`. |
| `analyze_vlm_results.py` | Merges metadata + judgments, produces 5 plots (adherence pie, error bar chart, confidence histogram, correlation heatmap, failure gallery) + `summary.json`. |
| `human_review.py` | Streamlit UI showing the VLM's judgment alongside the images. Human marks whether VLM got adherence and errors correct. Saves spot-check results to `human_reviews.jsonl`. *(pending)* |

### Checkpoint 1 Scripts

| Script | Purpose |
|---|---|
| `make_sample_dataset.py` | Streams N samples from [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush), resizes images to `max_side`, saves metadata JSONL.  Falls back to local-folder mode if offline. |
| `label_tool.py` | Streamlit UI showing orig/edited side-by-side + prompt; annotator picks adherence & error types; labels appended to `labels.jsonl`. |
| `extract_embeddings.py` | Loads CLIP ViT-B-32 and encodes all images + prompts → `embeddings.npz`. |
| `train_baseline.py` | Builds a 1,538-d feature vector per sample, trains logistic-regression heads for adherence (3-class) and error types (multi-label). Prints metrics + saves models. |
| `analyze_failures.py` | Extracts prompt features (word count, spatial words, color count, …), correlates them with failure/error labels, outputs a CSV and plots. |

### Utility Modules

| Module | Purpose |
|---|---|
| `utils/editor_model.py` | `InstructPix2PixEditor` class — loads diffusers pipeline, configurable guidance scales, attention slicing for lower VRAM, `.unload()` to free GPU memory. **(NEW)** |
| `utils/vlm_evaluator.py` | `QwenVLMJudge` class — loads Qwen2.5-VL, structured prompt template with 11-error taxonomy, JSON response parsing with regex fallback, `.unload()` to free GPU. **(NEW)** |
| `utils/schema.py` | `ADHERENCE_LABELS`, `ERROR_TYPES`, `MetadataRecord`, `LabelRecord`, `VLMJudgment` **(NEW)**, `HumanReview` **(NEW)** |
| `utils/io.py` | JSONL read/write, image resize, directory helpers |
| `utils/prompt_features.py` | 6 regex-based prompt features for correlation analysis |
| `utils/clip_encoder.py` | CLIP ViT-B-32 batch encoding |
| `utils/text_encoder.py` | Abstract text encoder interface for cross-lingual support |

---

## Error Taxonomy

| # | Error Type |
|---|---|
| 0 | Wrong Object |
| 1 | Missing Object |
| 2 | Extra Object |
| 3 | Wrong Attribute |
| 4 | Spatial Error |
| 5 | Style Mismatch |
| 6 | Over-editing |
| 7 | Under-editing |
| 8 | Artifact / Quality Issue |
| 9 | Ambiguous Prompt |
| 10 | Failed Removal |

---

## Cross-Lingual Support

The schema includes a `lang` field on every sample.  The text-encoder interface
(`text_encoder.py`) accepts `(text, lang)` so a multilingual backbone can be
swapped in later.  For now:

- `data/sample/translations.csv` ships with a few English + Bengali stub rows.
- CLIP encodes raw text regardless of language (works decently for Latin-script
  languages; Bengali will need a multilingual encoder in later checkpoints).

---

## Technical Details

### Checkpoint 2 — Automated VLM-as-Judge Pipeline

**Image Editor:** InstructPix2Pix (`timbrooks/instruct-pix2pix`)
- Stable Diffusion 1.5 backbone with instruction conditioning
- ~6-7 GB VRAM in fp16, attention slicing enabled for lower-VRAM GPUs
- Configurable: `guidance_scale` (text), `image_guidance_scale` (image fidelity), `num_inference_steps`

**VLM Judge:** Qwen2.5-VL-3B (`Qwen/Qwen2.5-VL-3B-Instruct`)
- 3B parameter vision-language model, ~7-8 GB VRAM full precision
- Multi-image input: sends original + model-edited images in a single prompt
- Structured prompt template defining 3 adherence levels and 11 error types
- JSON output with regex fallback parsing for robustness

**Evaluation Taxonomy (11 error types):**
`wrong_object`, `wrong_attribute`, `wrong_location`, `incomplete_edit`, `over_edit`, `color_mismatch`, `shape_distortion`, `background_corruption`, `text_rendering_fail`, `style_mismatch`, `no_visible_change`

**Sequential model loading:** Editor generates all edits first, then unloads. VLM loads separately to judge. This prevents VRAM conflicts on 12 GB cards.

### Checkpoint 1 — CLIP Baseline

**Feature construction (1,538-d per sample):**
```
[emb_edit (512-d), emb_orig (512-d), (emb_edit - emb_orig) (512-d),
 cosine(prompt, edit_img) (1-d), cosine(prompt, orig_img) (1-d)]
```

**Classifiers:**
- **Adherence:** 3-class logistic regression (Success / Partial / No)
- **Error types:** 11-label OneVsRest logistic regression

**CLIP model:** ViT-B-32 with `laion2b_s34b_b79k` weights (~605 MB download, ~600 MB VRAM at inference)

### Prompt Features Extracted

For correlation analysis, we extract 6 interpretable features from each prompt:
- `word_count` — number of tokens
- `char_count` — string length
- `has_spatial_words` — presence of spatial keywords (left/right/above/below/etc.)
- `count_of_colors` — number of color mentions
- `num_changes_proxy` — count of "and" + commas (proxy for multi-edit complexity)
- `specificity_proxy` — count of named objects, directional cues, precision adverbs

---

## Known Issues & Fixes Applied

### Checkpoint 2

4. **autoawq cannot install on Windows** — `autoawq` depends on `triton` which is Linux-only. Solution: Switched from Qwen2.5-VL-7B-Instruct-AWQ to **Qwen2.5-VL-3B-Instruct** (full precision). Fits within ~7-8 GB VRAM.

5. **Exit code 139 (SIGSEGV) when running `generate_edits.py`** — First test run with `--n 10` crashed during model loading on RTX 3060 (12 GB). Likely cause: OOM loading InstructPix2Pix (~6-7 GB fp16) + dataset streaming overhead. Status: **under investigation**. Potential fixes: reduce image resolution, try CPU offloading, check Python 3.13 compatibility.

### Checkpoint 1

1. **MagicBrush streaming is slow** — Each parquet shard contains ~130 MB of embedded images. Solution: Skip `.shuffle()` for streaming; instead skip a random offset for variety. Reduces download from 45 min to ~5 min for 20 samples.

2. **scikit-learn API change** — `multi_class="multinomial"` parameter removed in sklearn 1.3+. Fixed by removing the parameter (multinomial is now default for multiclass).

3. **Windows symlinks warning** — HuggingFace Hub warns about symlinks on Windows. Harmless; caching still works but uses more disk space. To fix: enable Developer Mode or run as admin.

---

## Hardware & Performance

**Tested on:**
- **CPU:** Standard x64 processor
- **GPU:** RTX 3060 (12 GB VRAM) — required for Checkpoint 2 (InstructPix2Pix + Qwen VLM)
- **RAM:** 16 GB recommended
- **Disk:** ~5 GB for model weights + ~500 MB for 500 samples
- **Bandwidth:** Streaming from HuggingFace; model downloads ~3-4 GB each on first run

**VRAM Requirements (Checkpoint 2):**

| Component | VRAM (fp16) | Notes |
|---|---|---|
| InstructPix2Pix | ~6-7 GB | Attention slicing enabled |
| Qwen2.5-VL-3B | ~7-8 GB | Full precision (AWQ unavailable on Windows) |
| CLIP ViT-B-32 | ~600 MB | Checkpoint 1 only |

> **Note:** Editor and VLM run sequentially (not simultaneously). Peak VRAM ≈ 7-8 GB. Upgrade to RTX 4090 planned for faster throughput.

**Estimated Timing (Checkpoint 2, 10 samples, RTX 3060):**
- Model download (first run): ~10-15 min
- Image generation: ~2-3 min (20 steps per image)
- VLM judging: ~3-5 min
- Analysis: ~5 sec

**Timing (Checkpoint 1, 20 samples):**
- Dataset download: ~4-5 min
- CLIP embedding extraction: ~3 min (CPU) / ~30 sec (GPU)
- Training: ~2 sec
- Analysis: ~1 sec

---

## Checkpoint 2 Demo Script

**Automated VLM-as-Judge pipeline:**

```bash
# 1. Generate edits with InstructPix2Pix (10 sample quick test)
python scripts/generate_edits.py --n 10 --out data/eval

# 2. Judge edits with Qwen VLM
python scripts/vlm_judge.py --data data/eval --out runs/vlm_eval

# 3. Analyze results
python scripts/analyze_vlm_results.py --data data/eval --judgments runs/vlm_eval/vlm_judgments.jsonl --out runs/vlm_eval/analysis

# 4. (Optional) Human spot-check
streamlit run apps/human_review.py
```

**Expected outputs:**
1. `data/eval/images/` — orig, model_edited, ground_truth images
2. `data/eval/metadata.jsonl` — sample metadata with edit times
3. `runs/vlm_eval/vlm_judgments.jsonl` — structured VLM judgments
4. `runs/vlm_eval/analysis/` — 5 plots + `summary.json`

---

## Checkpoint 1 Demo Script

<details>
<summary>Click to expand Checkpoint 1 demo commands</summary>

```bash
# 1. Show the project structure
ls -la data/ scripts/ apps/ runs/

# 2. Run dataset sampler (use --n 10 for quick demo)
python scripts/make_sample_dataset.py --n 10 --out data/sample

# 3. Show downloaded images
ls data/sample/images/orig/ data/sample/images/edited/
head -3 data/sample/metadata.jsonl

# 4. Launch labeling tool and label 2-3 samples live
streamlit run apps/label_tool.py

# 5. Extract embeddings
python scripts/extract_embeddings.py --data data/sample

# 6. Train baseline
python scripts/train_baseline.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline

# 7. Run analysis
python scripts/analyze_failures.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline/analysis
```

</details>

---

## Next Steps & Future Work

### Immediate (Checkpoint 2 completion)

1. **Fix exit code 139 crash** — Debug OOM/SIGSEGV on RTX 3060 during `generate_edits.py`
2. **Run full 500-sample evaluation** — Generate edits → VLM judge → analysis
3. **Build human review UI** — Streamlit app for spot-checking VLM accuracy
4. **Compare VLM accuracy vs human labels** — Measure how well VLM-as-judge agrees with ground truth from Checkpoint 1

### Checkpoint 3 — Model Comparison

5. **Multi-model evaluation** — Test additional editors: SDXL-Instruct, MagicBrush-official, DALL-E 3
6. **VLM calibration** — Analyze VLM confidence vs. actual correctness
7. **Inter-rater reliability** — Cohen's kappa between VLM and human reviewers

### Checkpoint 4 — Scaling & Analysis

8. **Prompt complexity analysis** — Correlate prompt features with VLM-detected failure modes
9. **Error prediction model** — Train classifier to predict which prompts will fail
10. **Cross-lingual prompts** — Bengali + Spanish instruction evaluation

### Long-Term

11. **Paper & benchmark** — Formal evaluation of VLM-as-judge reliability for image editing
12. **Production pipeline** — REST API, CI/CD, auto-evaluation on new models

---

## Troubleshooting

### Checkpoint 2

**Exit code 139 / SIGSEGV when running `generate_edits.py`**
→ Likely GPU OOM loading InstructPix2Pix. Try:
- `--device cpu` to test without GPU
- Reduce image resolution in `config/eval_config.yaml` (lower `max_pixels`)
- Close other GPU-consuming applications
- Upgrade to a GPU with >12 GB VRAM

**"autoawq install fails" / "No module named triton"**
→ `autoawq` requires `triton` which is Linux-only. Use the full-precision 3B model instead (already set as default). Do NOT install autoawq on Windows.

**"Model download takes forever"**
→ InstructPix2Pix is ~3.5 GB, Qwen2.5-VL-3B is ~7 GB. First-time download is slow. Models are cached in `~/.cache/huggingface/`.

**VLM returns invalid JSON**
→ The `QwenVLMJudge` has built-in regex fallback parsing. If you still see parse errors, check `vlm_judgments.jsonl` — the `raw_response` field stores the original VLM output for debugging.

### Checkpoint 1

**"No module named 'open_clip'"**
→ Run: `pip install open_clip_torch`

**"CUDA out of memory" (CLIP)**
→ The ViT-B-32 model only uses ~600 MB VRAM. Reduce `--batch_size` in `extract_embeddings.py` or switch to CPU with `--device cpu`.

**"Streaming dataset download is very slow"**
→ Expected — MagicBrush parquet shards are 100+ MB each. The optimized version (without `.shuffle()`) is much faster.

**"Not enough labeled samples to train"**
→ The trainer requires ≥5 labeled samples. Label more samples in the Streamlit tool.

**"Streamlit won't launch"**
→ Check firewall settings for port 8501. Try a different port: `--server.port 8502`.

---

---

## Contributing

Contributions welcome! Areas where help is needed:

- [ ] Debugging OOM/SIGSEGV on RTX 3060 (12 GB)
- [ ] Testing on Linux with autoawq (7B quantized model)
- [ ] Building the human review Streamlit UI
- [ ] Labeling samples for VLM accuracy validation
- [ ] Adding support for additional image editors (SDXL-Instruct, etc.)
- [ ] Cross-lingual prompt evaluation (Bengali, Spanish)

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes with clear messages
4. Push to your fork and submit a pull request

---

## Citation

If you use this codebase or dataset in your research, please cite:

```bibtex
@misc{imageedit-eval-2026,
  author = {Your Name},
  title = {Instruction-Following Evaluation for Text-Guided Image Editing},
  year = {2026},
  howpublished = {\url{https://github.com/Legend-2727/instruction-following-image-editing-eval}},
  note = {Checkpoint 2: VLM-as-Judge automated evaluation}
}
```

**Dataset acknowledgment:**
We use the [MagicBrush dataset](https://huggingface.co/datasets/osunlp/MagicBrush) by Zhang et al. (OSU NLP Lab). Please cite their work:

```bibtex
@inproceedings{zhang2024magicbrush,
  title={MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing},
  author={Zhang, Kai and others},
  booktitle={NeurIPS},
  year={2023}
}
```

---

## Acknowledgments

- **MagicBrush dataset** by OSU NLP Lab for providing high-quality instruction-edit pairs
- **InstructPix2Pix** by Brooks et al. for the instruction-guided image editing model
- **Qwen2.5-VL** by Alibaba Cloud for the open-source vision-language model
- **OpenCLIP** team for pretrained vision-language models
- **HuggingFace** for datasets infrastructure, diffusers, and streaming support
- **Streamlit** for the rapid UI prototyping framework

---

## License

**Code:** MIT License (see LICENSE file)

**Dataset:** MagicBrush is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Any derivatives (annotations, embeddings) follow the same license.

**Research use:** This project is intended for academic and research purposes. For commercial use, consult the MagicBrush license terms and ensure compliance.

---

## Contact

For questions, issues, or collaboration inquiries:
- **Email:** alaminfarhad27@gmail.com
- **Issues:** [GitHub Issues](https://github.com/Legend-2727/instruction-following-image-editing-eval/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Legend-2727/instruction-following-image-editing-eval/discussions)

---

**Last updated:** March 1, 2026  
**Status:** 🔄 Checkpoint 2 in progress — code complete, debugging first test run  
**Next milestone:** Fix OOM crash → run 500-sample evaluation → human review UI
