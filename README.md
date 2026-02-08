# Checkpoint 1 — Instruction-Following Evaluation for Text-Guided Image Editing

**A complete ML pipeline for evaluating instruction-following failures in text-guided image editing models.**

This project evaluates how well text-guided image-editing models follow natural language instructions. Each sample is a triplet **(original image, edit prompt, edited image)** from the MagicBrush dataset. Human annotators label **adherence** (Success / Partial / No) and **error types** (11-category multi-label taxonomy), and a CLIP-based baseline classifier predicts both from learned embeddings.

**Status:** ✅ **Fully implemented and verified end-to-end** (Feb 8, 2026)

---

## Overview

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

**Deliverables:**
- 16 Python files (~1,800 lines of code)
- Complete documentation (this README)
- Working demo ready for presentation
- All dependencies pinned in `requirements.txt`

### 🔄 Checkpoint 2 — In Progress (Target: 1-2 weeks)

**Goals:**
- [ ] Collect 200-500 real human labels (est. 10-20 hours)
- [ ] Measure inter-annotator agreement
- [ ] Retrain on real labels and analyze results
- [ ] Improve baseline model (test MLP heads)
- [ ] Add more prompt features (semantic complexity)

### 📅 Checkpoint 3 — Planned (Target: 3-4 weeks)

**Goals:**
- [ ] Multi-lingual prompt support (Bengali, Spanish)
- [ ] Explainability analysis (SHAP/LIME)
- [ ] Active learning for efficient labeling
- [ ] Model comparison framework (compare multiple editing models)

### 🎯 Long-Term Vision (Target: 2-3 months)

**Goals:**
- [ ] Publish benchmark dataset (1,000+ labeled triplets)
- [ ] Submit paper to CVPR/NeurIPS
- [ ] Deploy as production quality filter
- [ ] Package as pip-installable library

---

## What We Built

A complete research pipeline with **5 runnable scripts** + **interactive labeling tool**:

1. **Dataset sampler** — Streams samples from HuggingFace MagicBrush, resizes images, saves metadata
2. **Labeling tool** — Streamlit UI for human annotation with side-by-side image display
3. **Embedding extractor** — CLIP ViT-B-32 feature extraction (image + text)
4. **Baseline trainer** — Logistic regression classifiers for adherence + multi-label error prediction
5. **Failure analyzer** — Correlates prompt properties with editing failures

**Total codebase:** 16 files (~1,800 lines) including utilities, schemas, and configs.

---

## Quick Start (Verified Working)

```bash
# 0. Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows Git Bash
# OR: .venv\Scripts\activate   # On Windows CMD/PowerShell

# 1. Install dependencies (~2-3 min, downloads PyTorch + CLIP)
pip install --upgrade pip
pip install -r requirements.txt

# 2. Download sample dataset (20-200 triplets from MagicBrush)
#    Note: 20 samples = ~5 min, 200 samples = ~45 min due to large parquet shards
python scripts/make_sample_dataset.py --n 20 --out data/sample --max_side 512 --seed 42

# 3. Launch the labeling tool (opens in browser at localhost:8501)
streamlit run apps/label_tool.py
#    → Label as many samples as you want, then Ctrl-C to stop

# 4. Extract CLIP embeddings (~1-2 min on CPU, faster on GPU)
python scripts/extract_embeddings.py --data data/sample

# 5. Train baseline classifiers
python scripts/train_baseline.py \
    --data data/sample \
    --labels data/annotations/labels.jsonl \
    --out runs/baseline

# 6. Run prompt-failure correlation analysis
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
├── data/
│   ├── sample/
│   │   ├── images/orig/          # resized original images
│   │   ├── images/edited/        # resized edited images
│   │   ├── metadata.jsonl        # id, prompt, paths, lang
│   │   ├── translations.csv      # (optional) cross-lingual prompts
│   │   └── embeddings.npz        # (generated) cached CLIP embeddings
│   └── annotations/
│       └── labels.jsonl           # (generated) human labels
├── apps/
│   └── label_tool.py              # Streamlit labeling UI
├── scripts/
│   ├── make_sample_dataset.py     # data sampler (streaming / local)
│   ├── extract_embeddings.py      # CLIP feature extraction
│   ├── train_baseline.py          # adherence + error classifiers
│   ├── analyze_failures.py        # prompt–failure correlation analysis
│   └── utils/
│       ├── __init__.py
│       ├── io.py                  # JSONL / image I/O
│       ├── schema.py              # labels, taxonomy, dataclasses
│       ├── prompt_features.py     # prompt feature extraction
│       ├── clip_encoder.py        # CLIP image/text embedder
│       └── text_encoder.py        # text-encoder interface (cross-lingual)
├── notebooks/
│   └── checkpoint1_demo.ipynb     # (optional) interactive demo
└── runs/
    └── baseline/                  # (generated) models + metrics
        ├── adherence_model.joblib
        ├── error_model.joblib
        ├── metrics.json
        ├── confusion_matrix.png
        ├── classification_report.txt
        └── analysis/
            ├── correlations.csv
            ├── heatmap_correlations.png
            ├── error_type_frequency.png
            ├── wordcount_by_adherence.png
            └── adherence_distribution.png
```

---

## What Each Script Does

| Script | Purpose |
|---|---|
| `make_sample_dataset.py` | Streams N samples from [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush), resizes images to `max_side`, saves metadata JSONL.  Falls back to local-folder mode if offline. |
| `label_tool.py` | Streamlit UI showing orig/edited side-by-side + prompt; annotator picks adherence & error types; labels appended to `labels.jsonl`. |
| `extract_embeddings.py` | Loads CLIP ViT-B-32 and encodes all images + prompts → `embeddings.npz`. |
| `train_baseline.py` | Builds a 1 538-d feature vector per sample, trains logistic-regression heads for adherence (3-class) and error types (multi-label). Prints metrics + saves models. |
| `analyze_failures.py` | Extracts prompt features (word count, spatial words, color count, …), correlates them with failure/error labels, outputs a CSV and plots. |

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

### Architecture

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

1. **MagicBrush streaming is slow** — Each parquet shard contains ~130 MB of embedded images. Solution: Skip `.shuffle()` for streaming; instead skip a random offset for variety. Reduces download from 45 min to ~5 min for 20 samples.

2. **scikit-learn API change** — `multi_class="multinomial"` parameter removed in sklearn 1.3+. Fixed by removing the parameter (multinomial is now default for multiclass).

3. **Windows symlinks warning** — HuggingFace Hub warns about symlinks on Windows. Harmless; caching still works but uses more disk space. To fix: enable Developer Mode or run as admin.

---

## Hardware & Performance

**Tested on:**
- **CPU:** Standard x64 processor
- **GPU:** RTX 3060 (12 GB VRAM) — optional, CLIP runs fine on CPU
- **RAM:** 16 GB recommended
- **Disk:** ~200 MB for 20 samples (images + embeddings + models)
- **Bandwidth:** ~50-100 MB download for 20 samples (images embedded in parquet)

**Timing (20 samples):**
- Dataset download: ~4-5 min
- CLIP embedding extraction: ~3 min (CPU) / ~30 sec (GPU)
- Training: ~2 sec
- Analysis: ~1 sec

For 200 samples, expect 10× longer for dataset download (~45 min) and embedding extraction (~30 min CPU / ~5 min GPU).

---

## Checkpoint 1 Demo Script (What We Showed)

**Live demonstration flow:**

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
# (Open browser, demonstrate UI, save labels, then Ctrl-C)

# 5. Extract embeddings
python scripts/extract_embeddings.py --data data/sample

# 6. Show embeddings file
python -c "import numpy as np; d=np.load('data/sample/embeddings.npz', allow_pickle=True); print('Shapes:', {k: v.shape for k, v in d.items()})"

# 7. Train baseline
python scripts/train_baseline.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline
# (Show printed metrics and confusion matrix PNG)

# 8. Run analysis
python scripts/analyze_failures.py --data data/sample --labels data/annotations/labels.jsonl --out runs/baseline/analysis

# 9. Show all outputs
ls runs/baseline/
ls runs/baseline/analysis/
cat runs/baseline/metrics.json
```

**Outputs to display:**
1. ✅ Dataset folder with images and metadata
2. ✅ Labeling tool UI (running in browser)
3. ✅ Training metrics printed to console
4. ✅ Confusion matrix PNG
5. ✅ 5 analysis plots (heatmap, error frequency, box plot, pie chart)
6. ✅ Cross-lingual translations CSV

---

## Next Steps & Future Work

### Immediate Next Tasks (Checkpoint 2)

1. **Collect real human labels**
   - Goal: Label 200-500 samples with multiple annotators
   - Track inter-annotator agreement (Cohen's kappa)
   - Create labeling guidelines document
   - Time estimate: 10-20 hours of annotation work

2. **Improve baseline model**
   - Try non-linear heads (MLP with 1-2 hidden layers)
   - Experiment with fine-tuning CLIP on this task
   - Add data augmentation (random crops, flips for images)
   - Ensemble multiple CLIP backbones (ViT-B-32 + ViT-L-14)

3. **Expand prompt feature analysis**
   - Add semantic complexity metrics (parse tree depth, entity count)
   - Measure prompt ambiguity via paraphrase similarity
   - Analyze failure modes by prompt category (object add/remove/modify/style change)

4. **Add more datasets**
   - InstructPix2Pix dataset samples
   - IP2P-generated samples (for comparing model variants)
   - Multi-turn editing sequences

### Medium-Term Goals (Checkpoints 3-4)

5. **Multi-lingual support**
   - Integrate mBERT or XLM-RoBERTa for prompt encoding
   - Collect English + Bengali + Spanish prompts
   - Test cross-lingual transfer (train on English, test on Bengali)
   - Build translation pipeline for existing English prompts

6. **Error prediction explainability**
   - SHAP/LIME analysis on trained classifiers
   - Visualize which image regions correlate with each error type
   - Build an attention map overlay tool in Streamlit

7. **Active learning for labeling**
   - Train initial model on 100 labels
   - Use uncertainty sampling to prioritize next samples to label
   - Measure label efficiency (performance vs. # labels curve)

8. **Model comparison framework**
   - Evaluate InstructPix2Pix vs. MagicBrush-official vs. DALL-E vs. Stable Diffusion
   - Compute adherence/error rates per model
   - Statistical significance testing (McNemar's test)

### Long-Term Vision (Future Checkpoints)

9. **Automated failure detection in production**
   - Deploy classifier as inference-time quality filter
   - Flag low-confidence edits for human review
   - Build feedback loop: user corrections → retrain classifier

10. **Prompt optimization**
    - Given an edit goal, auto-generate prompts with high predicted adherence
    - Iterative refinement: suggest prompt improvements based on error predictions

11. **Full paper & benchmark**
    - Formalize the taxonomy with inter-rater reliability study
    - Publish dataset of 1,000+ labeled triplets
    - Benchmark SOTA vision-language models
    - Submit to CVPR/ICCV/ECCV or NeurIPS Datasets track

12. **Tool productization**
    - Package as pip-installable library
    - REST API for batch inference
    - Integration with image editing platforms (Canva, Photoshop plugins)

---

## Troubleshooting

**"No module named 'open_clip'"**
→ Run: `pip install open_clip_torch`

**"CUDA out of memory"**
→ The ViT-B-32 model only uses ~600 MB VRAM. If you see this, reduce `--batch_size` in `extract_embeddings.py` or switch to CPU with `--device cpu`.

**"Streaming dataset download is very slow"**
→ Expected behavior — MagicBrush parquet shards are 100+ MB each with embedded images. The optimized version (without `.shuffle()`) is much faster. Alternatively, download the full dataset once and use `--local_orig` / `--local_edited` mode.

**"Not enough labeled samples to train"**
→ The trainer requires ≥5 labeled samples. If you see this warning, label more samples in the Streamlit tool or generate synthetic labels for testing (see the verification script).

**"Streamlit won't launch"**
→ Check firewall settings for port 8501. Try `streamlit run apps/label_tool.py --server.port 8502` to use a different port.

---

---

## Contributing

Contributions welcome! Areas where help is needed:

- [ ] Labeling samples (especially non-English prompts)
- [ ] Adding new error types to the taxonomy
- [ ] Improving the baseline model architecture
- [ ] Building a labeling guidelines document
- [ ] Testing on other image editing datasets

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
  howpublished = {\url{https://github.com/yourusername/image-edit-eval}},
  note = {Checkpoint 1 implementation}
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
- **OpenCLIP** team for pretrained vision-language models
- **HuggingFace** for datasets infrastructure and streaming support
- **Streamlit** for the rapid UI prototyping framework

---

## License

**Code:** MIT License (see LICENSE file)

**Dataset:** MagicBrush is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Any derivatives (annotations, embeddings) follow the same license.

**Research use:** This project is intended for academic and research purposes. For commercial use, consult the MagicBrush license terms and ensure compliance.

---

## Contact

For questions, issues, or collaboration inquiries:
- **Email:** your.email@domain.com
- **Issues:** [GitHub Issues](https://github.com/yourusername/image-edit-eval/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/image-edit-eval/discussions)

---

**Last updated:** February 8, 2026  
**Status:** ✅ Checkpoint 1 complete — fully verified end-to-end pipeline  
**Next milestone:** Checkpoint 2 — Collect 200+ real human labels and improve baseline model
