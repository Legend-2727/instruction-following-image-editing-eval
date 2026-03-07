# Multilingual Instruction-Following Image Editing Error Classifier

**A multimodal ML/DL pipeline for classifying errors in text-guided image editing across low-resource languages (Nepali, Bangla, Hindi).**

Given a triplet of **(source image, edit instruction, edited image)**, the model predicts which of 11 error categories apply — working across English and three low-resource South Asian languages.

**Status:** ✅ **Checkpoint 3 complete** — Full pipeline trained and benchmarked on RTX 5090 (Mar 7, 2026)

---

## Key Results

### Cross-Lingual Error Classification (Test Set, n=380)

| Language | Macro-F1 | Weighted-F1 | Exact Match | Hamming Loss | XL Gap |
|----------|----------|-------------|-------------|--------------|--------|
| English  | 0.1901   | 0.3399      | 0.2526      | 0.1160       | —      |
| Nepali   | 0.1395   | 0.2764      | 0.2053      | 0.1261       | 0.0635 |
| Bangla   | 0.1600   | 0.2938      | 0.2447      | 0.1215       | 0.0461 |
| Hindi    | 0.1719   | 0.3099      | 0.2474      | 0.1194       | 0.0300 |

**Key insight:** Hindi shows the smallest cross-lingual gap (0.03), consistent with XLM-RoBERTa's stronger pre-training coverage. Nepali has the largest gap (0.06).

### Per-Class F1 Scores

| Error Type               | English | Nepali | Bangla | Hindi  |
|--------------------------|---------|--------|--------|--------|
| Wrong Object             | 0.1852  | 0.1961 | 0.2414 | 0.1538 |
| Missing Object           | 0.0606  | 0.0000 | 0.0000 | 0.0606 |
| Extra Object             | 0.0000  | 0.0000 | 0.0000 | 0.0000 |
| Wrong Attribute          | 0.1463  | 0.1429 | 0.1034 | 0.1653 |
| Spatial Error            | 0.0000  | 0.0000 | 0.0000 | 0.0417 |
| Style Mismatch           | 0.0000  | 0.0000 | 0.0000 | 0.0000 |
| Over-editing             | 0.2245  | 0.2000 | 0.1263 | 0.1584 |
| Under-editing            | 0.5506  | 0.4408 | 0.4962 | 0.5020 |
| Artifact / Quality Issue | 0.0000  | 0.1333 | 0.3077 | 0.2857 |
| Ambiguous Prompt         | 0.3333  | 0.0000 | 0.0000 | 0.0000 |
| Failed Removal           | 0.5909  | 0.4211 | 0.4848 | 0.5238 |

### Image Quality Metrics

- **CLIP-I Similarity:** 0.9063 ± 0.0883
- **SSIM:** 0.8723 ± 0.1369 (median: 0.9210)

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐
│ Source Image  │    │ Target Image │    │ Edit Instruction         │
│              │    │              │    │ (en / ne / bn / hi)      │
└──────┬───────┘    └──────┬───────┘    └────────────┬─────────────┘
       │                   │                         │
       ▼                   ▼                         ▼
 ┌─────────────────────────────────┐    ┌─────────────────────────┐
 │   CLIP ViT-B/32 (frozen)       │    │  XLM-RoBERTa-base       │
 │   Dual image encoding          │    │  Last 2 layers unfrozen │
 │   src_feat, tgt_feat           │    │  Multilingual text enc.  │
 └───────┬───────────────┬────────┘    └────────────┬─────────────┘
         │               │                          │
         ▼               ▼                          │
 ┌───────────────────────────────┐                  │
 │  DualImageCrossAttention      │                  │
 │  attn(tgt, src, src)          │                  │
 │  + diff_feat (tgt - src)      │                  │
 │  → concat [src, cross, diff]  │                  │
 └───────────────┬───────────────┘                  │
                 │                                  │
                 ▼                                  ▼
         ┌──────────────────────────────────────────────┐
         │  ImageTextFusion (cross-attention)            │
         │  img_proj(1536-d) × text_proj(768-d) → 512-d │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │  MLP Classifier        │
                   │  512 → 512 → 11        │
                   │  (GELU + Dropout 0.3)  │
                   └────────────┬───────────┘
                                │
                                ▼
                  11-dim multi-label output
                  (binary error classification)
```

**Model stats:** 369M total params, 18.3M trainable (5.0%) — vision frozen, text partially unfrozen.

---

## Pipeline Overview

### Checkpoint 3 Pipeline (Current — executed)

| Step | Script | Description | Status |
|------|--------|-------------|--------|
| 1 | `step1_load_dataset.py` | Stream 1,900 MagicBrush samples with retry logic | ✅ Done |
| 2 | `step2_translate.py` | Translate instructions to ne/bn/hi using NLLB-200-1.3B | ✅ Done |
| 3 | `step3_heuristic_judge.py` | Heuristic annotation (SSIM + pixel diff + keywords) | ✅ Done |
| 3' | `step3_vlm_judge.py` | VLM annotation with Qwen2.5-VL (optional upgrade) | ⏸️ Blocked by HF rate limits |
| 4 | `step4_human_review.py` | Gradio UI for human verification | ⏸️ Optional |
| 5 | `step5_train_classifier.py` | Train CLIP+XLM-R classifier with FocalLoss | ✅ Done |
| 6 | `step6_benchmark.py` | Cross-lingual benchmark + CLIP-I + SSIM | ✅ Done |

### Data Flow

```
MagicBrush (HF) ──stream──> 1,900 samples (965 unique images)
       │
       ▼
NLLB-200-1.3B ──translate──> en + ne + bn + hi instructions
       │
       ▼
Heuristic Judge ──annotate──> 11-class multi-label vectors
       │
       ▼
Train/Val/Test split (1330/190/380)
       │
       ▼
CLIP + XLM-R Classifier ──train 20 epochs──> best_model.pt
       │
       ▼
Cross-lingual Benchmark ──eval 4 languages──> BENCHMARK_REPORT.md
```

---

## Error Taxonomy (11 classes)

| # | Error Type               | Test Prevalence |
|---|--------------------------|-----------------|
| 0 | Wrong Object             | 17 (4.5%)       |
| 1 | Missing Object           | 10 (2.6%)       |
| 2 | Extra Object             | 3 (0.8%)        |
| 3 | Wrong Attribute          | 37 (9.7%)       |
| 4 | Spatial Error            | 12 (3.2%)       |
| 5 | Style Mismatch           | 11 (2.9%)       |
| 6 | Over-editing             | 33 (8.7%)       |
| 7 | Under-editing            | 97 (25.5%)      |
| 8 | Artifact / Quality Issue | 5 (1.3%)        |
| 9 | Ambiguous Prompt         | 4 (1.1%)        |
| 10| Failed Removal           | 15 (3.9%)       |

---

## Quick Start

### Setup

```bash
# Clone
git clone https://github.com/Legend-2727/instruction-following-image-editing-eval.git
cd instruction-following-image-editing-eval

# Install dependencies (requires CUDA GPU)
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Step 1: Download MagicBrush samples
python scripts/step1_load_dataset.py --n 1900 --out data/magicbrush --resume

# Step 2: Translate to Nepali, Bangla, Hindi
python scripts/step2_translate.py --data data/magicbrush --langs ne bn hi

# Step 3: Heuristic annotation
python scripts/step3_heuristic_judge.py --data data/magicbrush

# Step 5: Train classifier
nohup python scripts/step5_train_classifier.py \
  --data data/magicbrush --out runs/classifier \
  --epochs 25 --batch_size 16 --lr 2e-4 \
  --unfreeze_text --patience 7 \
  > logs/step5.log 2>&1 &

# Step 6: Cross-lingual benchmark
python scripts/step6_benchmark.py \
  --data data/magicbrush --model_dir runs/classifier --out runs/benchmark
```

### Optional: VLM-based annotation (higher quality, requires HF token)

```bash
# Set HuggingFace token for faster downloads
export HF_TOKEN=your_token_here

# Run Qwen2.5-VL judge instead of heuristic
python scripts/step3_vlm_judge.py --data data/magicbrush

# Human review via Gradio UI
python scripts/step4_human_review.py --data data/magicbrush
```

---

## Repository Structure

```
instruction-following-image-editing-eval/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── eval_config.yaml
├── scripts/
│   ├── step1_load_dataset.py        # Stream MagicBrush with retry
│   ├── step2_translate.py           # NLLB-200 translation (ne/bn/hi)
│   ├── step3_heuristic_judge.py     # Fast heuristic annotation
│   ├── step3_vlm_judge.py           # Qwen2.5-VL annotation (optional)
│   ├── step4_human_review.py        # Gradio human review UI
│   ├── step5_train_classifier.py    # CLIP+XLM-R classifier training
│   ├── step6_benchmark.py           # Cross-lingual benchmark
│   ├── compute_ssim.py              # SSIM computation utility
│   ├── run_pipeline.sh              # End-to-end pipeline runner
│   ├── make_sample_dataset.py       # (CP1) Dataset sampler
│   ├── extract_embeddings.py        # (CP1) CLIP feature extraction
│   ├── train_baseline.py            # (CP1) Logistic regression baseline
│   ├── analyze_failures.py          # (CP1) Correlation analysis
│   ├── analyze_vlm_results.py       # (CP2) VLM results analysis
│   ├── generate_edits.py            # (CP2) InstructPix2Pix edits
│   └── utils/
│       ├── __init__.py
│       ├── io.py                    # JSONL / image I/O
│       ├── schema.py                # Error taxonomy, dataclasses
│       ├── clip_encoder.py          # CLIP image/text embedder
│       ├── text_encoder.py          # Text encoder interface
│       ├── prompt_features.py       # Prompt feature extraction
│       ├── editor_model.py          # InstructPix2Pix wrapper
│       └── vlm_evaluator.py         # Qwen2.5-VL judge wrapper
├── apps/
│   ├── label_tool.py                # (CP1) Streamlit labeling UI
│   └── human_review_tool.py         # Gradio human review app
├── notebooks/
│   └── checkpoint1_demo.ipynb
├── data/                            # (git-ignored, generated at runtime)
│   └── magicbrush/
│       ├── images/source/           # 1,903 source PNGs
│       ├── images/target/           # 1,903 target PNGs
│       ├── metadata.jsonl           # 1,900 records with translations
│       ├── metadata_translated.jsonl
│       ├── vlm_annotations.jsonl    # Heuristic annotations
│       └── annotations_final.jsonl  # Final labels (11-class vectors)
├── runs/                            # (git-ignored, generated at runtime)
│   ├── classifier/
│   │   ├── best_model.pt            # Best checkpoint (~1.6 GB)
│   │   ├── last_model.pt            # Last checkpoint
│   │   ├── config.json
│   │   ├── test_metrics.json
│   │   ├── test_set.jsonl
│   │   └── training_history.json
│   └── benchmark/
│       ├── BENCHMARK_REPORT.md
│       └── benchmark_results.json
└── logs/                            # (git-ignored)
    ├── step2.log
    ├── step3.log
    ├── step5.log
    └── step6.log
```

---

## Technical Details

### Models Used

| Component | Model | Size | Role |
|-----------|-------|------|------|
| Vision encoder | CLIP ViT-B/32 (`laion2b_s34b_b79k`) | 87.8M params (frozen) | Dual image encoding |
| Text encoder | XLM-RoBERTa-base | 278M params (last 2 layers unfrozen) | Multilingual instruction encoding |
| Translator | NLLB-200-distilled-1.3B | 1.3B params | English → Nepali/Bangla/Hindi |
| Fusion + classifier | Cross-attention + MLP | 18.3M params (trainable) | Multimodal fusion → 11-class output |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=2e-4, weight_decay=0.01) |
| Scheduler | Linear warmup (100 steps) + cosine annealing |
| Loss | Focal Loss (gamma=2.0, alpha=1.0) + class-weighted pos_weight |
| Batch size | 16 (×2 gradient accumulation = effective 32) |
| Epochs | 25 max (early stopped at epoch 20) |
| Prediction threshold | 0.3 |
| Train/Val/Test | 1,330 / 190 / 380 |

### Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5090 (32 GB VRAM) |
| RAM | 49 GB |
| Disk | ~460 GB free |
| CUDA | 13.0 / Driver 580.95.05 |
| Python | 3.10.12 |
| PyTorch | 2.10.0+cu128 |

---

## What to Continue Next

### Immediate Improvements (high impact)

1. **VLM-based annotation upgrade** — Replace heuristic annotations with Qwen2.5-VL judgments. Set `HF_TOKEN` env var to avoid rate limiting, then run `step3_vlm_judge.py`. This should significantly improve label quality and boost F1 across all classes.

2. **Scale dataset** — Download all 5,000+ MagicBrush samples (currently 1,900). More data, especially for rare classes (Extra Object, Artifact/Quality), will help the tail classes.

3. **Data augmentation** — Apply random language mixing during training (randomly pick en/ne/bn/hi per sample) to improve cross-lingual transfer.

### Architecture Improvements

4. **Unfreeze more XLM-R layers** — Currently only last 2 layers are unfrozen. Gradually unfreezing more layers (e.g., last 4) with discriminative learning rates could help.

5. **Per-class threshold tuning** — Currently using a flat 0.3 threshold. Tune per-class thresholds on the validation set to optimize each class independently.

6. **Larger vision backbone** — Switch from CLIP ViT-B/32 to ViT-L/14 for richer visual features (requires ~4x more VRAM).

7. **Add more languages** — Extend to other low-resource languages (Urdu, Tamil, Thai) to strengthen the cross-lingual evaluation.

### Publication Path

8. **Ablation studies** — Freeze/unfreeze analysis, focal loss vs BCE, heuristic vs VLM labels, mono- vs cross-lingual.

9. **Comparison baselines** — Train a text-only baseline (no images) and image-only baseline (no text) to quantify multimodal contribution.

10. **Error analysis** — Deep-dive into failure modes: which samples are hardest, what do zero-F1 classes look like, are certain edit types systematically harder?

11. **Write paper** — Target ACL/EMNLP (NLP + multilinguality angle) or CVPR/ECCV (vision + editing angle). The unique contribution is the **multilingual error taxonomy for image editing evaluation**.

---

## Previous Checkpoints

<details>
<summary>Checkpoint 1 (Feb 8, 2026) — CLIP Baseline</summary>

- Built dataset sampler with HuggingFace streaming
- Implemented 11-category error taxonomy
- Created Streamlit labeling tool
- Integrated CLIP ViT-B-32 for feature extraction
- Trained baseline logistic regression classifiers
- Built prompt-failure correlation analyzer
- Verified end-to-end on 20 samples

</details>

<details>
<summary>Checkpoint 2 (Mar 1, 2026) — VLM-as-Judge Design</summary>

- Integrated InstructPix2Pix as SOTA open-source editor
- Integrated Qwen2.5-VL-3B as VLM-as-judge
- Built edit generation, VLM judging, and analysis scripts
- Added human verification Streamlit UI
- Had VRAM/dependency issues on Windows (resolved in CP3 by moving to Linux)

</details>

### Checkpoint 3 (Mar 7, 2026) — Multilingual Classifier (Current)

- Migrated to Linux (Ubuntu) + RTX 5090
- Downloaded 1,900 MagicBrush samples via streaming
- Translated all instructions to Nepali, Bangla, Hindi using NLLB-200
- Created heuristic annotation pipeline (SSIM + pixel diff + keywords)
- Designed and trained CLIP ViT-B/32 + XLM-RoBERTa multimodal classifier
- Implemented Focal Loss + partial unfreezing for class imbalance
- Ran cross-lingual benchmark across 4 languages
- Computed CLIP-I and SSIM image quality metrics
- Generated full benchmark report

---

## License

MIT

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
