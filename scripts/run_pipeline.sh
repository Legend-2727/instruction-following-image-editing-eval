#!/bin/bash
# run_pipeline.sh — Master script to run the full pipeline
#
# Usage:
#   bash scripts/run_pipeline.sh           # Run everything
#   bash scripts/run_pipeline.sh --step 3  # Start from step 3
#   bash scripts/run_pipeline.sh --quick   # Quick test with 100 samples

set -e

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
DATA_DIR="data/magicbrush"
RUNS_DIR="runs"

# Parse args
STEP=1
N_SAMPLES=5000
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2;;
        --quick) QUICK=true; N_SAMPLES=100; shift;;
        --n) N_SAMPLES="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  Multilingual Image Editing Error Classifier Pipeline"
echo "  Samples: $N_SAMPLES | Starting from step: $STEP"
echo "============================================================"

# ── Step 1: Load Dataset ────────────────────────────────────────────────────
if [ "$STEP" -le 1 ]; then
    echo ""
    echo "[STEP 1/6] Loading MagicBrush dataset ($N_SAMPLES samples)..."
    python scripts/step1_load_dataset.py --n $N_SAMPLES --out $DATA_DIR --resume
    echo "[STEP 1] ✓ Dataset loaded"
fi

# ── Step 2: Translate ───────────────────────────────────────────────────────
if [ "$STEP" -le 2 ]; then
    echo ""
    echo "[STEP 2/6] Translating instructions (EN → NE/BN/HI)..."
    python scripts/step2_translate.py --data $DATA_DIR --resume
    echo "[STEP 2] ✓ Translations complete"
fi

# ── Step 3: VLM Judge ──────────────────────────────────────────────────────
if [ "$STEP" -le 3 ]; then
    echo ""
    echo "[STEP 3/6] Running VLM judge (Qwen2.5-VL)..."
    python scripts/step3_vlm_judge.py --data $DATA_DIR --resume
    echo "[STEP 3] ✓ VLM annotations complete"
fi

# ── Step 4: Human Review (interactive — skip in automated mode) ─────────────
if [ "$STEP" -le 4 ] && [ "$STEP" -eq 4 ]; then
    echo ""
    echo "[STEP 4/6] Launching human review app..."
    echo "  Open http://localhost:7860 in your browser"
    echo "  Press Ctrl+C when done reviewing"
    python scripts/step4_human_review.py --data $DATA_DIR --port 7860
fi

# ── Step 5: Train Classifier ───────────────────────────────────────────────
if [ "$STEP" -le 5 ]; then
    echo ""
    echo "[STEP 5/6] Training multilingual error classifier..."
    python scripts/step5_train_classifier.py \
        --data $DATA_DIR \
        --out $RUNS_DIR/classifier \
        --epochs 15 \
        --batch_size 8 \
        --grad_accum 4 \
        --patience 3
    echo "[STEP 5] ✓ Training complete"
fi

# ── Step 6: Benchmark ──────────────────────────────────────────────────────
if [ "$STEP" -le 6 ]; then
    echo ""
    echo "[STEP 6/6] Running benchmark evaluation..."
    python scripts/step6_benchmark.py \
        --data $DATA_DIR \
        --model_dir $RUNS_DIR/classifier \
        --out $RUNS_DIR/benchmark
    echo "[STEP 6] ✓ Benchmark complete"
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results: $RUNS_DIR/benchmark/BENCHMARK_REPORT.md"
echo "============================================================"
