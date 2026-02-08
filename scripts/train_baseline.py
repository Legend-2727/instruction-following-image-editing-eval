#!/usr/bin/env python
"""train_baseline.py — Train lightweight classifiers on CLIP features.

Usage
-----
    python scripts/train_baseline.py \\
        --data data/sample \\
        --labels data/annotations/labels.jsonl \\
        --out runs/baseline

Outputs
-------
    runs/baseline/
        adherence_model.joblib
        error_model.joblib
        metrics.json
        confusion_matrix.png
        classification_report.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_labels, ensure_dirs
from utils.schema import (
    ADHERENCE_LABELS,
    ADHERENCE_TO_IDX,
    ERROR_TYPES,
    ERROR_TO_IDX,
    NUM_ERROR_TYPES,
    LabelRecord,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MIN_SAMPLES = 5  # minimum labeled samples to proceed


# ═══════════════════════════════════════════════════════════════════════════════
# Feature construction
# ═══════════════════════════════════════════════════════════════════════════════
def build_features(
    emb_orig: np.ndarray,
    emb_edit: np.ndarray,
    emb_text: np.ndarray,
) -> np.ndarray:
    """Build the (N, 1538) feature matrix from pre-computed embeddings.

    Features per sample::

        [emb_edit, emb_orig, (emb_edit − emb_orig),
         cosine(prompt, edit), cosine(prompt, orig)]
    """
    diff = emb_edit - emb_orig
    cos_edit = (emb_text * emb_edit).sum(axis=1, keepdims=True)   # (N,1)
    cos_orig = (emb_text * emb_orig).sum(axis=1, keepdims=True)   # (N,1)
    return np.concatenate([emb_edit, emb_orig, diff, cos_edit, cos_orig], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Train adherence + error classifiers.")
    parser.add_argument("--data", type=str, default="data/sample")
    parser.add_argument("--labels", type=str, default="data/annotations/labels.jsonl")
    parser.add_argument("--out", type=str, default="runs/baseline")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    ensure_dirs(out_dir)

    # ── Load embeddings ──────────────────────────────────────────────────
    emb_path = data_dir / "embeddings.npz"
    if not emb_path.exists():
        logger.error("No embeddings found at %s. Run extract_embeddings.py first.", emb_path)
        sys.exit(1)

    npz = np.load(str(emb_path), allow_pickle=True)
    all_ids = list(npz["ids"])
    emb_orig = npz["emb_orig"]
    emb_edit = npz["emb_edit"]
    emb_text = npz["emb_text"]
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}

    # ── Load labels ──────────────────────────────────────────────────────
    raw_labels = load_labels(args.labels)
    if not raw_labels:
        logger.warning("No labels found at %s. Cannot train.", args.labels)
        sys.exit(0)

    # Deduplicate: keep latest label per id
    label_map = {}
    for r in raw_labels:
        label_map[r["id"]] = r

    # Filter to labels that have embeddings
    labeled_ids = [sid for sid in label_map if sid in id_to_idx]
    if len(labeled_ids) < MIN_SAMPLES:
        logger.warning(
            "Only %d labeled samples (need ≥%d). Skipping training.",
            len(labeled_ids), MIN_SAMPLES,
        )
        sys.exit(0)
    logger.info("Using %d labeled samples for training.", len(labeled_ids))

    # ── Build X, y ───────────────────────────────────────────────────────
    indices = [id_to_idx[sid] for sid in labeled_ids]
    X = build_features(emb_orig[indices], emb_edit[indices], emb_text[indices])

    y_adh = np.array([ADHERENCE_TO_IDX[label_map[sid]["adherence"]] for sid in labeled_ids])
    y_err = np.array([
        LabelRecord.from_dict(label_map[sid]).error_vector() for sid in labeled_ids
    ])

    # ── Train/val split ──────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    try:
        X_tr, X_val, y_adh_tr, y_adh_val, y_err_tr, y_err_val, ids_tr, ids_val = \
            train_test_split(
                X, y_adh, y_err, labeled_ids,
                test_size=args.test_size,
                random_state=args.seed,
                stratify=y_adh,
            )
    except ValueError:
        # Not enough samples per class for stratified split
        logger.warning("Stratified split failed — falling back to random split.")
        X_tr, X_val, y_adh_tr, y_adh_val, y_err_tr, y_err_val, ids_tr, ids_val = \
            train_test_split(
                X, y_adh, y_err, labeled_ids,
                test_size=args.test_size,
                random_state=args.seed,
            )

    logger.info("Train: %d  |  Val: %d", len(X_tr), len(X_val))

    # ── Adherence classifier ─────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
        average_precision_score,
    )
    import joblib

    adh_clf = LogisticRegression(
        max_iter=1000, random_state=args.seed, C=1.0,
    )
    adh_clf.fit(X_tr, y_adh_tr)
    y_adh_pred = adh_clf.predict(X_val)

    adh_acc = accuracy_score(y_adh_val, y_adh_pred)
    adh_f1 = f1_score(y_adh_val, y_adh_pred, average="macro", zero_division=0)
    adh_cm = confusion_matrix(y_adh_val, y_adh_pred, labels=list(range(len(ADHERENCE_LABELS))))
    adh_report = classification_report(
        y_adh_val, y_adh_pred,
        target_names=ADHERENCE_LABELS,
        zero_division=0,
    )

    logger.info("── Adherence results ──")
    logger.info("Accuracy : %.4f", adh_acc)
    logger.info("Macro-F1 : %.4f", adh_f1)
    print(adh_report)

    # ── Error classifier (multi-label) ───────────────────────────────────
    metrics: dict = {
        "adherence_accuracy": float(adh_acc),
        "adherence_macro_f1": float(adh_f1),
        "adherence_confusion_matrix": adh_cm.tolist(),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
    }

    # Only train error classifier if at least some errors exist
    has_errors = y_err_tr.sum() > 0
    if has_errors:
        err_clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, random_state=args.seed, C=1.0)
        )
        err_clf.fit(X_tr, y_err_tr)
        y_err_pred = err_clf.predict(X_val)

        # Per-class metrics
        err_f1_per = f1_score(y_err_val, y_err_pred, average=None, zero_division=0)
        err_f1_micro = f1_score(y_err_val, y_err_pred, average="micro", zero_division=0)
        err_f1_macro = f1_score(y_err_val, y_err_pred, average="macro", zero_division=0)

        # mAP (only for columns that have positive labels in val)
        try:
            err_proba = err_clf.predict_proba(X_val) if hasattr(err_clf, "predict_proba") else None
            if err_proba is not None:
                cols_with_pos = [c for c in range(y_err_val.shape[1]) if y_err_val[:, c].sum() > 0]
                if cols_with_pos:
                    mAP = float(np.mean([
                        average_precision_score(y_err_val[:, c], err_proba[:, c])
                        for c in cols_with_pos
                    ]))
                else:
                    mAP = 0.0
            else:
                mAP = 0.0
        except Exception:
            mAP = 0.0

        logger.info("── Error-type results ──")
        logger.info("Micro-F1 : %.4f", err_f1_micro)
        logger.info("Macro-F1 : %.4f", err_f1_macro)
        logger.info("mAP      : %.4f", mAP)
        for i, name in enumerate(ERROR_TYPES):
            logger.info("  %-25s F1=%.4f", name, err_f1_per[i] if i < len(err_f1_per) else 0.0)

        metrics["error_micro_f1"] = float(err_f1_micro)
        metrics["error_macro_f1"] = float(err_f1_macro)
        metrics["error_mAP"] = float(mAP)
        metrics["error_per_class_f1"] = {
            ERROR_TYPES[i]: float(err_f1_per[i]) for i in range(len(err_f1_per))
        }

        joblib.dump(err_clf, str(out_dir / "error_model.joblib"))
        logger.info("Saved error model → %s", out_dir / "error_model.joblib")
    else:
        logger.warning("No error labels in training set — skipping error classifier.")

    # ── Save everything ──────────────────────────────────────────────────
    joblib.dump(adh_clf, str(out_dir / "adherence_model.joblib"))
    logger.info("Saved adherence model → %s", out_dir / "adherence_model.joblib")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", out_dir / "metrics.json")

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write("=== Adherence ===\n")
        f.write(adh_report)
    logger.info("Saved report → %s", out_dir / "classification_report.txt")

    # ── Confusion matrix plot ────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            adh_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ADHERENCE_LABELS,
            yticklabels=ADHERENCE_LABELS,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Adherence Confusion Matrix")
        fig.tight_layout()
        fig.savefig(str(out_dir / "confusion_matrix.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved confusion matrix → %s", out_dir / "confusion_matrix.png")
    except Exception as exc:
        logger.warning("Could not save confusion matrix plot: %s", exc)

    logger.info("Done. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()
