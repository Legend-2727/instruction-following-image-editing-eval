# Evidence-Grounded Taxonomy Evaluator (Explained)

## Notebook Overview
This companion document mirrors the explained lineage notebook for the multilingual evidence-grounded taxonomy evaluator. It documents each cell by purpose, dependencies, key logic, outputs, and pipeline role, and preserves the v1 -> v2 -> v3 progression.

### Scope
- Languages: EN / HI / BN
- Inputs per row: source image + edited image + one-language instruction
- Outputs: multi-label taxonomy predictions plus structured evidence signals

## Cell 1

**Purpose**
- Establishes notebook context by stating the multilingual task, protocol, and evidence-grounded objective.

**Why it matters**
- This is the contract for the rest of the notebook: every downstream choice (data expansion, evaluation split, metrics, and artifact naming) is interpreted relative to this framing.

**Inputs / dependencies**
- No runtime variables; this is documentation state for readers and maintainers.

**Key logic**
- Defines the fixed EN/HI/BN multilingual scope.
- States that split-by-original happens before language expansion.
- Clarifies that inference is single-prompt per row.
- Anchors primary evaluation to the taxonomy-positive benchmark.
- Introduces evidence-grounded reasoning signals rather than pure black-box scoring.

**Outputs**
- Human-readable specification of experiment intent and protocol invariants.

**How it connects to the pipeline**
- This is the entry point for environment/setup and research framing before any code execution.

**Notes / implementation details**
- The updated intro now explicitly distinguishes v1, v2, and v3 lineage, so readers can map each training cell to a concrete experimental stage.

## Cell 2

**Purpose**
- Imports all dependencies and chooses runtime device (CUDA or CPU).

**Why it matters**
- Every subsequent stage depends on these libraries for data I/O, modeling, training, evaluation, and artifact export.

**Inputs / dependencies**
- Python environment with PyTorch, torchvision, transformers, pandas/numpy, PIL, and sklearn installed.

**Key logic**
- Imports utility modules (`os`, `json`, `math`, `copy`, `random`, `Path`) used across data and training stages.
- Imports tensor/model stack (`torch`, `nn`, `F`, `Dataset`, `DataLoader`).
- Imports backbones (`ViTModel`, `XLMRobertaModel`) and tokenizer (`AutoTokenizer`).
- Imports classification metrics needed for multi-label reporting.
- Detects hardware and binds global `device`.

**Outputs**
- Global symbols for all core libraries.
- `device` variable used in model/data movement.

**How it connects to the pipeline**
- Environment/setup stage; all downstream cells assume these imports and `device` exist.

**Notes / implementation details**
- Choosing device early avoids hidden CPU/GPU mismatch bugs later in dataloading, loss evaluation, and checkpoint restore.

## Cell 3

**Purpose**
- Centralizes experiment configuration: paths, multilingual protocol constants, model IDs, and optimization hyperparameters.

**Why it matters**
- A reproducible research notebook needs an explicit parameter surface. This cell is the single source for most run-defining knobs.

**Inputs / dependencies**
- `Path` import from previous cell.
- Expected Kaggle-style dataset directories and split JSONL files.

**Key logic**
- Defines data roots (`IMAGE_ROOT`, `SPLITS_DIR`) and split files (`TRAIN_JSONL`, `VAL_JSONL`, `TEST_JSONL`).
- Defines output directory `WORK_DIR` for v1 artifacts.
- Locks multilingual scope to `LANGS = ('en', 'hi', 'bn')`.
- Sets calibration-aware training mix via `TRAIN_NEG_RATIO`.
- Sets benchmark mode to taxonomy-positive for comparability.
- Declares backbone IDs (ViT + XLM-R), feature dimensions, and text length.
- Declares optimizer/training controls (batch size, epochs, LR split for heads vs encoders, weight decay, AMP, focal gamma, grad clip).
- Declares partial unfreezing depth for text and vision encoders.

**Outputs**
- Global config constants reused by all downstream cells.
- Created output directory for artifact writes.

**How it connects to the pipeline**
- Configuration stage linking setup to data loading, model construction, and v1 training.

**Notes / implementation details**
- Separate LR for heads vs encoders encodes a standard transfer-learning assumption: new heads need larger updates than pretrained backbones.

## Cell 4

**Purpose**
- Performs an early filesystem sanity check for required image and split assets.

**Why it matters**
- Failing fast on path issues prevents costly debugging after model/tokenizer loading and partial training.

**Inputs / dependencies**
- Config variables from the previous cell (`IMAGE_ROOT`, split paths).
- `Path` utilities.

**Key logic**
- Builds a list of required paths (root, source/target image dirs, train/val/test JSONLs).
- Prints pass/fail status per path.
- Aggregates status and raises `FileNotFoundError` if any critical input is missing.

**Outputs**
- Console validation report.
- Hard stop on invalid environment.

**How it connects to the pipeline**
- Validation checkpoint between config and data processing.

**Notes / implementation details**
- This keeps failures deterministic and front-loaded, which is important for smoke-test discipline in research pipelines.

## Cell 5

**Purpose**
- Defines canonical taxonomy labels, operation labels, and core data helper functions used to build multilingual training/eval rows.

**Why it matters**
- This cell is the practical label schema and preprocessing backbone: it maps raw JSONL fields to model-ready targets while preserving multilingual behavior.

**Inputs / dependencies**
- Config (`LANGS`, `SEED`) and imported libraries (`json`, `re`, `random`, `numpy`, `torch`, `Path`).

**Key logic**
- Declares 11 taxonomy classes and index mapping (`ERROR_TYPES`, `ERROR_TO_IDX`).
- Declares 8 operation categories (`OP_TYPES`, `OP_TO_IDX`) for auxiliary supervision and evidence cues.
- Provides utilities:
- `seed_everything`: reproducibility across Python/NumPy/Torch.
- `load_jsonl`: robust line-wise loading.
- `save_json`: consistent UTF-8 artifact export.
- `normalize_path`: normalizes separators toward POSIX-style relative paths.
- `labels_to_multihot`: converts taxonomy label list to multi-hot target vector.
- `infer_prompt_operation`: multilingual regex heuristics to infer prompt operation class.
- `expand_rows_by_language`: converts one original sample into separate language rows (EN/HI/BN), each with single instruction text.
- `filter_taxonomy_positive`: selects rows with at least one taxonomy label.
- `build_train_rows`: combines positives and sampled negatives for calibration-aware training.
- Seeds are fixed at the end for deterministic sampling.

**Outputs**
- Global taxonomy/operation mappings and reusable helper functions.

**How it connects to the pipeline**
- Data schema and row-construction stage; directly feeds split expansion and dataset objects.

**Notes / implementation details**
- The split-by-original then expand-by-language protocol is implemented here, preserving no-leakage intent while enabling multilingual evaluation.
- Inference remains one-prompt-per-row because each expanded row carries exactly one language instruction.

## Cell 6

**Purpose**
- Loads train/val/test originals and materializes multilingual row tables for training and evaluation variants.

**Why it matters**
- This converts split JSONL files into the exact learning/evaluation units consumed by the dataset class and loaders.

**Inputs / dependencies**
- Split paths (`TRAIN_JSONL`, `VAL_JSONL`, `TEST_JSONL`).
- Helper functions from prior cell (`load_jsonl`, `build_train_rows`, `expand_rows_by_language`, `filter_taxonomy_positive`).
- Config flags (`SMOKE_TEST`, `TRAIN_NEG_RATIO`, `LANGS`).

**Key logic**
- Loads original-id split files.
- Applies optional smoke-test truncation while keeping train/val/test structure.
- Builds training rows with sampled negatives (`TRAIN_NEG_RATIO`).
- Builds full multilingual val/test rows.
- Builds taxonomy-positive benchmark val/test rows for comparable headline metrics.
- Writes a JSON dataset summary with counts and protocol metadata.

**Outputs**
- `train_rows`, `val_rows_full`, `test_rows_full`.
- `val_rows_benchmark`, `test_rows_benchmark`.
- `dataset_summary.json` artifact in `WORK_DIR`.

**How it connects to the pipeline**
- Data loading and multilingual expansion stage immediately before transforms and dataset wrapping.

**Notes / implementation details**
- The benchmark subset is intentionally separate from full-set evaluation so comparability and calibration analysis can coexist.

## Cell 7

**Purpose**
- Defines the image normalization/resize transform used during sample loading.

**Why it matters**
- Pretrained ViT backbones assume normalized image distributions and fixed resolution; mismatched preprocessing can degrade performance severely.

**Inputs / dependencies**
- `IMG_SIZE` config and `torchvision.transforms` import.

**Key logic**
- Uses ImageNet mean/std normalization.
- Applies deterministic resize to `IMG_SIZE x IMG_SIZE`.
- Converts PIL images to tensors and normalizes channels.

**Outputs**
- `EVAL_TRANSFORM` callable used by dataset instances.

**How it connects to the pipeline**
- Transform stage between row expansion and dataset class, ensuring source and target images are encoded consistently.

**Notes / implementation details**
- The same transform is used for train and eval in this notebook, favoring protocol stability over augmentation complexity.

## Cell 8

**Purpose**
- Creates tokenizer and dataset class that maps multilingual row dictionaries into model-ready tensors.

**Why it matters**
- This cell is the bridge from tabular metadata to batched multimodal inputs (`source image`, `target image`, `instruction`).

**Inputs / dependencies**
- `TEXT_MODEL_ID`, `MAX_TEXT_LEN`, row lists from prior cells, and `EVAL_TRANSFORM`.

**Key logic**
- Instantiates tokenizer from XLM-R model family.
- Defines `EvidenceTaxonomyDataset` with:
- `__len__` for DataLoader compatibility.
- `_load_image` that joins root + relative path, converts to RGB, and applies transform.
- `__getitem__` that loads source/target images, tokenizes one language instruction, and emits:
  - IDs/lang metadata
  - image tensors
  - token IDs + attention mask
  - taxonomy multi-hot target vector
  - binary `has_error`
  - operation label ID (`op_type_id`)

**Outputs**
- `baseline_tokenizer` and dataset class definition.

**How it connects to the pipeline**
- Dataset stage, immediately feeding dataloaders and model forward/training loops.

**Notes / implementation details**
- Each item contains a single-language instruction to preserve one-prompt inference semantics across EN/HI/BN variants.

## Cell 9

**Purpose**
- Defines and instantiates the evidence-grounded architecture: dual-image encoder + multilingual text encoder + explicit evidence feature construction + taxonomy head.

**Why it matters**
- This cell implements the paper-critical methodological claim: predictions are driven through interpretable evidence channels, not only opaque late fusion.

**Inputs / dependencies**
- Backbone IDs and dims from config.
- `NUM_CLASSES`, `OP_TYPES`, and `device` from prior setup.

**Key logic**
- Utility controls:
- `freeze_all` and `unfreeze_last_n_layers` implement partial fine-tuning.
- `MLP` helper standardizes small prediction heads.
- `EvidenceGroundedTaxonomyModel`:
- Backbones: ViT for source/target images, XLM-R for text.
- Projection layers align modalities into shared hidden space.
- Prompt-conditioned grounding:
  - text CLS is projected to query `q`.
  - patch-text similarities produce source/target attention maps.
  - attention-weighted local visual summaries (`src_local`, `tgt_local`).
- Edit evidence features:
  - `patch_diff` absolute patch-level change.
  - `local_diff` (prompt-focused change).
  - `outside_diff` (off-target change).
  - correspondence scores (`corr_max`, `corr_mean`) checking whether source-local semantics persist in edited image.
- Evidence heads:
  - source/target presence
  - local/global/outside change scores
  - operation logits/probabilities from text features.
- Evidence fusion:
  - concatenates textual, global visual, local visual, difference, scalar evidence, and op probabilities.
  - passes fused vector to taxonomy head for 11-label logits.
- Returns both `taxonomy_logits` and structured `evidence` dictionary.
- Instantiates model and moves it to selected device.

**Outputs**
- `model` object with evidence outputs and taxonomy logits.

**How it connects to the pipeline**
- Core architecture stage used by v1 training and later v2/v3 fine-tuning.

**Notes / implementation details**
- “Evidence-grounded” here means explicit intermediate signals are computed and exposed (`src_attn`, `tgt_attn`, presence, correspondence, local/outside change, op cues), then used by the final classifier.
- Partial unfreezing balances adaptation with stability and compute cost.

## Cell 10

**Purpose**
- Defines losses, metrics, threshold tuning, evaluation routine, and optimizer/scaler setup for v1 training.

**Why it matters**
- This cell determines what the model optimizes, how model quality is measured, and how outputs are converted to multi-label decisions.

**Inputs / dependencies**
- `train_rows`, `model`, taxonomy constants, operation labels, and optimization hyperparameters from config.

**Key logic**
- `FocalBCEWithLogitsLoss` for multi-label taxonomy with class imbalance emphasis.
- `compute_pos_weight` estimates per-class imbalance weights from training targets.
- Metric utilities compute micro/macro F1 and AP variants (`mAP_micro`, supported-class macro AP).
- `compute_per_class_metrics` exports class-wise precision/recall/F1/support.
- `tune_thresholds` performs per-class threshold search on validation predictions.
- `run_eval` executes inference with mixed precision, computes combined loss (taxonomy + aux operation), and logs structured evidence rows.
- `collect_trainable_params` separates head and encoder parameters for differential learning rates.
- Instantiates:
- taxonomy and operation losses,
- AdamW optimizer with two parameter groups,
- AMP gradient scaler.

**Outputs**
- Training/evaluation function set, configured criteria, optimizer, and AMP scaler.

**How it connects to the pipeline**
- Optimization/measurement stage that powers v1 training loop and benchmark evaluation.

**Notes / implementation details**
- Per-class threshold tuning is crucial in multi-label tasks because a global 0.5 threshold often underperforms on imbalanced classes.

## Cell 11

**Purpose**
- Builds dataset instances and dataloaders for train/full-eval/benchmark-eval splits.

**Why it matters**
- Dataloaders are the execution boundary between static row objects and efficient mini-batch training/evaluation.

**Inputs / dependencies**
- Row tables from split expansion cell.
- `EvidenceTaxonomyDataset`, tokenizer, transforms, and batch/worker config.

**Key logic**
- Creates datasets for:
- training rows (positives + sampled negatives),
- full validation/test rows,
- taxonomy-positive validation/test benchmark rows.
- Wraps each dataset with `DataLoader` and appropriate shuffle flag.
- Enables pinned memory when running on CUDA.
- Prints batch counts for sanity checking.

**Outputs**
- `train_loader`, `val_loader_full`, `test_loader_full`, `val_loader_benchmark`, `test_loader_benchmark`.

**How it connects to the pipeline**
- Final data plumbing stage before v1 optimization loop.

**Notes / implementation details**
- Maintaining both full and benchmark loaders supports two reporting regimes: strict comparability (benchmark) and broader calibration visibility (full).

## Cell 12

**Purpose**
- Runs v1 model training with validation-based model selection and threshold tuning.

**Why it matters**
- Produces the first fully trained evidence-grounded checkpoint and establishes baseline lineage artifacts.

**Inputs / dependencies**
- Model, losses, optimizer/scaler, train/val benchmark loaders, and training hyperparameters.

**Key logic**
- Iterates epochs and mini-batches with mixed precision forward pass.
- Optimizes joint objective: taxonomy focal loss + weighted operation auxiliary loss.
- Applies gradient unscaling and clipping for stability.
- Evaluates each epoch on taxonomy-positive validation benchmark.
- Tunes per-class thresholds on validation probabilities.
- Computes multi-label metrics and uses `macro_f1_supported` for model selection.
- Saves best model state and threshold JSON.
- Implements patience-based early stopping (disabled in smoke mode).
- Writes training history artifact.

**Outputs**
- In-memory best checkpoint (`best_state`) and threshold list.
- Files: `best_model.pt`, `best_thresholds.json`, `train_history.json` in `WORK_DIR`.

**How it connects to the pipeline**
- v1 training stage; downstream evaluation/export and later v2 warm-start depend on these artifacts.

**Notes / implementation details**
- Selection on taxonomy-positive validation keeps headline metric comparability with previous benchmark conventions.

## Cell 13

**Purpose**
- Performs post-training evaluation on benchmark and full test sets, then exports reproducible v1 artifacts.

**Why it matters**
- Converts training results into publishable outputs: metrics, thresholds, per-class tables, predictions, and evidence traces.

**Inputs / dependencies**
- Best model state/checkpoint, tuned thresholds, test loaders, and metric/eval helpers.

**Key logic**
- Restores best model.
- Runs primary benchmark evaluation (taxonomy-positive test) and secondary full-test evaluation.
- Applies tuned per-class thresholds to probabilities.
- Computes aggregate and per-class metrics.
- Exports `metrics.json` with config snapshot and thresholds.
- Exports `label_map.json`.
- Exports row-level benchmark predictions JSONL with gold labels, predicted labels, and class probabilities.
- Exports evidence CSV for benchmark rows.

**Outputs**
- Files in `WORK_DIR`: `metrics.json`, `label_map.json`, `per_class_test_benchmark.csv`, `predictions_test_benchmark.jsonl`, `evidence_test_benchmark.csv` plus model/threshold files.

**How it connects to the pipeline**
- v1 evaluation and artifact stage; provides baseline outputs reused for analysis and v2 initialization.

**Notes / implementation details**
- Dual reporting (benchmark + full) supports both strict comparability and broader behavior diagnosis.

## Cell 14

**Purpose**
- Builds a readable evidence preview by mapping numeric evidence signals and predicted labels into compact reason strings.

**Why it matters**
- Improves interpretability and qualitative auditing, making evidence-grounded behavior easier to inspect.

**Inputs / dependencies**
- `evidence_tb` and `benchmark_pred_rows` generated in the previous evaluation cell.

**Key logic**
- Defines `evidence_to_reason` to summarize predicted operation, presence/correspondence/change scores, and label set.
- Joins prediction labels back onto evidence rows using `(id, lang)` key.
- Creates `reason` text field for quick manual review.
- Displays a preview and exports first 50 rows.

**Outputs**
- `evidence_view` DataFrame for inspection.
- `evidence_preview.csv` artifact.

**How it connects to the pipeline**
- Interpretability inspection stage after v1 artifact generation.

**Notes / implementation details**
- This is intentionally optional and qualitative, complementing quantitative benchmark metrics.

## Cell 15

**Purpose**
- Resolves model/loaders robustly from notebook global state to support downstream v2 execution even if variable names differ.

**Why it matters**
- Notebook lineage workflows often rerun cells out-of-order; this resolver reduces fragility and supports reproducible continuation.

**Inputs / dependencies**
- Existing notebook globals and possible DataLoader/dataset variable variants.

**Key logic**
- Re-imports needed utilities locally for standalone robustness.
- `_pick_first` resolves first existing symbol among candidate names.
- `_find_loader_by_keywords` finds likely loaders by name patterns.
- Attempts canonical names first, then keyword search fallback.
- If loaders are still missing, rebuilds them from available dataset objects.
- Validates required objects and raises explicit errors if unresolved.
- Prints final binding summary for model/device/loaders.

**Outputs**
- Guaranteed local bindings: `model`, `device`, `train_loader`, `val_loader`, `test_loader_benchmark`, `test_loader_full` (or explicit failure).

**How it connects to the pipeline**
- Transition/support stage between v1 and v2, ensuring fine-tuning cells can run reliably in interactive settings.

**Notes / implementation details**
- This cell adds resiliency, not new learning logic; training behavior remains defined by later v2/v3 cells.

## Cell 16

**Purpose**
- Prefers taxonomy-focused validation loader for v2 model selection when available.

**Why it matters**
- Aligns v2 model selection with benchmark intent (taxonomy-positive) rather than potentially broader validation variants.

**Inputs / dependencies**
- Loader variables from the resolver cell and any benchmark-specific val loader aliases.

**Key logic**
- Checks a priority order of benchmark-oriented validation loader names.
- Rebinds `val_loader` and `val_loader_name` if a preferred candidate exists.
- Prints final loader selection summary.

**Outputs**
- Finalized `val_loader` choice for v2 fine-tuning.

**How it connects to the pipeline**
- Small but important pre-v2 selection policy adjustment.

**Notes / implementation details**
- This keeps lineage comparisons fair by selecting on a validation slice closer to primary reporting conditions.

## Cell 17

**Purpose**
- Runs v2 fine-tuning from saved v1 checkpoint with class-weighted taxonomy BCE and updated optimization settings, then evaluates and exports v2 artifacts.

**Why it matters**
- Implements the first lineage improvement step: targeted refinement for minority taxonomy classes and stable warm-start adaptation.

**Inputs / dependencies**
- v1 checkpoint (`best_model.pt`), resolved loaders/model/device, taxonomy class names, and existing architecture.

**Key logic**
- Resolves required globals and verifies presence.
- Loads v1 best weights into current model.
- Defines helper functions for batching, metrics, threshold tuning, and evaluation.
- Configures v2 training policy:
- weighted BCE loss (`pos_weight` clipped for stability),
- lower LR and stronger regularization,
- taxonomy-only optimization objective (no aux op loss term in objective).
- Trains for up to `V2_EPOCHS` with gradient clipping.
- Tunes thresholds on validation each epoch.
- Selects best checkpoint by `macro_f1_supported`.
- Evaluates best v2 on benchmark and full test.
- Exports v2 history, thresholds, metrics, per-class benchmark CSV, and prediction JSONL.

**Outputs**
- Files in `evidence_grounded_taxonomy_eval_v2/`: `best_model.pt`, `best_thresholds.json`, `metrics.json`, `train_history_v2.json`, `per_class_test_benchmark.csv`, `predictions_test_benchmark.jsonl`.

**How it connects to the pipeline**
- v2 lineage stage: warm-start refinement of v1 with altered loss/optimization strategy.

**Notes / implementation details**
- The v2 objective emphasizes class imbalance correction; this can improve supported-class macro F1 while potentially affecting calibration-oriented AP metrics.

## Cell 18

**Purpose**
- Runs v3 fine-tuning (prefer warm-start from v2, fallback to v1) with blended taxonomy losses and composite model selection, then exports v3 artifacts.

**Why it matters**
- Implements the second lineage refinement focused on balancing classification quality (F1) and calibration-sensitive ranking quality (AP).

**Inputs / dependencies**
- Existing model/loaders/device globals.
- v2 or v1 checkpoint availability.
- Taxonomy constants and architecture from earlier cells.

**Key logic**
- Validates required globals.
- Selects taxonomy-focused validation loader when available.
- Chooses start checkpoint: v2 preferred, v1 fallback.
- Defines helper functions including optional operation-label extraction.
- Sets v3 strategy:
- milder class weighting (`sqrt` imbalance, lower cap),
- blended taxonomy loss: plain BCE + weighted BCE,
- optional low-weight auxiliary operation loss if op labels exist,
- lower LR/weight decay,
- composite model-selection score: `0.70 * macro_f1_supported + 0.30 * mAP_micro`.
- Trains with gradient clipping and threshold tuning each epoch.
- Early-stops by composite score.
- Evaluates best v3 on benchmark and full test.
- Exports v3 history, thresholds, metrics (including blend config), per-class CSV, and prediction JSONL.

**Outputs**
- Files in `evidence_grounded_taxonomy_eval_v3/`: `best_model.pt`, `best_thresholds.json`, `metrics.json`, `train_history_v3.json`, `per_class_test_benchmark.csv`, `predictions_test_benchmark.jsonl`.

**How it connects to the pipeline**
- v3 lineage stage, completing progression from base training (v1) to imbalance-focused tuning (v2) to F1/AP trade-off tuning (v3).

**Notes / implementation details**
- v3 explicitly encodes a metric trade-off objective for model selection, making the lineage rationale transparent for methods reporting.

## Notebook Summary

This lineage notebook implements a full multilingual evidence-grounded taxonomy evaluation pipeline for instruction-following image edits using EN/HI/BN prompt variants. The pipeline proceeds from environment and configuration setup, through group-preserving split loading and language-row expansion, into dual-image + multilingual-text modeling, and finally through three training generations (v1, v2, v3).

### End-to-End Pipeline
- setup and reproducible configuration
- path and data contract checks
- taxonomy/operation schema and multilingual row expansion
- transform + dataset + dataloader construction
- evidence-grounded architecture definition
- training/evaluation utilities with threshold tuning
- v1 training and artifact export
- interpretability preview generation
- v2 warm-start fine-tuning and export
- v3 warm-start fine-tuning and export

### Architecture and Evidence-Grounding
- Vision backbone: ViT applied separately to source and edited images.
- Text backbone: XLM-R for multilingual instruction encoding.
- Prompt-conditioned grounding computes patch attention in both source and target images from text-derived query features.
- Structured evidence includes presence, correspondence, local/global/outside change, and prompt operation cues.
- Final taxonomy logits are predicted from fused evidence-aware features rather than raw late fusion alone.

### v1 -> v2 -> v3 Evolution
- v1 establishes the baseline evidence-grounded training with focal multi-label taxonomy loss plus auxiliary operation supervision.
- v2 warm-starts from v1 and emphasizes minority taxonomy classes through weighted BCE and adjusted optimization settings.
- v3 warm-starts from v2 (or v1 fallback) and balances F1 and AP using blended taxonomy losses plus a composite selection score.

### Produced Artifacts
- model checkpoints (`best_model.pt`) per version
- threshold files (`best_thresholds.json`) per version
- aggregate metrics (`metrics.json`) and per-class benchmark tables
- benchmark prediction JSONL outputs
- evidence CSV exports and preview tables for interpretability

### Remaining Limitations
- prompt-operation inference still uses heuristic regex rules rather than learned normalization.
- no explicit cross-language calibration report is generated in this notebook itself.
- training uses fixed image preprocessing without augmentation ablations.
- lineage is notebook-centric; packaging into CLI modules would further improve reproducibility for large-scale reruns.
