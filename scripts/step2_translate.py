#!/usr/bin/env python
"""step2_translate.py — Translate English instructions to Nepali, Bangla, Hindi.

Uses Meta's NLLB-200 (No Language Left Behind) model which natively supports
all three target languages. Runs on GPU with float16 for speed.

Usage
-----
    python scripts/step2_translate.py --data data/magicbrush --model facebook/nllb-200-3.3B
    python scripts/step2_translate.py --data data/magicbrush --model facebook/nllb-200-distilled-1.3B  # faster
    python scripts/step2_translate.py --data data/magicbrush --resume  # continue interrupted run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_jsonl, save_jsonl, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# NLLB-200 language codes for our target languages
LANG_CODES = {
    "ne": "npi_Deva",   # Nepali
    "bn": "ben_Beng",   # Bangla
    "hi": "hin_Deva",   # Hindi
}
SRC_LANG = "eng_Latn"   # English source


class NLLBTranslator:
    """Wrapper for NLLB-200 translation model."""

    def __init__(self, model_id: str = "facebook/nllb-200-3.3B", device: str = "cuda"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading NLLB model: %s on %s", model_id, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        self.device = device
        logger.info("NLLB model loaded successfully")

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
        max_length: int = 256,
    ) -> list[str]:
        """Translate a batch of texts from src_lang to tgt_lang."""
        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=max_length,
            )

        translations = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return translations

    def unload(self):
        """Free GPU memory."""
        import gc
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Translate instructions to Nepali/Bangla/Hindi")
    parser.add_argument("--data", type=str, default="data/magicbrush")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    meta_path = data_dir / "metadata.jsonl"
    out_path = data_dir / "metadata_translated.jsonl"

    records = load_jsonl(meta_path)
    if not records:
        logger.error("No metadata found at %s", meta_path)
        sys.exit(1)

    logger.info("Loaded %d records from %s", len(records), meta_path)

    # Check for resume
    if args.resume and out_path.exists():
        translated = load_jsonl(out_path)
        translated_ids = {r["id"] for r in translated}
        # Find which records still need translation
        already_done = {r["id"]: r for r in translated}
        pending = [r for r in records if r["id"] not in translated_ids]
        if not pending:
            logger.info("All records already translated.")
            return
        logger.info("Resuming: %d already done, %d remaining", len(translated_ids), len(pending))
    else:
        already_done = {}
        pending = records

    # Load translator
    translator = NLLBTranslator(model_id=args.model, device=args.device)

    results = list(already_done.values())

    # Translate in batches for each language
    for lang_code, nllb_code in LANG_CODES.items():
        logger.info("Translating to %s (%s)...", lang_code, nllb_code)
        field_name = f"instruction_{lang_code}"

        all_texts = [r["instruction_en"] for r in pending]

        translated_texts = []
        for i in tqdm(range(0, len(all_texts), args.batch_size),
                      desc=f"Translating → {lang_code}"):
            batch = all_texts[i:i + args.batch_size]
            batch_translated = translator.translate_batch(
                batch, src_lang=SRC_LANG, tgt_lang=nllb_code
            )
            translated_texts.extend(batch_translated)

        # Attach translations to pending records
        for rec, trans in zip(pending, translated_texts):
            rec[field_name] = trans

    translator.unload()

    # Merge back
    for rec in pending:
        results.append(rec)

    # Sort by original order
    id_order = {r["id"]: i for i, r in enumerate(records)}
    results.sort(key=lambda r: id_order.get(r["id"], 999999))

    save_jsonl(results, out_path)
    logger.info("Saved translated metadata (%d records) → %s", len(results), out_path)

    # Also update the original metadata with translations
    save_jsonl(results, meta_path)
    logger.info("Updated original metadata with translations → %s", meta_path)

    # Print sample
    if results:
        sample = results[0]
        logger.info("--- Sample translation ---")
        logger.info("EN: %s", sample.get("instruction_en", ""))
        logger.info("NE: %s", sample.get("instruction_ne", ""))
        logger.info("BN: %s", sample.get("instruction_bn", ""))
        logger.info("HI: %s", sample.get("instruction_hi", ""))


if __name__ == "__main__":
    main()
