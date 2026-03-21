#!/usr/bin/env python
"""regenerate_translation_v2.py — regenerate multilingual translations with QA.

This script reads translation jobs created by ``build_translation_v2_manifest.py``
and writes one JSONL row per (sample_id, language) with:
- a freshly generated translation from English,
- optional back-translation into English,
- heuristic QA metadata,
- a machine-readable status for downstream review.

Typical usage
-------------
    # Smoke test without downloading a model
    python scripts/regenerate_translation_v2.py \
        --backend dummy \
        --jobs artifacts/translation_v2/translation_jobs.jsonl \
        --out artifacts/translation_v2/translation_results_smoke.jsonl \
        --limit 6

    # Real run on GPU with NLLB 1.3B
    python scripts/regenerate_translation_v2.py \
        --backend nllb \
        --model facebook/nllb-200-distilled-1.3B \
        --device cuda \
        --dtype float32 \
        --jobs artifacts/translation_v2/translation_jobs.jsonl \
        --out artifacts/translation_v2/translation_results_nllb13b.jsonl \
        --batch_size 8 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import ensure_dirs, load_jsonl, append_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LANG_CODES = {
    "en": "eng_Latn",
    "bn": "ben_Beng",
    "hi": "hin_Deva",
    "ne": "npi_Deva",
}

SCRIPT_BLOCKS = {
    "bn": [(0x0980, 0x09FF)],
    "hi": [(0x0900, 0x097F)],
    "ne": [(0x0900, 0x097F)],
}

EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]{2,}")
SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate translation_v2 rows with QA")
    parser.add_argument("--jobs", type=str, default="artifacts/translation_v2/translation_jobs.jsonl")
    parser.add_argument("--out", type=str, default="artifacts/translation_v2/translation_results.jsonl")
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--langs", nargs="+", default=["bn", "hi", "ne"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=["nllb", "dummy"], default="nllb")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--skip_backtranslation", action="store_true")
    parser.add_argument("--only_status", nargs="+", default=["pending"])
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", str(text).strip())


def normalize_for_similarity(text: str) -> str:
    text = NON_ALNUM_RE.sub(" ", str(text).lower())
    return SPACE_RE.sub(" ", text).strip()


def is_expected_script(ch: str, lang: str) -> bool:
    if lang not in SCRIPT_BLOCKS:
        return True
    code = ord(ch)
    return any(start <= code <= end for start, end in SCRIPT_BLOCKS[lang])


def script_char_ratio(text: str, lang: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    count = sum(1 for ch in letters if is_expected_script(ch, lang))
    return count / len(letters)


def latin_char_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    count = sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, ""))
    return count / len(letters)


def english_tokens(text: str) -> List[str]:
    return EN_TOKEN_RE.findall(text)


def similarity(a: str, b: str) -> float:
    aa = normalize_for_similarity(a)
    bb = normalize_for_similarity(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def length_ratio(source: str, target: str) -> float:
    s = len("".join(str(source).split()))
    t = len("".join(str(target).split()))
    if s == 0:
        return 0.0
    return t / s


def length_score(ratio: float) -> float:
    if 0.45 <= ratio <= 1.90:
        return 1.0
    if 0.35 <= ratio <= 2.30:
        return 0.5
    return 0.0


class BaseTranslator:
    name = "base"

    def translate_batch(self, texts: Sequence[str], src_lang: str, tgt_lang: str) -> List[str]:
        raise NotImplementedError

    def unload(self) -> None:
        return None


class DummyTranslator(BaseTranslator):
    name = "dummy"

    def translate_batch(self, texts: Sequence[str], src_lang: str, tgt_lang: str) -> List[str]:
        out: List[str] = []
        for text in texts:
            if tgt_lang == LANG_CODES["en"]:
                out.append(f"BACKTRANS[{src_lang}->{tgt_lang}] {normalize_spaces(text)}")
            else:
                out.append(f"[{tgt_lang}] {normalize_spaces(text)}")
        return out


class NLLBTranslator(BaseTranslator):
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        num_beams: int,
        max_source_length: int,
        max_new_tokens: int,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = model_id
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.num_beams = num_beams
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_kwargs = {}
        if dtype != "auto":
            model_kwargs["dtype"] = dtype_map[dtype]

        logger.info("Loading tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("Loading model: %s on %s (dtype=%s)", model_id, device, dtype)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)
        self.model = self.model.to(device).eval()

    def translate_batch(self, texts: Sequence[str], src_lang: str, tgt_lang: str) -> List[str]:
        if not texts:
            return []
        self.tokenizer.src_lang = src_lang
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "torch") and self.device == "cuda":
            self.torch.cuda.empty_cache()


def build_translator(args: argparse.Namespace) -> BaseTranslator:
    if args.backend == "dummy":
        return DummyTranslator()
    return NLLBTranslator(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
    )


def qa_translation(
    source_en: str,
    translated_text: str,
    backtranslated_en: str,
    lang: str,
) -> Dict[str, object]:
    src = normalize_spaces(source_en)
    tgt = normalize_spaces(translated_text)
    back = normalize_spaces(backtranslated_en)

    script_ratio = script_char_ratio(tgt, lang)
    latin_ratio = latin_char_ratio(tgt)
    source_target_similarity = similarity(src, tgt)
    back_similarity = similarity(src, back)
    tgt_len_ratio = length_ratio(src, tgt)
    eng_tokens = english_tokens(tgt)

    notes: List[str] = []
    if eng_tokens:
        preview = ", ".join(eng_tokens[:6])
        notes.append(f"latin_tokens={len(eng_tokens)} [{preview}]")
    notes.append(f"script_ratio={script_ratio:.3f}")
    notes.append(f"latin_ratio={latin_ratio:.3f}")
    notes.append(f"len_ratio={tgt_len_ratio:.3f}")
    notes.append(f"back_sim={back_similarity:.3f}")
    notes.append(f"copy_sim={source_target_similarity:.3f}")

    score = (
        0.45 * back_similarity
        + 0.25 * script_ratio
        + 0.15 * (1.0 - min(latin_ratio, 1.0))
        + 0.15 * length_score(tgt_len_ratio)
    )

    hard_fail = False
    if not tgt:
        hard_fail = True
        notes.append("empty_translation")
    if script_ratio < 0.40:
        hard_fail = True
        notes.append("low_target_script_ratio")
    if back and back_similarity < 0.42:
        hard_fail = True
        notes.append("weak_backtranslation_alignment")
    if source_target_similarity > 0.82:
        hard_fail = True
        notes.append("translation_too_close_to_english")
    if tgt_len_ratio < 0.30 or tgt_len_ratio > 2.60:
        hard_fail = True
        notes.append("extreme_length_ratio")

    needs_review = False
    if len(eng_tokens) >= 4:
        needs_review = True
        notes.append("many_latin_tokens")
    if latin_ratio > 0.22:
        needs_review = True
        notes.append("high_latin_ratio")
    if score < 0.72:
        needs_review = True
        notes.append("low_qa_score")
    if not (0.45 <= tgt_len_ratio <= 1.90):
        needs_review = True
        notes.append("length_ratio_outside_soft_band")

    if hard_fail:
        flag = "qa_fail"
    elif needs_review:
        flag = "qa_review"
    else:
        flag = "qa_pass"

    return {
        "qa_score": round(float(score), 4),
        "qa_flag": flag,
        "qa_notes": "; ".join(dict.fromkeys(notes)),
        "metrics": {
            "script_ratio": round(script_ratio, 4),
            "latin_ratio": round(latin_ratio, 4),
            "length_ratio": round(tgt_len_ratio, 4),
            "backtranslation_similarity": round(back_similarity, 4),
            "source_target_similarity": round(source_target_similarity, 4),
            "english_token_count": len(eng_tokens),
        },
    }


def load_jobs_for_processing(
    jobs_path: Path,
    out_path: Path,
    langs: Sequence[str],
    only_status: Sequence[str],
    limit: int | None,
    resume: bool,
) -> List[Dict]:
    jobs = load_jsonl(jobs_path)
    if not jobs:
        raise FileNotFoundError(f"No translation jobs found at {jobs_path}")

    done = set()
    if resume and out_path.exists():
        existing = load_jsonl(out_path)
        done = {(row["id"], row["lang"]) for row in existing}
        logger.info("Resume enabled: %d existing translations in %s", len(done), out_path)

    selected: List[Dict] = []
    for row in jobs:
        key = (row.get("id"), row.get("lang"))
        if row.get("lang") not in langs:
            continue
        if row.get("translation_status", "pending") not in only_status:
            continue
        if resume and key in done:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def batched(items: Sequence[Dict], batch_size: int) -> Iterable[Sequence[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def process_jobs(args: argparse.Namespace) -> Tuple[Counter, Path]:
    jobs_path = Path(args.jobs)
    out_path = Path(args.out)
    ensure_dirs(out_path.parent)
    summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")

    todo = load_jobs_for_processing(
        jobs_path=jobs_path,
        out_path=out_path,
        langs=args.langs,
        only_status=args.only_status,
        limit=args.limit,
        resume=args.resume,
    )
    if not todo:
        logger.info("No translation jobs selected. Nothing to do.")
        return Counter(), summary_out

    logger.info("Selected %d jobs from %s", len(todo), jobs_path)
    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for row in todo:
        by_lang[row["lang"]].append(row)

    translator = build_translator(args)
    counters: Counter = Counter()

    try:
        for lang in args.langs:
            rows = by_lang.get(lang, [])
            if not rows:
                continue
            tgt_code = LANG_CODES[lang]
            logger.info("Processing %d jobs for %s (%s)", len(rows), lang, tgt_code)

            for batch in batched(rows, args.batch_size):
                source_texts = [normalize_spaces(row["instruction_en"]) for row in batch]
                translations = translator.translate_batch(source_texts, LANG_CODES["en"], tgt_code)
                if args.skip_backtranslation:
                    backtranslations = [""] * len(translations)
                else:
                    backtranslations = translator.translate_batch(translations, tgt_code, LANG_CODES["en"])

                for row, translated_text, backtranslated_en in zip(batch, translations, backtranslations):
                    qa = qa_translation(
                        source_en=row["instruction_en"],
                        translated_text=translated_text,
                        backtranslated_en=backtranslated_en,
                        lang=lang,
                    )
                    result = dict(row)
                    result.update(
                        {
                            "translation_source_text": row["instruction_en"],
                            "translated_text_v2": normalize_spaces(translated_text),
                            "backtranslate_to_en": normalize_spaces(backtranslated_en),
                            "translation_model": translator.name,
                            "qa_method": "heuristic_backtranslation_v1",
                            "qa_score": qa["qa_score"],
                            "qa_flag": qa["qa_flag"],
                            "qa_notes": qa["qa_notes"],
                            "qa_metrics": qa["metrics"],
                            "translation_status": qa["qa_flag"],
                            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    append_jsonl(result, out_path)
                    counters[f"lang:{lang}"] += 1
                    counters[f"status:{qa['qa_flag']}"] += 1
    finally:
        translator.unload()

    summary = {
        "jobs_path": str(jobs_path),
        "out_path": str(out_path),
        "model": translator.name,
        "backend": args.backend,
        "langs": list(args.langs),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "resume": bool(args.resume),
        "skip_backtranslation": bool(args.skip_backtranslation),
        "counts": dict(counters),
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved summary -> %s", summary_out)
    return counters, summary_out


def main() -> None:
    args = parse_args()
    counters, summary_out = process_jobs(args)
    if counters:
        logger.info("Completed translation run: %s", dict(counters))
    else:
        logger.info("No new rows written. Summary: %s", summary_out)


if __name__ == "__main__":
    main()
