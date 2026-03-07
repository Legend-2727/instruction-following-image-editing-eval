#!/usr/bin/env python
"""step3_vlm_judge.py — Use Qwen2.5-VL as an automated judge for image edits.

Evaluates each (source_img, instruction_en, target_img) triple and produces
structured 11-class error annotations saved to vlm_annotations.jsonl.

Usage
-----
    python scripts/step3_vlm_judge.py --data data/magicbrush
    python scripts/step3_vlm_judge.py --data data/magicbrush --model Qwen/Qwen2.5-VL-7B-Instruct --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_jsonl, append_jsonl, ensure_dirs
from utils.schema import ERROR_TYPES, ADHERENCE_LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert image editing quality evaluator. You will be given:
1. An ORIGINAL image (Image 1)
2. An EDITED image (Image 2) that was produced by an image editing model
3. An INSTRUCTION that the editing model was supposed to follow

Your task: evaluate whether the edit instruction was correctly followed and identify ALL errors.

ERROR TAXONOMY (select ALL that apply):
0. Wrong Object — edited the wrong object
1. Missing Object — a required object is missing from the edit
2. Extra Object — unwanted objects appeared
3. Wrong Attribute — object has wrong color/size/shape
4. Spatial Error — objects are mispositioned
5. Style Mismatch — visual style conflicts with original
6. Over-editing — too many regions changed beyond the instruction
7. Under-editing — instruction only partially applied
8. Artifact / Quality Issue — noise, blur, distortion, seams
9. Ambiguous Prompt — the instruction itself is unclear
10. Failed Removal — object that should be removed is still there

RESPOND WITH ONLY VALID JSON (no markdown):
{
  "adherence": "Success" | "Partial" | "No",
  "error_types": [<list of error type names from above, or [] if Success>],
  "error_label_vector": [0,0,0,0,0,0,0,0,0,0,0],
  "reasoning": "<1-3 sentence explanation>",
  "confidence": <float 0.0 to 1.0>
}
"""

USER_TEMPLATE = """\
INSTRUCTION: "{instruction}"

Evaluate whether Image 2 (edited) correctly follows this instruction compared to Image 1 (original).
Identify ALL errors from the taxonomy. Respond with ONLY valid JSON."""


class VLMJudge:
    """Qwen2.5-VL based edit quality judge."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info("Loading VLM judge: %s", model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        logger.info("VLM judge loaded: %s", model_id)

    def judge(self, source_img: Image.Image, target_img: Image.Image, instruction: str) -> dict:
        """Evaluate a single edit and return structured judgment."""
        from qwen_vl_utils import process_vision_info

        source_img = source_img.convert("RGB")
        target_img = target_img.convert("RGB")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": source_img},
                    {"type": "image", "image": target_img},
                    {"type": "text", "text": USER_TEMPLATE.format(instruction=instruction)},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        gen_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)
        ]
        raw_response = self.processor.batch_decode(
            gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        parsed = self._parse_response(raw_response)
        parsed["raw_response"] = raw_response
        parsed["model_name"] = self.model_id
        return parsed

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse VLM JSON response with fallback regex extraction."""
        result = {
            "adherence": "No",
            "error_types": [],
            "error_label_vector": [0] * 11,
            "reasoning": "",
            "confidence": 0.0,
        }

        try:
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            if isinstance(data, dict):
                # Adherence
                adh = data.get("adherence", "No")
                if adh in ADHERENCE_LABELS:
                    result["adherence"] = adh
                else:
                    for label in ADHERENCE_LABELS:
                        if label.lower() in str(adh).lower():
                            result["adherence"] = label
                            break

                # Error types
                raw_errors = data.get("error_types", [])
                if isinstance(raw_errors, list):
                    matched = []
                    for e in raw_errors:
                        if e in ERROR_TYPES:
                            matched.append(e)
                        elif isinstance(e, str):
                            e_lower = e.lower()
                            for et in ERROR_TYPES:
                                if et.lower() in e_lower or e_lower in et.lower():
                                    matched.append(et)
                                    break
                    result["error_types"] = matched

                # Build error vector from matched types
                vec = [0] * 11
                for e in result["error_types"]:
                    idx = ERROR_TYPES.index(e) if e in ERROR_TYPES else -1
                    if 0 <= idx < 11:
                        vec[idx] = 1
                result["error_label_vector"] = vec

                # Also try to use the model's own vector if provided
                if "error_label_vector" in data and isinstance(data["error_label_vector"], list):
                    if len(data["error_label_vector"]) == 11:
                        result["error_label_vector"] = [int(v) for v in data["error_label_vector"]]

                result["reasoning"] = str(data.get("reasoning", ""))
                try:
                    result["confidence"] = float(data.get("confidence", 0.0))
                except (ValueError, TypeError):
                    result["confidence"] = 0.0

                return result

        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: regex
        logger.warning("JSON parse failed, using fallback for: %s", raw[:200])
        for label in ADHERENCE_LABELS:
            if label.lower() in raw.lower():
                result["adherence"] = label
                break
        for i, et in enumerate(ERROR_TYPES):
            if et.lower() in raw.lower():
                result["error_types"].append(et)
                result["error_label_vector"][i] = 1
        result["reasoning"] = raw[:300]
        return result

    def unload(self):
        """Free GPU memory."""
        logger.info("Unloading VLM judge")
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="VLM Judge: annotate image edits")
    parser.add_argument("--data", type=str, default="data/magicbrush")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples to judge")
    args = parser.parse_args()

    data_dir = Path(args.data)
    meta_path = data_dir / "metadata.jsonl"
    out_path = data_dir / "vlm_annotations.jsonl"

    metadata = load_jsonl(meta_path)
    if not metadata:
        logger.error("No metadata at %s. Run step1_load_dataset.py first.", meta_path)
        sys.exit(1)

    # Resume support
    done_ids = set()
    if args.resume and out_path.exists():
        existing = load_jsonl(out_path)
        done_ids = {r["id"] for r in existing}
        logger.info("Resuming — %d already judged", len(done_ids))

    todo = [m for m in metadata if m["id"] not in done_ids]
    if args.max_samples:
        todo = todo[:args.max_samples]

    if not todo:
        logger.info("All samples already judged.")
        return

    logger.info("%d samples to judge", len(todo))

    # Load VLM
    judge = VLMJudge(model_id=args.model, device=args.device)
    ensure_dirs(out_path.parent)

    for meta in tqdm(todo, desc="VLM Judging"):
        uid = meta["id"]
        instruction = meta["instruction_en"]
        src_path = data_dir / meta["source_path"]
        tgt_path = data_dir / meta["target_path"]

        if not src_path.exists() or not tgt_path.exists():
            logger.warning("Missing images for %s — skipping", uid)
            continue

        src_img = Image.open(src_path)
        tgt_img = Image.open(tgt_path)

        t0 = time.time()
        try:
            result = judge.judge(src_img, tgt_img, instruction)
        except Exception as e:
            logger.warning("VLM failed for %s: %s", uid, e)
            continue
        elapsed = time.time() - t0

        annotation = {
            "id": uid,
            "instruction_en": instruction,
            "adherence": result["adherence"],
            "error_types": result["error_types"],
            "error_label_vector": result["error_label_vector"],
            "reasoning": result["reasoning"],
            "confidence": result["confidence"],
            "model_name": result["model_name"],
            "judge_time_sec": round(elapsed, 2),
        }

        append_jsonl(annotation, out_path)

    judge.unload()
    logger.info("All annotations saved → %s", out_path)


if __name__ == "__main__":
    main()
