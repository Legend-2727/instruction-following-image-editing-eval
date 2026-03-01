"""vlm_evaluator.py — VLM-as-judge using Qwen2.5-VL.

Sends (original image, edited image, instruction) to Qwen2.5-VL and gets
structured JSON judgments: adherence rating, error types, and reasoning.

Usage::

    from utils.vlm_evaluator import QwenVLMJudge
    judge = QwenVLMJudge("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    result = judge.evaluate(orig_img, edited_img, "make the sky sunset")
    print(result)  # VLMJudgment dataclass
"""

from __future__ import annotations

import gc
import json
import logging
import re
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ── System & user prompt templates ───────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert image editing evaluator. You will be given:
1. An ORIGINAL image (first image)
2. An EDITED image (second image)
3. An edit INSTRUCTION that was supposed to be applied

Your task: evaluate whether the edit instruction was correctly followed.

You MUST respond with ONLY valid JSON (no markdown fences, no extra text):
{
  "adherence": "<Success|Partial|No>",
  "error_types": ["<error1>", "<error2>"],
  "reasoning": "<1-2 sentence explanation>",
  "confidence": <0.0 to 1.0>
}

Adherence values:
- "Success": The edit fully matches the instruction
- "Partial": The edit partially follows the instruction but has issues
- "No": The edit does not follow the instruction at all

Error types (pick ALL that apply, or empty list if Success):
- "Wrong Object": Wrong object was modified
- "Missing Object": An object that should be added/present is missing
- "Extra Object": Unwanted objects were added
- "Wrong Attribute": Color/shape/size is wrong
- "Spatial Error": Spatial placement is wrong
- "Style Mismatch": Art style or visual style doesn't match
- "Over-editing": Too many changes beyond what was requested
- "Under-editing": Changes are too subtle or incomplete
- "Artifact / Quality Issue": Visual artifacts, blurriness, distortion
- "Ambiguous Prompt": The instruction itself is unclear
- "Failed Removal": Object that should be removed is still there
"""

_USER_TEMPLATE = """\
The edit instruction was: "{instruction}"

Evaluate whether the edited image (second image) correctly follows this instruction compared to the original image (first image).

Respond with ONLY valid JSON."""


class QwenVLMJudge:
    """Wrapper for Qwen2.5-VL as an automated edit quality judge."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info("Loading VLM judge: %s", model_id)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.device = device

        logger.info("VLM judge loaded successfully")

    def evaluate(
        self,
        original_image: Image.Image,
        edited_image: Image.Image,
        instruction: str,
    ) -> dict:
        """Judge an edit and return a dict suitable for VLMJudgment.from_dict().

        Returns dict with keys: adherence, error_types, reasoning,
        confidence, raw_response, model_name.
        """
        from qwen_vl_utils import process_vision_info

        original_image = original_image.convert("RGB")
        edited_image = edited_image.convert("RGB")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": edited_image},
                    {
                        "type": "text",
                        "text": _USER_TEMPLATE.format(instruction=instruction),
                    },
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
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = self._parse_response(raw_response)
        parsed["raw_response"] = raw_response
        parsed["model_name"] = self.model_id
        return parsed

    # ── Response parsing ─────────────────────────────────────────────────
    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse the VLM JSON response with fallback regex extraction."""
        from scripts.utils.schema import ADHERENCE_LABELS, ERROR_TYPES

        result = {
            "adherence": "No",
            "error_types": [],
            "reasoning": "",
            "confidence": 0.0,
        }

        # Try direct JSON parse
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            if isinstance(data, dict):
                # Adherence
                adh = data.get("adherence", "No")
                if adh in ADHERENCE_LABELS:
                    result["adherence"] = adh
                elif isinstance(adh, str):
                    # Fuzzy match
                    adh_lower = adh.lower().strip()
                    for label in ADHERENCE_LABELS:
                        if label.lower() in adh_lower:
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
                            # Fuzzy: check substring match
                            e_lower = e.lower()
                            for et in ERROR_TYPES:
                                if et.lower() in e_lower or e_lower in et.lower():
                                    matched.append(et)
                                    break
                    result["error_types"] = matched

                result["reasoning"] = str(data.get("reasoning", ""))

                conf = data.get("confidence", 0.0)
                try:
                    result["confidence"] = float(conf)
                except (ValueError, TypeError):
                    result["confidence"] = 0.0

                return result

        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: regex extraction from raw text
        logger.warning("JSON parse failed, using regex fallback for: %s", raw[:200])

        for label in ADHERENCE_LABELS:
            if label.lower() in raw.lower():
                result["adherence"] = label
                break

        for et in ERROR_TYPES:
            if et.lower() in raw.lower():
                result["error_types"].append(et)

        result["reasoning"] = raw[:200]
        return result

    def unload(self):
        """Free GPU memory."""
        logger.info("Unloading VLM judge")
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
