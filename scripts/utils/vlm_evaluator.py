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
_SYSTEM_PROMPT_CONSERVATIVE = """\
You are an expert evaluator of instruction-following image edits.

You will be given:
1. The ORIGINAL image
2. The EDITED image
3. The edit INSTRUCTION

Return ONLY valid JSON.
Do not include markdown.
Do not include explanation outside JSON.
Do not include any prose before or after the JSON.

Schema:
{
  "adherence": "Success" | "Partial" | "No",
  "error_types": ["<error1>", "<error2>"],
  "confidence": <float between 0.0 and 1.0>
}

Rules:
- If the instruction is fully satisfied, use "Success" and [].
- If the instruction is partly satisfied, use "Partial".
- If the requested change is absent or clearly failed, use "No".
- Use at most 2 error types unless absolutely necessary.
- Prefer "Under-editing" when the requested edit is missing or too weak.
- Use "Wrong Attribute" only when the requested change is present but wrong.
- Use "Spatial Error" only when position / pose / placement is clearly wrong.
- Use "Artifact / Quality Issue" only for visible distortion, blur, or artifacts.
- If adherence is "Success", error_types must be [].

Allowed error_types:
[
  "Wrong Object",
  "Missing Object",
  "Extra Object",
  "Wrong Attribute",
  "Spatial Error",
  "Style Mismatch",
  "Over-editing",
  "Under-editing",
  "Artifact / Quality Issue",
  "Ambiguous Prompt",
  "Failed Removal"
]
"""

_SYSTEM_PROMPT_FAILURE_FOCUSED = """\
You are an expert evaluator of instruction-following image edits.

You will be given:
1. The ORIGINAL image
2. The EDITED image
3. The edit INSTRUCTION

Return ONLY valid JSON.
Do not include markdown.
Do not include explanation outside JSON.
Do not include any prose before or after the JSON.

Schema:
{
  "adherence": "Success" | "Partial" | "No",
  "error_types": ["<error1>", "<error2>"],
  "confidence": <float between 0.0 and 1.0>
}

Rules:
- Be slightly stricter than a normal evaluator.
- Look carefully for missing, weak, or incorrect edits.
- Use at most 2 error types unless absolutely necessary.
- Prefer "Under-editing" when the requested edit is missing or incomplete.
- Use "Wrong Attribute" only when the edited property is clearly wrong.
- Use "Spatial Error" only when location / pose / placement is clearly wrong.
- Use "Artifact / Quality Issue" only when visible visual corruption is present.
- If adherence is "Success", error_types must be [].

Allowed error_types:
[
  "Wrong Object",
  "Missing Object",
  "Extra Object",
  "Wrong Attribute",
  "Spatial Error",
  "Style Mismatch",
  "Over-editing",
  "Under-editing",
  "Artifact / Quality Issue",
  "Ambiguous Prompt",
  "Failed Removal"
]
"""

_USER_TEMPLATE = """\
Instruction: "{instruction}"

Evaluate whether the EDITED image follows the instruction relative to the ORIGINAL image.

Return ONLY valid JSON matching the schema."""


class QwenVLMJudge:
    """Wrapper for Qwen2.5-VL as an automated edit quality judge.
    
    Enforces strict JSON output with auditable failure tracking.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        prompt_type: str = "conservative",
    ):
        """Initialize the VLM judge.
        
        Args:
            prompt_type: 'conservative' (minimal labels) or 'failure_focused' (stricter).
        """
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info("Loading VLM judge: %s (prompt_type=%s)", model_id, prompt_type)

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
        self.prompt_type = prompt_type.lower()
        
        # Select system prompt based on type
        if self.prompt_type == "failure_focused":
            self.system_prompt = _SYSTEM_PROMPT_FAILURE_FOCUSED
        else:
            self.system_prompt = _SYSTEM_PROMPT_CONSERVATIVE

        logger.info("VLM judge loaded successfully")

    def evaluate(
        self,
        original_image: Image.Image,
        edited_image: Image.Image,
        instruction: str,
    ) -> dict:
        """Judge an edit and return a dict with strict JSON parsing.

        Returns dict with keys: adherence, error_types, confidence,
        raw_response, model_name, parse_failed.
        """
        from qwen_vl_utils import process_vision_info

        original_image = original_image.convert("RGB")
        edited_image = edited_image.convert("RGB")

        messages = [
            {"role": "system", "content": self.system_prompt},
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

    # ── Response parsing with strict validation ────────────────────────────────
    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse VLM response with strict JSON validation.
        
        Rules:
        - Attempt JSON parse (strips markdown code fences)
        - Validate adherence, error_types, confidence against canonical schema
        - Apply normalization: Success -> empty taxonomy, No + empty -> Under-editing
        - On parse failure, return dict with parse_failed=True and no fabricated labels
        """
        from utils.schema import ADHERENCE_LABELS, ERROR_TYPES, ADHERENCE_TO_IDX, ERROR_TO_IDX

        # Result template
        result = {
            "adherence": None,
            "error_types": [],
            "confidence": None,
            "parse_failed": True,
        }

        # Attempt JSON parse
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            # Validate adherence
            adh = data.get("adherence", None)
            if adh not in ADHERENCE_LABELS:
                logger.warning("Invalid adherence value: %r. Expected one of %s", adh, ADHERENCE_LABELS)
                return result  # parse_failed=True

            # Validate error_types
            raw_errors = data.get("error_types", [])
            if not isinstance(raw_errors, list):
                logger.warning("error_types is not a list: %r", raw_errors)
                return result

            validated_errors = []
            for error in raw_errors:
                if error not in ERROR_TYPES:
                    logger.warning("Unknown error type: %r. Valid: %s", error, ERROR_TYPES)
                    return result  # Strict: any invalid error -> parse failure
                validated_errors.append(error)

            # Validate confidence
            conf = data.get("confidence", None)
            try:
                conf = float(conf)
                if not (0.0 <= conf <= 1.0):
                    logger.warning("Confidence out of range [0.0, 1.0]: %f", conf)
                    return result
            except (ValueError, TypeError):
                logger.warning("Invalid confidence value: %r", conf)
                return result

            # ── Successful validation; apply normalization rules ──────────────
            # Rule 1: If Success, error_types must be empty
            if adh == "Success":
                validated_errors = []
            
            # Rule 2: Deduplicate and maintain canonical order
            validated_errors = sorted(
                list(set(validated_errors)),
                key=lambda e: ERROR_TO_IDX.get(e, 999)
            )
            
            # Rule 3: Cap at 2 error types unless rare edge case
            if len(validated_errors) > 2:
                logger.warning("More than 2 error types; capping to 2: %s", validated_errors)
                validated_errors = validated_errors[:2]
            
            # Rule 4: If No + empty taxonomy, assign Under-editing
            if adh == "No" and len(validated_errors) == 0:
                validated_errors = ["Under-editing"]

            result["adherence"] = adh
            result["error_types"] = validated_errors
            result["confidence"] = conf
            result["parse_failed"] = False
            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("JSON parse failed: %s. Raw: %s", e, raw[:200])
            return result  # parse_failed=True, no fabricated labels

    def unload(self):
        """Free GPU memory."""
        logger.info("Unloading VLM judge")
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
