"""editor_model.py — Wrapper for instruction-based image editing models.

Supports:
  - InstructPix2Pix  (timbrooks/instruct-pix2pix)

Usage::

    from utils.editor_model import load_editor
    editor = load_editor("timbrooks/instruct-pix2pix", device="cuda")
    edited = editor.edit(pil_image, "make the sky sunset")
    edited.save("out.png")
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class InstructPix2PixEditor:
    """Thin wrapper around the HuggingFace diffusers InstructPix2Pix pipeline."""

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        num_inference_steps: int = 20,
    ):
        from diffusers import (
            StableDiffusionInstructPix2PixPipeline,
            EulerAncestralDiscreteScheduler,
        )

        logger.info("Loading editor model: %s (dtype=%s)", model_id, torch_dtype)

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        self.pipe.to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        # Memory optimization for smaller GPUs
        self.pipe.enable_attention_slicing()

        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.model_id = model_id

        logger.info("Editor model loaded successfully on %s", device)

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        guidance_scale: Optional[float] = None,
        image_guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Image.Image:
        """Apply an instruction-based edit to *image* and return the result."""
        gs = guidance_scale or self.guidance_scale
        igs = image_guidance_scale or self.image_guidance_scale
        steps = num_inference_steps or self.num_inference_steps

        image = image.convert("RGB")

        result = self.pipe(
            prompt=prompt,
            image=image,
            guidance_scale=gs,
            image_guidance_scale=igs,
            num_inference_steps=steps,
        ).images[0]

        return result

    def unload(self):
        """Free GPU memory by deleting the pipeline."""
        logger.info("Unloading editor model from %s", self.device)
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_editor(
    model_id: str = "timbrooks/instruct-pix2pix",
    device: str = "cuda",
    torch_dtype: str = "float16",
    **kwargs,
) -> InstructPix2PixEditor:
    """Convenience factory: creates an editor from a config-friendly dtype string."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return InstructPix2PixEditor(
        model_id=model_id,
        device=device,
        torch_dtype=dtype_map.get(torch_dtype, torch.float16),
        **kwargs,
    )
