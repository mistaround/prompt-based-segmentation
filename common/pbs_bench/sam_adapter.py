"""Adapter for the SAM-style predictors.

Several of the models in this repo are forks of Meta's SAM that keep its
`sam_model_registry` + `SamPredictor` shape, so one adapter covers them all.
They differ only in how the mask-count argument is spelled, which `multimask_kw`
captures. `multimask_kw=None` covers a predictor that takes no such argument at
all and always returns its full candidate set (TinySAM): there the box prompt
also has to pick by predicted score, which is the closest available equivalent
to asking the others for a single mask.

(MobileSAM and EdgeSAM predate this module and keep their own adapters, since
their numbers were already measured and committed against those; everything
added afterwards uses this.)
"""
from __future__ import annotations

import pathlib
from typing import Dict

import numpy as np
import torch

from .runner import count_flops


class SamStyleAdapter:
    def __init__(
        self,
        name: str,
        variant: str,
        model,
        predictor,
        multimask_kw: str = "multimask_output",
        encoder_input_size: int = 1024,
        device: str = "cpu",
    ):
        self.name = name
        self.variant = variant
        self.model = model.to(device).eval()
        self.predictor = predictor
        self.multimask_kw = multimask_kw
        self.encoder_input_size = encoder_input_size
        self.device = device

    # `multimask_output=False` and `num_multimask_outputs=1` mean the same thing:
    # one mask for an unambiguous box prompt.
    def _single(self) -> Dict[str, object]:
        if self.multimask_kw is None:
            return {}
        return {self.multimask_kw: False if self.multimask_kw == "multimask_output" else 1}

    def _multi(self) -> Dict[str, object]:
        if self.multimask_kw is None:
            return {}
        return {self.multimask_kw: True if self.multimask_kw == "multimask_output" else 3}

    def set_image(self, bgr: np.ndarray) -> None:
        # ascontiguousarray, not just the ::-1 view: torchvision's ToTensor (used
        # by EfficientViT-SAM's preprocessing) rejects negative strides.
        self.predictor.set_image(np.ascontiguousarray(bgr[:, :, ::-1]))

    @staticmethod
    def _best(masks, scores) -> np.ndarray:
        masks = np.asarray(masks)
        scores = np.asarray(scores.cpu() if torch.is_tensor(scores) else scores)
        return masks[int(np.argmax(scores))].astype(bool)

    def predict_box(self, box: np.ndarray) -> np.ndarray:
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(box, dtype=np.float32), **self._single()
        )
        if self.multimask_kw is None:
            return self._best(masks, scores)
        return np.asarray(masks[0]).astype(bool)

    def predict_point(self, xy: np.ndarray) -> np.ndarray:
        # A single point is genuinely ambiguous (part / subpart / whole), so the
        # model emits three candidates; take its own highest-scoring one.
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray([xy], dtype=np.float32),
            point_labels=np.array([1]),
            **self._multi(),
        )
        return self._best(masks, scores)

    def param_stats(self) -> Dict[str, float]:
        def n(mod):
            return round(sum(p.numel() for p in mod.parameters()) / 1e6, 3)

        enc = n(self.model.image_encoder)
        dec = n(self.model.mask_decoder)
        pe = n(self.model.prompt_encoder)
        return {
            "total": round(enc + dec + pe, 3),
            "image_encoder": enc,
            "prompt_encoder": pe,
            "mask_decoder": dec,
        }

    def encoder_gflops(self):
        s = self.encoder_input_size
        return count_flops(self.model.image_encoder, (torch.zeros(1, 3, s, s),))


def checkpoint_mb(path) -> float | None:
    p = pathlib.Path(path) if path else None
    return round(p.stat().st_size / 1e6, 1) if p and p.is_file() else None
