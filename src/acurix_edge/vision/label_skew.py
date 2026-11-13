"""Specialist "label skew" — deterministic geometric rule (no training), AEX-CV-007.

Measures the angle between the label axis and the product axis (Anexo 04A §1: a rotated product
with an aligned label is NOT skewed):
1. product = dominant object of the frame (same method as the quality gate: border background,
   largest contour, filled);
2. label   = largest region INSIDE the (eroded) product whose gray differs from the product's
   median gray by more than `label_delta`;
3. axes    = orientation of the principal axis from image moments (area-based, sub-pixel);
   a near-square product (elongation < product_square_elongation) uses its minimum-area rectangle;
4. skew    = label angle - product angle, wrapped to [-45°, 45°) (axes are defined modulo 90°:
   a wide label on a tall product is aligned). Positive = counter-clockwise as displayed.
   Limit: skews at or beyond ±45° cannot be told apart from the perpendicular alignment.

Output: InferenceResult (schema 1.2) with ONE prediction: class LABEL_SKEWED if |skew| >
tolerance else OK; attributes angle_deg, abs_angle_deg, borderline (tolerance band), product and
label angles, tolerance values and params_status ("provisional"). `confidence` is a TECHNICAL
confidence of the geometric detection (label solidity x elongation), never a probability.
Label not found / ambiguous shape -> status INCONCLUSIVE (this specialist never decides "missing").
Parameters come ONLY from a verified artifact (CAS-005); approval comes only from ModelPolicy.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from acurix_edge.capture.sources import Image
from acurix_edge.contracts.inference import (
    ImageQuality,
    InferenceRequest,
    InferenceResult,
    InferenceStatus,
    InputDescriptor,
    ModelRef,
    Prediction,
    RuntimeRef,
)
from acurix_edge.domain.artifacts import VerifiedArtifact
from acurix_edge.vision.geometry import moment_axis, object_axis_deg, rectangularity, soft_weights, wrap90
from acurix_edge.vision.quality_gate import MIN_RECTANGULARITY, find_object, to_gray

ARTIFACT_ID = "label-skew-geometric"
STAGE_KIND = "specialist"
CLASS_OK = "OK"
CLASS_SKEWED = "LABEL_SKEWED"
RUNTIME = RuntimeRef(name="opencv-geometric", version=cv2.__version__, device="cpu")
_PARAM_KEYS = {"id", "version", "kind", "status", "note", "skew_tolerance_deg", "borderline_low_deg",
               "object_delta", "label_delta", "product_erode_px", "min_label_area_fraction",
               "max_label_area_fraction", "min_label_elongation", "product_square_elongation"}


class ParamsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SkewParams:
    status: str
    skew_tolerance_deg: float
    borderline_low_deg: float
    object_delta: float
    label_delta: float
    product_erode_px: int
    min_label_area_fraction: float
    max_label_area_fraction: float
    min_label_elongation: float
    product_square_elongation: float

    @classmethod
    def from_artifact(cls, artifact: VerifiedArtifact) -> SkewParams:
        try:
            raw = json.loads(artifact.data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ParamsError(f"parameters file is not JSON ({type(exc).__name__})") from exc
        if not isinstance(raw, dict) or set(raw) != _PARAM_KEYS:
            raise ParamsError(f"parameters must have exactly the keys {sorted(_PARAM_KEYS)}")
        if raw["id"] != artifact.ref.id or raw["version"] != artifact.ref.version:
            raise ParamsError("parameters id/version differ from the verified artifact")
        nums = ("skew_tolerance_deg", "borderline_low_deg", "object_delta", "label_delta",
                "min_label_area_fraction", "max_label_area_fraction", "min_label_elongation",
                "product_square_elongation")
        for k in nums:
            v = raw[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ParamsError(f"{k} must be a finite number >= 0")
        if raw["status"] != "provisional":
            raise ParamsError("status must be 'provisional' in 0.3.0 (approval never comes from this file)")
        erode = raw["product_erode_px"]
        if isinstance(erode, bool) or not isinstance(erode, int) or not 0 <= erode <= 20:
            raise ParamsError("product_erode_px must be int in 0..20")
        p = cls(status=raw["status"], product_erode_px=erode, **{k: float(raw[k]) for k in nums})
        if not 0 < p.borderline_low_deg <= p.skew_tolerance_deg < 45:
            raise ParamsError("need 0 < borderline_low_deg <= skew_tolerance_deg < 45")
        if not 0 < p.min_label_area_fraction < p.max_label_area_fraction <= 1:
            raise ParamsError("need 0 < min_label_area_fraction < max_label_area_fraction <= 1")
        if p.min_label_elongation <= 1 or p.product_square_elongation < 1:
            raise ParamsError("need min_label_elongation > 1 and product_square_elongation >= 1")
        return p


MIN_PRODUCT_RECTANGULARITY = MIN_RECTANGULARITY  # same limit as the gate: no reliable reference axis below


@dataclass(frozen=True, slots=True)
class SkewMeasurement:
    skew_deg: float
    product_angle_deg: float
    label_angle_deg: float
    label_bbox: tuple[int, int, int, int]
    label_area_fraction: float
    label_elongation: float
    label_solidity: float
    technical_confidence: float


def measure(image: Image, params: SkewParams) -> SkewMeasurement | str:
    """Return the measurement, or an INCONCLUSIVE reason code."""
    gray = to_gray(image)
    product = find_object(gray, params.object_delta)
    if product is None:
        return "PRODUCT_NOT_FOUND"
    if rectangularity(product.contour) < MIN_PRODUCT_RECTANGULARITY:
        return "PRODUCT_SHAPE_AMBIGUOUS"  # no reliable reference axis: never a spurious OK/SKEWED
    k = max(1, 2 * params.product_erode_px + 1)
    inner = cv2.erode(product.mask, np.ones((k, k), np.uint8))
    inner_px = gray[inner > 0]
    if inner_px.size == 0:
        return "PRODUCT_TOO_SMALL"
    product_level = float(np.median(inner_px))
    diff = np.abs(gray.astype(np.int16) - int(round(product_level))).astype(np.uint8)
    label = np.where((diff > params.label_delta) & (inner > 0), 255, 0).astype(np.uint8)
    label = np.asarray(cv2.morphologyEx(label, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), dtype=np.uint8)
    contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return "LABEL_NOT_FOUND"
    biggest = max(contours, key=cv2.contourArea)
    label_mask = np.zeros_like(gray)
    cv2.drawContours(label_mask, [biggest], -1, 255, thickness=cv2.FILLED)
    area = float(cv2.countNonZero(label_mask))
    fraction = area / product.area
    if not params.min_label_area_fraction <= fraction <= params.max_label_area_fraction:
        return "LABEL_NOT_FOUND"
    # Sub-pixel axes: soft membership weights around each hard mask (contrast = median |diff| inside).
    label_contrast = float(np.median(diff[label_mask > 0]))
    soft_label = soft_weights(gray, label_mask, product_level, label_contrast)
    soft_label[inner == 0] = 0.0  # never take product-edge pixels as label
    product_contrast = float(abs(product_level - product.background_level))
    soft_product = soft_weights(gray, product.mask, product.background_level, product_contrast)
    soft_product[product.mask > 0] = 1.0  # the label inside the product is still product
    label_axis = moment_axis(soft_label)
    if label_axis is None:
        return "LABEL_NOT_FOUND"
    if label_axis.elongation < params.min_label_elongation:
        return "LABEL_SHAPE_AMBIGUOUS"  # near-square label: its axis is undefined
    product_angle = object_axis_deg(soft_product, product.contour, params.product_square_elongation)
    (_, _), (rw, rh), _ = cv2.minAreaRect(biggest)
    solidity = min(1.0, area / max(1.0, float(rw) * float(rh)))
    elong_factor = min(1.0, (label_axis.elongation - 1.0) / 1.0)
    confidence = max(0.0, min(1.0, solidity * elong_factor))
    x, y, w, h = cv2.boundingRect(biggest)
    return SkewMeasurement(skew_deg=wrap90(label_axis.angle_deg - product_angle),
                           product_angle_deg=product_angle, label_angle_deg=label_axis.angle_deg,
                           label_bbox=(x, y, w, h), label_area_fraction=fraction,
                           label_elongation=label_axis.elongation, label_solidity=solidity,
                           technical_confidence=round(confidence, 4))


class LabelSkewSpecialist:
    """First implementation of the InspectionStage interface (domain.stages)."""

    stage_kind = STAGE_KIND

    def __init__(self, artifact: VerifiedArtifact) -> None:
        if artifact.ref.id != ARTIFACT_ID:
            raise ParamsError(f"artifact {artifact.ref.id} is not {ARTIFACT_ID}")
        self.params = SkewParams.from_artifact(artifact)
        self._artifact = artifact
        self.stage_id = f"{STAGE_KIND}:{ARTIFACT_ID}"

    @property
    def model(self) -> ModelRef:
        return self._artifact.ref

    @property
    def approved(self) -> bool:
        return self._artifact.approved  # from ModelPolicy only (always False in 0.3.0)

    def run(self, request: InferenceRequest) -> InferenceResult:
        started = time.perf_counter()
        result = measure(request.frame, self.params)
        elapsed = (time.perf_counter() - started) * 1000.0
        common: dict[str, Any] = dict(inspection_id=request.inspection_id, runtime=RUNTIME, input=request.input,
                                      model=self.model, inference_ms=elapsed)
        if isinstance(result, str):
            return InferenceResult(status=InferenceStatus.INCONCLUSIVE, error_code=result, **common)
        p = self.params
        abs_skew = abs(result.skew_deg)
        cls = CLASS_SKEWED if abs_skew > p.skew_tolerance_deg else CLASS_OK
        borderline = p.borderline_low_deg <= abs_skew <= p.skew_tolerance_deg
        x, y, w, h = result.label_bbox
        pred = Prediction(class_id=cls, confidence=result.technical_confidence, bbox=(x, y, w, h), attributes={
            "angle_deg": round(result.skew_deg, 3), "abs_angle_deg": round(abs_skew, 3),
            "borderline": borderline, "product_angle_deg": round(result.product_angle_deg, 3),
            "label_angle_deg": round(result.label_angle_deg, 3),
            "label_area_fraction": round(result.label_area_fraction, 4),
            "label_elongation": round(min(result.label_elongation, 1e6), 3),
            "tolerance_deg": p.skew_tolerance_deg, "borderline_low_deg": p.borderline_low_deg,
            "params_status": p.status, "confidence_kind": "technical_geometric_not_probability"})
        return InferenceResult(status=InferenceStatus.OK, predictions=(pred,), **common)


def valid_input(width: int, height: int) -> InputDescriptor:
    return InputDescriptor(width, height, ImageQuality.VALID)
