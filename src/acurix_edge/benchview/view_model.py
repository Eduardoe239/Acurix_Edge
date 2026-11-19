"""Pure functions that turn the engine state into the JSON the bench page renders.

SAFETY RULES enforced HERE (one place, property-tested):
- a conclusive verdict (PASS / DEFECT) is NEVER put in the output: if the system ever produced
  one, the page shows "NÃO EXIBIDO" (this tool is not approved to present conclusive results);
- the specialist's output is labelled "Prévia (modo sombra) — não decide" and its class is shown
  only as neutral text (never as a verdict);
- texts are pt-BR, numbers with decimal comma; every value is a plain string/number (no markup).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from acurix_edge.config.settings import QualityConfig

SHADOW_TITLE = "Prévia (modo sombra) — não decide"
VERDICT_PHRASE = "O sistema ainda não decide: especialista em validação"
PRIVACY_NOTICE = "Câmera apontada só para a bancada. Nenhuma imagem é gravada."
TOOL_NOTICE = "Ferramenta de desenvolvimento (AEX-BENCH-VIEW-001) — não é o console da release 0.5.0."
OWNER = "Luís Eduardo de Sousa Barbosa"
NO_DECISION = "NO_DECISION"
REVIEW = "REVIEW"
HIDDEN_VERDICT = "NÃO EXIBIDO"
DISPLAYABLE_VERDICTS = frozenset({NO_DECISION, REVIEW})  # PASS/DEFECT are never displayable here
RESET_HELP = ("Reconecte o cabo USB, encerre o visualizador e execute "
              "python -m acurix_edge reset-fault --reason \"<motivo>\" (mesma pasta). "
              "Depois inicie o visualizador de novo.")

QUALITY_REASON_PT = {
    "FOCUS_LOW": "Imagem sem foco",
    "UNDEREXPOSED": "Imagem escura demais",
    "OVEREXPOSED": "Imagem clara demais",
    "CRUSHED_SHADOWS": "Sombras saturadas",
    "BLOWN_HIGHLIGHTS": "Áreas estouradas de luz",
    "REFLECTION": "Reflexo na peça",
    "BACKGROUND_OUT_OF_RANGE": "Fundo fora do gabarito",
    "BACKGROUND_NOT_UNIFORM": "Fundo não uniforme",
    "OBJECT_NOT_FOUND": "Peça não encontrada no gabarito",
    "OBJECT_TOO_LARGE": "Peça perto demais da câmera",
    "OBJECT_OUTSIDE_ROI": "Peça fora da área do gabarito",
    "OBJECT_ROTATED": "Peça girada além do limite",
    "QUALITY_GATE_ERROR": "Erro na verificação de qualidade",
}
SHADOW_INCONCLUSIVE_PT = {
    "PRODUCT_NOT_FOUND": "peça não encontrada",
    "PRODUCT_SHAPE_AMBIGUOUS": "contorno da peça ambíguo",
    "PRODUCT_TOO_SMALL": "peça pequena demais",
    "LABEL_NOT_FOUND": "etiqueta não encontrada",
    "LABEL_SHAPE_AMBIGUOUS": "formato da etiqueta ambíguo",
}
DECISION_REASON_PT = {
    "INPUT_QUALITY": "imagem recusada pela porta de qualidade",
    "INFERENCE_NOT_OK": "nenhum modelo aprovado para decidir",
    "NO_APPROVED_RULE": "nenhuma regra de decisão aprovada",
    "HEALTH_NOT_OK": "saúde do sistema fora do normal",
    "SYSTEM_NOT_OBSERVING": "sistema fora de observação",
    "RULE_ERROR": "erro na regra de decisão",
    "EVENT_NOT_PERSISTED": "evento não registrado",
    "EVENT_RECORD_CORRUPT": "registro de evento corrompido",
    "IDEMPOTENCY_CONFLICT": "conflito de identificação da inspeção",
}


def fmt(value: float, decimals: int = 1) -> str:
    """pt-BR number: decimal comma, no thousands separator."""
    if not math.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


def _pct(value: float, lo: float, hi: float) -> float:
    if hi <= lo or not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(100.0, (value - lo) / (hi - lo) * 100.0)), 2)


def limit_bar(label: str, value: float, value_text: str, *, scale: tuple[float, float],
              thresholds: Sequence[float], band: tuple[float, float] | None, attention: bool,
              scale_text: tuple[str, str], detail: str = "") -> dict[str, Any]:
    """Data for one `ax-limit` bar: every position already in percent of the scale."""
    lo, hi = scale
    return {"label": label, "value_text": value_text, "value_pct": _pct(value, lo, hi),
            "thresholds_pct": [_pct(t, lo, hi) for t in thresholds],
            "band_pct": [_pct(band[0], lo, hi), _pct(band[1], lo, hi)] if band else None,
            "attention": attention, "scale_text": list(scale_text), "detail": detail}


# -- verdict --------------------------------------------------------------------------------
def verdict_view(result: str | None, reason: str | None) -> dict[str, Any]:
    """The REAL system verdict, filtered: never PASS/DEFECT (shown as NÃO EXIBIDO)."""
    if result is None:
        return {"value": NO_DECISION, "modifier": "none", "phrase": VERDICT_PHRASE,
                "reason": "aguardando o primeiro quadro", "hidden": False}
    if result not in DISPLAYABLE_VERDICTS:
        return {"value": HIDDEN_VERDICT, "modifier": "none",
                "phrase": "Resultado conclusivo não exibido: esta ferramenta não está aprovada para isso.",
                "reason": "", "hidden": True}
    return {"value": result, "modifier": "review" if result == REVIEW else "none", "phrase": VERDICT_PHRASE,
            "reason": DECISION_REASON_PT.get(reason or "", reason or ""), "hidden": False}


# -- quality gate ---------------------------------------------------------------------------
def quality_view(quality: Mapping[str, Any] | None, q: QualityConfig) -> dict[str, Any]:
    if not quality:
        return {"status": "—", "valid": None, "reasons": [], "bars": [], "roi": None}
    metrics: Mapping[str, float] = quality.get("metrics") or {}
    failing = {str(r.get("metric")) for r in quality.get("reasons") or []}
    reasons = [QUALITY_REASON_PT.get(str(r.get("code")), str(r.get("code"))) for r in quality.get("reasons") or []]

    def m(key: str) -> float:
        v = metrics.get(key, float("nan"))
        return float(v) if isinstance(v, (int, float)) else float("nan")

    sharp_hi = q.min_sharpness * 4
    bright_frac, dark_frac = m("bright_fraction"), m("dark_fraction")
    rotation = m("object_rotation_abs_deg")
    outside = m("object_outside_roi_px")
    found = m("object_area_fraction_roi") >= q.min_object_fraction
    if not found:
        position_detail = "peça não encontrada no gabarito"
    elif m("object_rotation_measured") < 1:
        position_detail = "rotação não medida (contorno pouco retangular)"
    else:
        position_detail = "dentro da área do gabarito" if outside <= 0 else f"{fmt(outside, 0)} px fora da área"
    bars = [
        limit_bar("Nitidez", m("sharpness_laplacian_var"), fmt(m("sharpness_laplacian_var"), 0),
                  scale=(0, sharp_hi), thresholds=[q.min_sharpness], band=(q.min_sharpness, sharp_hi),
                  attention="sharpness_laplacian_var" in failing,
                  scale_text=("0", f"{fmt(sharp_hi, 0)}+"), detail=f"mínimo {fmt(q.min_sharpness, 0)}"),
        limit_bar("Brilho", m("brightness_mean"), fmt(m("brightness_mean"), 0), scale=(0, 255),
                  thresholds=[q.min_brightness, q.max_brightness], band=(q.min_brightness, q.max_brightness),
                  attention="brightness_mean" in failing, scale_text=("0", "255"),
                  detail=f"faixa {fmt(q.min_brightness, 0)}–{fmt(q.max_brightness, 0)}"),
        limit_bar("Saturação", bright_frac * 100, f"{fmt(bright_frac * 100)} %",
                  scale=(0, q.max_bright_fraction * 200), thresholds=[q.max_bright_fraction * 100],
                  band=(0, q.max_bright_fraction * 100),
                  attention=bool(failing & {"bright_fraction", "dark_fraction", "specular_fraction_roi"}),
                  scale_text=("0 %", f"{fmt(q.max_bright_fraction * 200, 0)} %"),
                  detail=f"áreas estouradas; sombras saturadas {fmt(dark_frac * 100)} %"),
        limit_bar("Posição", rotation, f"{fmt(rotation)}°", scale=(0, q.max_object_rotation_deg * 2),
                  thresholds=[q.max_object_rotation_deg], band=(0, q.max_object_rotation_deg),
                  attention=bool(failing & {"object_rotation_abs_deg", "object_outside_roi_px",
                                            "object_area_fraction_roi"}),
                  scale_text=("0°", f"{fmt(q.max_object_rotation_deg * 2, 0)}°"), detail=position_detail),
    ]
    status = str(quality.get("status", "—"))
    return {"status": status, "valid": status == "VALID", "reasons": reasons, "bars": bars}


# -- shadow specialist ----------------------------------------------------------------------
def shadow_view(stages: Sequence[Mapping[str, Any]] | None, quality_valid: bool | None) -> dict[str, Any]:
    base: dict[str, Any] = {"title": SHADOW_TITLE, "bar": None, "text": "", "angle_text": "—"}
    if quality_valid is None:
        return {**base, "text": "aguardando o primeiro quadro"}
    if not quality_valid or not stages:
        return {**base, "text": "Não executada: a imagem foi recusada pela porta de qualidade."}
    run = stages[0]
    result: Mapping[str, Any] = run.get("result") or {}
    status = str(result.get("status", ""))
    if not run.get("shadow"):
        if run.get("approved"):  # unreachable while ADR-016 locks approvals; never shown here anyway
            return {**base, "text": "Especialista aprovado: saída não exibida nesta ferramenta."}
        code = str(result.get("error_code") or status)
        return {**base, "text": f"Especialista indisponível ({code})."}
    if status == "INCONCLUSIVE":
        code = str(result.get("error_code") or "")
        return {**base, "text": f"Inconclusiva: {SHADOW_INCONCLUSIVE_PT.get(code, code)}."}
    preds = result.get("predictions") or []
    if status != "OK" or not preds:
        return {**base, "text": f"Sem medição ({status or 'sem resultado'})."}
    attrs: Mapping[str, Any] = preds[0].get("attributes") or {}
    tol = float(attrs.get("tolerance_deg", 5.0))
    low = float(attrs.get("borderline_low_deg", 3.0))
    angle = float(attrs.get("angle_deg", float("nan")))
    abs_angle = abs(angle)
    inclined = str(preds[0].get("class_id")) != "OK"
    neutral = (f"Sugestão da prévia: etiqueta inclinada (acima de {fmt(tol, 0)}°)" if inclined
               else f"Sugestão da prévia: etiqueta alinhada (até {fmt(tol, 0)}°)")
    bar = limit_bar("Ângulo da etiqueta", abs_angle, f"{fmt(angle)}°", scale=(0, tol * 2), thresholds=[tol],
                    band=(low, tol), attention=abs_angle >= low, scale_text=("0°", f"{fmt(tol * 2, 0)}°"),
                    detail=f"faixa limite {fmt(low, 0)}–{fmt(tol, 0)}°")
    return {**base, "bar": bar, "text": neutral, "angle_text": f"{fmt(angle)}°"}


# -- cascade trail ---------------------------------------------------------------------------
def trail_view(quality: Mapping[str, Any], shadow: Mapping[str, Any], verdict: Mapping[str, Any]
               ) -> list[dict[str, str]]:
    q_valid = quality.get("valid")
    if q_valid is None:
        q_step = {"kind": "skip", "name": "Qualidade da imagem", "detail": "aguardando quadro", "value": "—"}
    elif q_valid:
        q_step = {"kind": "ok", "name": "Qualidade da imagem", "detail": "imagem apta", "value": "VALID"}
    else:
        reasons = quality.get("reasons") or []
        q_step = {"kind": "warn", "name": "Qualidade da imagem",
                  "detail": "; ".join(str(r) for r in reasons[:2]), "value": "INPUT_INVALID"}
    return [
        q_step,
        {"kind": "skip", "name": "Anomalia", "detail": "não implementada", "value": "—"},
        {"kind": "skip", "name": "Roteador", "detail": "não implementada", "value": "—"},
        {"kind": "skip", "name": "Especialista — etiqueta (sombra)", "detail": str(shadow.get("text", "")),
         "value": str(shadow.get("angle_text", "—"))},
        {"kind": "skip", "name": "Regra final", "detail": str(verdict.get("reason", "")),
         "value": str(verdict.get("value", NO_DECISION))},
    ]


# -- status strip + banner -------------------------------------------------------------------
_STATE_DOT = {"OBSERVING": "ok", "DEGRADED": "review", "FAULT": "fault"}


def banner_view(system_state: str, camera_status: str, problem: str | None, problem_code: str | None,
                frames_processed: int = 0) -> dict[str, str] | None:
    if system_state == "FAULT":
        camera = camera_status == "FAULT" or (problem_code == "AEX-CAP-001")
        return {"title": "Câmera desconectada — inspeção parada" if camera else "Falha travada — inspeção parada",
                "text": RESET_HELP if camera else f"{problem or 'Falha travada.'} {RESET_HELP}",
                "code": "AEX-CAP-001" if camera else (problem_code or "AEX-STATE-002")}
    if problem:
        title = "Inspeção interrompida" if frames_processed > 0 else "Inspeção não iniciada"
        return {"title": title, "text": problem, "code": problem_code or ""}
    return None


_VERDICT_WORDS = re.compile(r"\b(PASS|DEFECT)\b")
SCRUBBED = "[ocultado]"


def scrub(value: Any) -> Any:
    """Fail-closed last barrier (review B3): ANY string anywhere in the output that contains a
    conclusive verdict word is replaced, whatever its origin (exception text, codes, config)."""
    if isinstance(value, str):
        return SCRUBBED if _VERDICT_WORDS.search(value) else value
    if isinstance(value, Mapping):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def build_status(*, system_state: str, camera_status: str, latency_p95_ms: float | None,
                 fps: float | None, ram_free_gb: float | None, quality_config: QualityConfig,
                 result: str | None, reason: str | None, detail: Mapping[str, Any] | None,
                 frame_size: tuple[int, int] | None, problem: str | None, problem_code: str | None,
                 frames_processed: int, running: bool, image_stale: bool = False) -> dict[str, Any]:
    q = quality_config
    quality_raw: Mapping[str, Any] | None = (detail or {}).get("quality")
    quality = quality_view(quality_raw, q)
    shadow = shadow_view((detail or {}).get("stages"), quality["valid"])
    verdict = verdict_view(result, reason)
    roi = {"x": q.roi_x * 100, "y": q.roi_y * 100, "w": q.roi_w * 100, "h": q.roi_h * 100}
    if quality_raw and frame_size and frame_size[0] > 0 and frame_size[1] > 0:
        rx, ry, rw, rh = (float(v) for v in quality_raw.get("roi_px") or (0, 0, 0, 0))
        if rw > 0 and rh > 0:
            w, h = frame_size
            roi = {"x": rx / w * 100, "y": ry / h * 100, "w": rw / w * 100, "h": rh / h * 100}
    out: dict[str, Any] = scrub({
        "tool": "bench-view", "tool_notice": TOOL_NOTICE, "privacy_notice": PRIVACY_NOTICE,
        "mode": "OBSERVATION_ONLY", "running": running, "frames_processed": frames_processed,
        "image_note": ("Imagem congelada: último quadro antes da interrupção." if image_stale else ""),
        "strip": {
            "camera": camera_status, "camera_dot": {"OK": "ok", "FAULT": "fault"}.get(camera_status, "idle"),
            "system": system_state, "system_dot": _STATE_DOT.get(system_state, "idle"),
            "latency_text": f"{fmt(latency_p95_ms, 0)} ms" if latency_p95_ms is not None else "—",
            "fps_text": fmt(fps) if fps is not None else "—",
            "ram_text": f"{fmt(ram_free_gb, 2)} GB" if ram_free_gb is not None else "—",
            "envelope_text": f"{q.version} · {'provisório' if q.status != 'validated' else 'validado'}",
        },
        "banner": banner_view(system_state, camera_status, problem, problem_code, frames_processed),
        "verdict": verdict, "quality": quality, "shadow": shadow,
        "trail": trail_view(quality, shadow, verdict),
        "roi_pct": {k: round(v, 2) for k, v in roi.items()},
    })
    return out
