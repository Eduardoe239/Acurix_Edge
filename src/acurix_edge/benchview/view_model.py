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
from acurix_edge.health.checks import ADR014_MIN_FREE_RAM_MB

SHADOW_TITLE = "Prévia (modo sombra) — não decide"
VERDICT_PHRASE = "O sistema ainda não decide: especialista em validação"
PRIVACY_NOTICE = "Câmera apontada só para a bancada. Nenhuma imagem é gravada."
PRIVACY_NOTICE_SIMULATED = "Fonte simulada: nenhuma câmera em uso. Nenhuma imagem é gravada."
TOOL_NAME = "Acurix Edge · Visualizador de bancada"
TOOL_CAPTION = "ferramenta de desenvolvimento"
TOOL_NOTICE = "Ferramenta de desenvolvimento (AEX-BENCH-VIEW-001) — não é o console da release 0.5.0."
OWNER = "Luís Eduardo de Sousa Barbosa"
SIMULATED_NOTICE = "FONTE SIMULADA — imagem gerada por código, não é câmera real"
NO_DECISION = "NO_DECISION"
REVIEW = "REVIEW"
HIDDEN_VERDICT = "NÃO EXIBIDO"
DISPLAYABLE_VERDICTS = frozenset({NO_DECISION, REVIEW})  # PASS/DEFECT are never displayable here
RESET_HELP = ("Reconecte o cabo USB, encerre o visualizador e execute "
              "python -m acurix_edge reset-fault --reason \"<motivo>\" (mesma pasta). "
              "Depois inicie o visualizador de novo.")

RESET_HELP_SIMULATED = ("Fonte simulada: não há cabo nem câmera. Encerre o visualizador e inicie o demo "
                        "de novo (a pasta temporária é nova a cada execução).")

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
DECISION_REASON_PT = {  # reasons that do not depend on the system state
    "INFERENCE_NOT_OK": "Especialista em validação: nenhum modelo aprovado para decidir",
    "NO_APPROVED_RULE": "Nenhuma regra de decisão aprovada",
    "RULE_ERROR": "Erro na regra de decisão",
    "EVENT_NOT_PERSISTED": "Evento não registrado no banco",
    "EVENT_RECORD_CORRUPT": "Registro de evento corrompido",
    "IDEMPOTENCY_CONFLICT": "Inspeção repetida com conteúdo diferente",
}
POSITION_SCALE_PX = 100.0  # bar scale for the displacement (values above are clamped: "100+ px")
# Order of importance of the quality-gate refusal reasons (owner, 19/11/2026: position first).
# A gate error means nothing was measured, so it leads; unknown codes go last, in gate order.
QUALITY_REASON_ORDER = (
    "QUALITY_GATE_ERROR", "OBJECT_OUTSIDE_ROI", "OBJECT_NOT_FOUND", "OBJECT_TOO_LARGE", "OBJECT_ROTATED",
    "BACKGROUND_OUT_OF_RANGE", "BACKGROUND_NOT_UNIFORM", "FOCUS_LOW", "UNDEREXPOSED", "OVEREXPOSED",
    "CRUSHED_SHADOWS", "BLOWN_HIGHLIGHTS", "REFLECTION",
)


def ordered_reason_codes(codes: Sequence[str]) -> list[str]:
    """Distinct codes, most important first; unknown codes last, in the gate's own order."""
    rank = {c: i for i, c in enumerate(QUALITY_REASON_ORDER)}
    first_seen = {c: i for i, c in reversed(list(enumerate(codes)))}
    return sorted(first_seen, key=lambda c: (rank.get(c, len(rank)), first_seen[c]))


STATE_REASONS = frozenset({"SYSTEM_NOT_OBSERVING", "HEALTH_NOT_OK"})  # explained by the REAL health entries
HealthIssue = tuple[str, str] | tuple[str, str, str]  # (entry name, status[, AEX code])
# Shop-floor phrase chosen by (entry, status, AEX code) — never by the name alone (review 003, M1):
# a phrase states a cause only when that exact status/code says so. Anything else is described
# neutrally from the real data ("<entrada>: <status>" / "<grandeza> não medida").
HEALTH_ENTRY_PT = {
    "memory": "RAM livre", "models": "modelos", "decision_rule": "regra de decisão", "camera": "câmera",
    "database": "banco de dados", "storage": "disco", "clock": "relógio do computador", "cpu": "processador",
    "gpu": "placa de vídeo", "queue": "fila de captura", "quality_envelope": "porta de qualidade",
}
HEALTH_CAUSE_PT: dict[tuple[str, str, str], str] = {
    ("memory", "DEGRADED", "AEX-CAS-004"):
        f"RAM livre abaixo de {ADR014_MIN_FREE_RAM_MB / 1024:.1f} GB".replace(".", ","),
    ("models", "DEGRADED", ""): "nenhum modelo aprovado",
    ("models", "DEGRADED", "AEX-CFG-002"): "modelo carregado fora da lista aprovada",
    ("decision_rule", "DEGRADED", ""): "nenhuma regra de decisão aprovada",
    ("queue", "DEGRADED", "AEX-CAP-002"): "fila de captura descartando quadros",
    ("quality_envelope", "DEGRADED", "AEX-CFG-001"): "envelope de qualidade recusado",
    ("quality_envelope", "DEGRADED", "AEX-SYS-001"): "porta de qualidade com erros seguidos",
    ("storage", "DEGRADED", "AEX-DB-002"): "pouco espaço em disco",
}


def health_phrase(issue: HealthIssue) -> str:
    name, status = issue[0], issue[1]
    code = issue[2] if len(issue) > 2 else ""
    known = HEALTH_CAUSE_PT.get((name, status, code))
    if known:
        return known
    label = HEALTH_ENTRY_PT.get(name, name)
    if status == "UNKNOWN":
        return f"{label}: não medido"
    return f"{label}: {status}"


def fmt(value: float, decimals: int = 1) -> str:
    """pt-BR number: decimal comma, no thousands separator."""
    if not math.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


def fmt_limit(value: float) -> str:
    """A configured limit: no decimals when whole, one otherwise (0,5° is never shown as 0°)."""
    return fmt(value, 0 if math.isfinite(value) and value == round(value) else 1)


def _num(value: Any, default: float) -> float:
    """A numeric attribute; missing/None/non-numeric -> default (never an exception in /api/status)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _pct(value: float, lo: float, hi: float) -> float:
    if hi <= lo or not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(100.0, (value - lo) / (hi - lo) * 100.0)), 2)


def limit_bar(label: str, value: float, value_text: str, *, scale: tuple[float, float],
              thresholds: Sequence[float], band: tuple[float, float] | None, attention: bool,
              scale_text: tuple[str, str], detail: str = "") -> dict[str, Any]:
    """Data for one `ax-limit` bar: every position already in percent of the scale."""
    lo, hi = scale
    measured = math.isfinite(value)  # not measured -> no marker at all (never a fake 0)
    return {"label": label, "value_text": value_text, "value_pct": _pct(value, lo, hi) if measured else None,
            "thresholds_pct": [_pct(t, lo, hi) for t in thresholds],
            "band_pct": [_pct(band[0], lo, hi), _pct(band[1], lo, hi)] if band else None,
            "attention": attention, "scale_text": list(scale_text), "detail": detail}


# -- verdict --------------------------------------------------------------------------------
def state_reason(system_state: str, health_issues: Sequence[HealthIssue]) -> str:
    """Why the system is not deciding, from the REAL state and the REAL health entries that are
    not OK (nothing invented: see `health_phrase`)."""
    if system_state == "IDLE":
        return "Sessão de inspeção não iniciada"
    if system_state == "FAULT":
        return "Sistema em FAULT: inspeção parada"
    causes = [health_phrase(i) for i in health_issues]
    if system_state == "OBSERVING" and not causes:  # recovered since the last frame (review 003, B3)
        return "Aguardando o próximo quadro"
    return f"Sistema em {system_state}: " + "; ".join(causes) if causes else f"Sistema em {system_state}"


def reason_text(reason: str | None, system_state: str, health_issues: Sequence[HealthIssue],
                quality_reasons: Sequence[str]) -> str:
    if system_state in ("IDLE", "FAULT"):  # the current state wins over the last frame's reason
        return state_reason(system_state, health_issues)
    if reason == "INPUT_QUALITY":  # ALL reasons, most important first (owner, 19/11/2026)
        listed = "; ".join(quality_reasons)
        return "Imagem recusada pela porta de qualidade" + (f": {listed}" if listed else "")
    if reason in STATE_REASONS:
        return state_reason(system_state, health_issues)
    return DECISION_REASON_PT.get(reason or "", reason or "")


def verdict_view(result: str | None, reason: str | None, *, system_state: str = "OBSERVING",
                 health_issues: Sequence[HealthIssue] = (), quality_reasons: Sequence[str] = ()
                 ) -> dict[str, Any]:
    """The REAL system verdict, filtered: never PASS/DEFECT (shown as NÃO EXIBIDO)."""
    if result is None:
        waiting = (state_reason(system_state, health_issues) if system_state in ("IDLE", "FAULT")
                   else "Aguardando o primeiro quadro")
        return {"value": NO_DECISION, "modifier": "none", "phrase": VERDICT_PHRASE, "reason": waiting,
                "hidden": False}
    if result not in DISPLAYABLE_VERDICTS:
        return {"value": HIDDEN_VERDICT, "modifier": "none",
                "phrase": "Resultado conclusivo não exibido: esta ferramenta não está aprovada para isso.",
                "reason": "", "hidden": True}
    return {"value": result, "modifier": "review" if result == REVIEW else "none", "phrase": VERDICT_PHRASE,
            "reason": reason_text(reason, system_state, health_issues, quality_reasons), "hidden": False}


# -- quality gate ---------------------------------------------------------------------------
def quality_view(quality: Mapping[str, Any] | None, q: QualityConfig) -> dict[str, Any]:
    if not quality:
        return {"status": "—", "valid": None, "reasons": [], "bars": [], "roi": None}
    metrics: Mapping[str, float] = quality.get("metrics") or {}
    failing = {str(r.get("metric")) for r in quality.get("reasons") or []}
    codes = ordered_reason_codes([str(r.get("code")) for r in quality.get("reasons") or []])
    reasons = [QUALITY_REASON_PT.get(c, c) for c in codes]  # ALL reasons, most important first
    outside_limit = float(q.max_object_outside_px)

    def m(key: str) -> float:
        v = metrics.get(key, float("nan"))
        return float(v) if isinstance(v, (int, float)) else float("nan")

    sharp_hi = q.min_sharpness * 4
    bright_frac, dark_frac = m("bright_fraction"), m("dark_fraction")
    outside = m("object_outside_roi_px")
    found = m("object_area_fraction_roi") >= q.min_object_fraction
    # Position = displacement of the piece relative to the template, in px (limit: envelope
    # max_object_outside_px; 15 px in 0.2.0, 0 px in 0.1.0).
    if not found:
        position_value, position_text, position_detail = math.nan, "—", "peça não encontrada no gabarito"
    else:
        position_value, position_text = max(0.0, outside), f"{fmt(max(0.0, outside), 0)} px"
        position_detail = ("dentro do gabarito" if outside <= 0 else "dentro da tolerância"
                           if outside <= outside_limit else "peça deslocada para fora do gabarito")
    # Product rotation: only shown when it was really measured on a piece inside the template tolerance.
    if not found:
        rotation_value, rotation_text, rotation_detail = math.nan, "—", "não medida: peça não encontrada"
    elif outside > outside_limit:
        rotation_value, rotation_text, rotation_detail = math.nan, "—", "não avaliada: peça fora do gabarito"
    elif m("object_rotation_measured") < 1:
        rotation_value, rotation_text, rotation_detail = math.nan, "—", "não medida: contorno pouco retangular"
    else:
        rotation_value = m("object_rotation_abs_deg")
        rotation_text = f"{fmt(m('object_rotation_deg'))}°"
        rotation_detail = f"limite {fmt_limit(q.max_object_rotation_deg)}°"
    position_hi = max(POSITION_SCALE_PX, outside_limit * 2)
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
                  detail=f"limite {fmt_limit(q.max_bright_fraction * 100)} % · sombras {fmt(dark_frac * 100)} %"),
        limit_bar("Posição", position_value, position_text, scale=(0, position_hi),
                  thresholds=[outside_limit], band=(0, outside_limit),
                  attention=bool(failing & {"object_outside_roi_px", "object_area_fraction_roi"}),
                  scale_text=("0 px", f"{fmt(position_hi, 0)}+ px"),
                  detail=f"limite {fmt_limit(outside_limit)} px · {position_detail}"),
        limit_bar("Rotação do produto", rotation_value, rotation_text, scale=(0, q.max_object_rotation_deg * 2),
                  thresholds=[q.max_object_rotation_deg], band=(0, q.max_object_rotation_deg),
                  attention="object_rotation_abs_deg" in failing,
                  scale_text=("0°", f"{fmt(q.max_object_rotation_deg * 2, 0)}°"), detail=rotation_detail),
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
    tol, low, angle = _num(attrs.get("tolerance_deg"), 5.0), _num(attrs.get("borderline_low_deg"), 3.0), \
        _num(attrs.get("angle_deg"), math.nan)
    if not math.isfinite(angle):
        return {**base, "text": "Sem medição (ângulo ausente)."}
    angle = round(angle, 1)  # classify the value the operator SEES (review 003, M2): "3,0°" is never "alinhada"
    category = shadow_category(abs(angle), low, tol)
    bar = limit_bar("Ângulo da etiqueta", abs(angle), f"{fmt(angle)}°", scale=(0, tol * 2), thresholds=[tol],
                    band=(low, tol), attention=SHADOW_TONE[category] == "review",
                    scale_text=("0°", f"{fmt_limit(tol * 2)}°"),
                    detail=f"faixa limite {fmt_limit(low)}–{fmt_limit(tol)}°")
    return {**base, "bar": bar, "text": shadow_text(category, low, tol), "angle_text": f"{fmt(angle)}°",
            "category": category, "tone": SHADOW_TONE[category]}


# The text and the marker colour come from ONE category (owner review 19/11/2026): they always agree.
SHADOW_TONE = {"aligned": "neutral", "limit": "review", "inclined": "review"}


def shadow_category(abs_angle: float, low: float, tol: float) -> str:
    """Same boundaries as the specialist: > tol inclined; low..tol borderline; < low aligned."""
    if abs_angle > tol:
        return "inclined"
    return "limit" if abs_angle >= low else "aligned"


def shadow_text(category: str, low: float, tol: float) -> str:
    if category == "inclined":
        return "Etiqueta inclinada"
    if category == "limit":
        return f"Caso limite ({fmt_limit(low)}–{fmt_limit(tol)}°)"
    return "Etiqueta alinhada"


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
                  "detail": "; ".join(str(r) for r in reasons), "value": "INPUT_INVALID"}
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
                frames_processed: int = 0, simulated: bool = False) -> dict[str, str] | None:
    help_text = RESET_HELP_SIMULATED if simulated else RESET_HELP
    if system_state == "FAULT":
        camera = camera_status == "FAULT" or (problem_code == "AEX-CAP-001")
        title = ("Fonte simulada interrompida — inspeção parada" if simulated and camera
                 else "Câmera desconectada — inspeção parada" if camera else "Falha travada — inspeção parada")
        return {"title": title,
                "text": help_text if camera else f"{problem or 'Falha travada.'} {help_text}",
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
                 frames_processed: int, running: bool, image_stale: bool = False,
                 simulated: bool = False, health_issues: Sequence[HealthIssue] = ()) -> dict[str, Any]:
    q = quality_config
    quality_raw: Mapping[str, Any] | None = (detail or {}).get("quality")
    quality = quality_view(quality_raw, q)
    shadow = shadow_view((detail or {}).get("stages"), quality["valid"])
    verdict = verdict_view(result, reason, system_state=system_state, health_issues=health_issues,
                           quality_reasons=quality["reasons"])
    roi = {"x": q.roi_x * 100, "y": q.roi_y * 100, "w": q.roi_w * 100, "h": q.roi_h * 100}
    if quality_raw and frame_size and frame_size[0] > 0 and frame_size[1] > 0:
        rx, ry, rw, rh = (float(v) for v in quality_raw.get("roi_px") or (0, 0, 0, 0))
        if rw > 0 and rh > 0:
            w, h = frame_size
            roi = {"x": rx / w * 100, "y": ry / h * 100, "w": rw / w * 100, "h": rh / h * 100}
    out: dict[str, Any] = scrub({
        "tool": "bench-view", "tool_name": TOOL_NAME, "tool_caption": TOOL_CAPTION, "tool_notice": TOOL_NOTICE,
        "privacy_notice": PRIVACY_NOTICE_SIMULATED if simulated else PRIVACY_NOTICE,
        "mode": "OBSERVATION_ONLY", "running": running, "frames_processed": frames_processed,
        "source": ({"kind": "SIMULATED", "simulated": True, "notice": SIMULATED_NOTICE} if simulated
                   else {"kind": "CAMERA", "simulated": False, "notice": ""}),
        "image_note": ("Imagem congelada: último quadro antes da interrupção." if image_stale else ""),
        "strip": {
            "camera": f"{camera_status} (simulada)" if simulated else camera_status,
            "camera_dot": {"OK": "ok", "FAULT": "fault"}.get(camera_status, "idle"),
            "system": system_state, "system_dot": _STATE_DOT.get(system_state, "idle"),
            "latency_text": f"{fmt(latency_p95_ms, 0)} ms" if latency_p95_ms is not None else "—",
            "fps_text": fmt(fps) if fps is not None else "—",
            "ram_text": f"{fmt(ram_free_gb, 2)} GB" if ram_free_gb is not None else "—",
            "envelope_text": f"{q.version} · {'provisório' if q.status != 'validated' else 'validado'}",
        },
        "banner": banner_view(system_state, camera_status, problem, problem_code, frames_processed, simulated),
        "verdict": verdict, "quality": quality, "shadow": shadow,
        "trail": trail_view(quality, shadow, verdict),
        "roi_pct": {k: round(v, 2) for k, v in roi.items()},
    })
    return out
