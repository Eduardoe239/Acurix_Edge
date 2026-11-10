"""Contract-level errors and the AEX error catalog (Anexo 02A §3).

Format `AEX-<domain>-<number>`. The code is fixed; `detail` text may vary.
Codes without HTTP status are internal (incidents, logs, event `reason`).
This module is the single source of the catalog in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    API_001 = "AEX-API-001"
    API_002 = "AEX-API-002"
    API_003 = "AEX-API-003"
    API_004 = "AEX-API-004"
    API_005 = "AEX-API-005"
    AUTH_001 = "AEX-AUTH-001"
    AUTH_002 = "AEX-AUTH-002"
    AUTH_003 = "AEX-AUTH-003"
    AUTH_004 = "AEX-AUTH-004"
    AUTH_005 = "AEX-AUTH-005"
    AUTH_006 = "AEX-AUTH-006"
    SES_001 = "AEX-SES-001"
    SES_002 = "AEX-SES-002"
    SES_003 = "AEX-SES-003"
    STATE_001 = "AEX-STATE-001"
    STATE_002 = "AEX-STATE-002"
    CFG_001 = "AEX-CFG-001"
    CFG_002 = "AEX-CFG-002"
    REV_001 = "AEX-REV-001"
    EXP_001 = "AEX-EXP-001"
    CAP_001 = "AEX-CAP-001"
    CAP_002 = "AEX-CAP-002"
    CAS_001 = "AEX-CAS-001"
    CAS_002 = "AEX-CAS-002"
    CAS_003 = "AEX-CAS-003"
    CAS_004 = "AEX-CAS-004"
    DB_001 = "AEX-DB-001"
    DB_002 = "AEX-DB-002"
    DEV_001 = "AEX-DEV-001"
    SYS_001 = "AEX-SYS-001"
    # Owner decision 10/11/2026 (Ordem 2 complement): internal, no HTTP status.
    CLK_001 = "AEX-CLK-001"


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    http_status: int | None
    title: str


CATALOG: dict[ErrorCode, ErrorSpec] = {
    ErrorCode.API_001: ErrorSpec(422, "Dados inválidos"),
    ErrorCode.API_002: ErrorSpec(413, "Corpo grande demais"),
    ErrorCode.API_003: ErrorSpec(404, "Recurso não encontrado"),
    ErrorCode.API_004: ErrorSpec(409, "Conflito de idempotência"),
    ErrorCode.API_005: ErrorSpec(400, "Cabeçalho obrigatório ausente"),
    ErrorCode.AUTH_001: ErrorSpec(401, "Não autenticado"),
    ErrorCode.AUTH_002: ErrorSpec(401, "Credenciais inválidas"),
    ErrorCode.AUTH_003: ErrorSpec(429, "Conta bloqueada temporariamente"),
    ErrorCode.AUTH_004: ErrorSpec(403, "Permissão insuficiente"),
    ErrorCode.AUTH_005: ErrorSpec(403, "Reautenticação necessária"),
    ErrorCode.AUTH_006: ErrorSpec(409, "Configuração inicial já feita"),
    ErrorCode.SES_001: ErrorSpec(409, "Já existe sessão ativa"),
    ErrorCode.SES_002: ErrorSpec(503, "Estação não está apta"),
    ErrorCode.SES_003: ErrorSpec(409, "Sessão já encerrada"),
    ErrorCode.STATE_001: ErrorSpec(409, "Transição de estado não permitida"),
    ErrorCode.STATE_002: ErrorSpec(409, "Falha travada"),
    ErrorCode.CFG_001: ErrorSpec(422, "Configuração inválida"),
    ErrorCode.CFG_002: ErrorSpec(422, "Modelo não aprovado na configuração"),
    ErrorCode.REV_001: ErrorSpec(409, "Evento não aguarda revisão"),
    ErrorCode.EXP_001: ErrorSpec(422, "Período de exportação inválido"),
    ErrorCode.CAP_001: ErrorSpec(503, "Câmera indisponível"),
    ErrorCode.CAP_002: ErrorSpec(None, "Descarte de frames acima do limite"),
    ErrorCode.CAS_001: ErrorSpec(None, "Etapa da cascata excedeu o tempo"),
    ErrorCode.CAS_002: ErrorSpec(None, "Saída de etapa fora do contrato"),
    ErrorCode.CAS_003: ErrorSpec(None, "Hash de modelo divergente"),
    ErrorCode.CAS_004: ErrorSpec(None, "Memória insuficiente para a etapa"),
    ErrorCode.DB_001: ErrorSpec(503, "Banco indisponível"),
    ErrorCode.DB_002: ErrorSpec(507, "Disco cheio"),
    ErrorCode.DEV_001: ErrorSpec(None, "Periférico sem comunicação"),
    ErrorCode.SYS_001: ErrorSpec(500, "Erro interno"),
    ErrorCode.CLK_001: ErrorSpec(None, "Relógio não confiável"),
}


class ContractError(ValueError):
    """Raised when a payload does not satisfy a versioned contract."""
