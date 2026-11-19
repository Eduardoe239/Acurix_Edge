"""Builds the bench viewer: configuration (with the expected camera profile, like `capture`),
real Application + CaptureService + the label-skew stage (shadow), engine and local server.
Nothing is opened before every check passed; the camera is only opened by `engine.start()`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from acurix_edge.app import Application
from acurix_edge.benchview.engine import FPS_RANGE, STREAM_WIDTH_RANGE, BenchViewEngine
from acurix_edge.benchview.server import BenchViewServer
from acurix_edge.benchview.synthetic import SyntheticSceneSource, synthetic_profile_config
from acurix_edge.capture.profile import expected_profile
from acurix_edge.capture.service import CaptureService
from acurix_edge.capture.sources import CaptureFactory, WebcamSource
from acurix_edge.config.settings import ConfigError, load_config, validate
from acurix_edge.health.service import Check

# Page shows FAULT within 2 s: frame_timeout + watchdog period (<= 0.05 s) + page poll (0.5 s).
MAX_FRAME_TIMEOUT_S = 1.4
DEMO_MARKER = "ACURIX_DEMO_SINTETICO.txt"
DEMO_MARKER_TEXT = ("Pasta do demo sintético do visualizador de bancada (bench-view --demo-sintetico).\n"
                    "Contém apenas dados de FONTE SIMULADA. Não usar como pasta de dados de uma estação real.\n")


class BenchViewError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def build_bench_view(*, config_path: Path | None, base_dir: Path, device: int = 0, port: int | None = None,
                     fps: float = 8.0, stream_width: int = 640, capture_factory: CaptureFactory | None = None,
                     host_checks: Mapping[str, Check] | None = None, synthetic: bool = False) -> BenchViewServer:
    """`synthetic=True` (`--demo-sintetico`): the source is the generated scene, NEVER a real camera;
    no camera profile is required (it is set in code); `config_path` is optional."""
    if config_path is None and not synthetic:
        raise BenchViewError(["--config é obrigatório (exceto com --demo-sintetico)"])
    if synthetic and capture_factory is not None:
        raise BenchViewError(["o demo sintético não aceita outra fábrica de captura"])
    try:
        cfg = load_config(config_path)
        if synthetic:  # the expected profile is the synthetic scene's own, whatever the file says
            cfg = dataclasses.replace(cfg, capture=dataclasses.replace(cfg.capture, **synthetic_profile_config()))
            validate(cfg)
    except ConfigError as exc:
        raise BenchViewError([f"configuração inválida: {p}" for p in exc.problems]) from exc
    problems: list[str] = []
    expected = expected_profile(cfg.capture)
    if expected is None:
        problems.append("a configuração não tem o perfil da câmera ([capture] camera_*); "
                        "meça com `python -m acurix_edge camera-profile`")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        problems.append("--device deve ser um inteiro >= 0")
    if not FPS_RANGE[0] <= fps <= FPS_RANGE[1]:
        problems.append(f"--fps deve estar entre {FPS_RANGE[0]:g} e {FPS_RANGE[1]:g}")
    if not STREAM_WIDTH_RANGE[0] <= stream_width <= STREAM_WIDTH_RANGE[1]:
        problems.append(f"--stream-width deve estar entre {STREAM_WIDTH_RANGE[0]} e {STREAM_WIDTH_RANGE[1]}")
    if cfg.capture.frame_timeout_s > MAX_FRAME_TIMEOUT_S:
        problems.append(f"capture.frame_timeout_s deve ser <= {MAX_FRAME_TIMEOUT_S:g} s no visualizador: "
                        "detecção + watchdog + consulta da página (0,5 s) precisam caber em 2 s")
    chosen_port = cfg.api.port if port is None else port
    if not (chosen_port == 0 or 1024 <= chosen_port <= 65535):  # 0 = ephemeral (tests)
        problems.append("--port deve estar entre 1024 e 65535")
    problems.extend(_data_dir_problems(cfg.database_path(base_dir), base_dir / DEMO_MARKER, synthetic))
    if problems or expected is None:
        raise BenchViewError(problems)
    if synthetic:  # this folder now belongs to the demo: real mode will refuse it
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / DEMO_MARKER).write_text(DEMO_MARKER_TEXT, encoding="utf-8")
    source: WebcamSource = (SyntheticSceneSource() if synthetic else
                            WebcamSource(device, device_label=expected.device_label, capture_factory=capture_factory))
    app = Application(cfg, base_dir, capture=CaptureService(source, cfg.capture), host_checks=host_checks)
    app.boot()
    try:
        stage = app.load_label_skew_stage()
        engine = BenchViewEngine(app, stage, fps=fps, stream_width=stream_width)
        return BenchViewServer(engine, chosen_port)
    except OSError as exc:
        app.shutdown()
        raise BenchViewError([f"porta {chosen_port} indisponível em 127.0.0.1 ({type(exc).__name__})"]) from exc
    except BaseException:  # never leave the store/camera open on a failed build (review 002, B)
        app.shutdown()
        raise


def _data_dir_problems(database: Path, marker: Path, synthetic: bool) -> list[str]:
    """Simulated and real data never share a folder (review 002, M): the demo only uses an empty
    folder or one it created itself (marker); real mode refuses a folder marked as demo."""
    if synthetic and database.exists() and not marker.exists():
        return [f"--base-dir {marker.parent} já tem dados de uma estação real: o demo sintético "
                "só usa pasta vazia ou criada por ele (omita --base-dir para usar uma pasta temporária)"]
    if not synthetic and marker.exists():
        return [f"--base-dir {marker.parent} é uma pasta do demo sintético: use outra pasta para a câmera real"]
    return []
