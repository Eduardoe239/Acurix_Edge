"""AEX-HEALTH-005 acceptance: HLT-006, AEX-CAP-001, AEX-DB-001/002, AEX-CAS-004.

Failure injected in every component/resource; RAM below the ADR-014 minimum -> DEGRADED;
database unavailable -> FAULT; output shaped as the OpenAPI Health/Status schemas.
"""

import re
import subprocess
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from acurix_edge.app import Application
from acurix_edge.config.settings import AppConfig, StorageConfig
from acurix_edge.contracts.errors import ErrorCode
from acurix_edge.domain.state_machine import SystemState
from acurix_edge.health.checks import (
    ADR014_MIN_FREE_RAM_MB,
    CLOCK_FLOOR_UTC,
    ClockCheck,
    CpuCheck,
    GpuCheck,
    memory_check,
)
from acurix_edge.health.service import OPENAPI_COMPONENTS, ComponentStatus as C, HealthService
from tests.support import (
    FIXTURE_POLICY,
    HEALTHY_HOST,
    FixtureApprovedRule,
    FixtureOkAdapter,
    TempDirTestCase,
    request,
)

OPENAPI = Path(__file__).resolve().parents[2] / "docs" / "02-architecture" / "api" / "openapi-v1.yaml"


def fixed(status, detail="injected", code=None):
    return (lambda: (status, detail, code)) if code else (lambda: (status, detail))


class HealthyAppCase(TempDirTestCase):
    def app(self, *, host=None, camera=fixed(C.OK), rule=True, inference=True, config=None) -> Application:
        a = Application(config or AppConfig(), self.tmp, model_policy=FIXTURE_POLICY,
                        inference=FixtureOkAdapter() if inference else None,
                        rule=FixtureApprovedRule() if rule else None,
                        capture_probe=camera, host_checks={**HEALTHY_HOST, **(host or {})})
        a.boot()
        self.addCleanup(a.shutdown)
        return a

    def entry(self, a, name):
        st = a.status()
        return next(e for e in st["components"] + st["resources"] if e["component"] == name), st


class InjectedFailureTests(HealthyAppCase):
    def test_baseline_all_healthy_is_ok(self):
        a = self.app()
        st = a.status()
        self.assertEqual(st["overall"], "OK")
        code, body = a.health_document()
        self.assertEqual((code, body["state"], body["overall"]), (200, "IDLE", "OK"))

    def test_each_component_failure_is_visible(self):
        # (name, how to inject, expected entry status, expected code, expected overall)
        cases = [
            ("camera", dict(camera=None), "UNKNOWN", "AEX-CAP-001", "DEGRADED"),
            ("camera", dict(camera=fixed(C.FAULT, "unplugged", ErrorCode.CAP_001)), "FAULT", "AEX-CAP-001",
             "DEGRADED"),
            ("models", dict(inference=False), "DEGRADED", None, "DEGRADED"),
            # clock is critical since 10/11/2026: its FAULT (went back before the last event) is overall FAULT
            ("clock", dict(host={"clock": fixed(C.FAULT, "back in time", ErrorCode.CLK_001)}), "FAULT",
             "AEX-CLK-001", "FAULT"),
            ("clock", dict(host={"clock": fixed(C.DEGRADED, "jump", ErrorCode.CLK_001)}), "DEGRADED",
             "AEX-CLK-001", "DEGRADED"),
            ("memory", dict(host={"memory": memory_check(lambda: {"total_gb": 7.73, "available_gb": 0.79})}),
             "DEGRADED", "AEX-CAS-004", "DEGRADED"),
            ("cpu", dict(host={"cpu": fixed(C.UNKNOWN)}), "UNKNOWN", None, "DEGRADED"),
            ("gpu", dict(host={"gpu": fixed(C.DEGRADED)}), "DEGRADED", None, "DEGRADED"),
            ("decision_rule", dict(rule=False), "DEGRADED", None, "DEGRADED"),
            ("storage", dict(config=AppConfig(storage=StorageConfig(min_free_disk_mb=10**9))), "DEGRADED",
             "AEX-DB-002", "DEGRADED"),
        ]
        for name, inject, status, code, overall in cases:
            with self.subTest(component=name, status=status):
                a = self.app(**inject)
                e, st = self.entry(a, name)
                self.assertEqual((e["status"], e["code"], st["overall"]), (status, code, overall))
                self.assertTrue(e["detail"])
                self.assertRegex(e["checked_at_utc"], r"Z$")
                a.shutdown()

    def test_models_empty_list_is_honest_not_hidden(self):
        a = Application(AppConfig(), self.tmp, host_checks=HEALTHY_HOST)  # product default (NoOp)
        a.boot()
        self.addCleanup(a.shutdown)
        e, st = self.entry(a, "models")
        self.assertEqual(e["status"], "DEGRADED")
        self.assertIn("approved_models: 0", e["detail"])
        self.assertEqual(st["models"], [])

    def test_queue_drops_are_visible(self):
        a = self.app()
        a.queue.drop()
        e, st = self.entry(a, "queue")
        self.assertEqual((e["status"], e["code"], st["overall"]), ("DEGRADED", "AEX-CAP-002", "DEGRADED"))
        self.assertEqual(st["counters"]["frames_dropped"], 1)

    def test_raising_check_is_fault_with_code(self):
        def boom():
            raise RuntimeError("probe crashed")

        a = self.app(host={"cpu": boom})
        e, _ = self.entry(a, "cpu")
        self.assertEqual((e["status"], e["code"]), ("FAULT", "AEX-SYS-001"))

    def test_database_unavailable_is_fault_and_503(self):
        a = self.app()
        a.store.close()
        e, st = self.entry(a, "database")
        self.assertEqual((e["status"], e["code"]), ("FAULT", "AEX-DB-001"))
        self.assertEqual(st["overall"], "FAULT")
        self.assertIs(a.state.state, SystemState.FAULT)
        self.assertTrue(st["fault_latched"])
        code, body = a.health_document()
        self.assertEqual((code, body["state"], body["overall"]), (503, "FAULT", "FAULT"))

    def test_low_ram_session_is_degraded_and_never_pass(self):
        low = memory_check(lambda: {"total_gb": 7.73, "available_gb": (ADR014_MIN_FREE_RAM_MB - 1) / 1024})
        a = self.app(host={"memory": low})
        a.start_session()
        self.assertIs(a.state.state, SystemState.DEGRADED)
        self.assertEqual(a.process(request()).result, "NO_DECISION")
        ok = self.app(host={"memory": memory_check(lambda: {"total_gb": 7.73, "available_gb": 1.5})})
        self.assertEqual(self.entry(ok, "memory")[0]["status"], "OK")

    def test_critical_disabled_is_fault(self):
        h = HealthService()
        h.register("database", fixed(C.DISABLED), critical=True)
        self.assertIs(h.evaluate().overall, C.FAULT)
        h2 = HealthService()
        h2.register("gpu", fixed(C.DISABLED), critical=False)
        h2.register("database", fixed(C.OK), critical=True)
        self.assertIs(h2.evaluate().overall, C.OK)  # optional GPU absent: CPU fallback, not degraded


class ProbeUnitTests(unittest.TestCase):
    def test_clock(self):
        # Clock problems other than "earlier than the last event" are DEGRADED + AEX-CLK-001.
        before_floor = ClockCheck(now=lambda: CLOCK_FLOOR_UTC - timedelta(days=1))
        self.assertEqual(before_floor()[0::2], (C.DEGRADED, ErrorCode.CLK_001))
        t = [datetime(2026, 11, 10, tzinfo=timezone.utc)]
        clock = ClockCheck(now=lambda: t[0])
        self.assertEqual(clock(), (C.OK, "UTC clock plausible"))
        t[0] -= timedelta(seconds=30)
        self.assertEqual(clock()[0::2], (C.DEGRADED, ErrorCode.CLK_001))  # wall clock jumped back
        t[0] += timedelta(seconds=1)
        self.assertIs(clock()[0], C.DEGRADED)  # still behind a time already observed
        t[0] += timedelta(seconds=30)
        self.assertIs(clock()[0], C.OK)  # caught up

    def test_clock_unreadable_is_degraded_not_raised(self):
        def broken():
            raise OSError("rtc")

        self.assertEqual(ClockCheck(now=broken)()[0::2], (C.DEGRADED, ErrorCode.CLK_001))
        naive = ClockCheck(now=lambda: datetime(2026, 11, 10))
        self.assertEqual(naive()[0::2], (C.DEGRADED, ErrorCode.CLK_001))
        bad_last = ClockCheck(now=lambda: datetime(2026, 11, 10, tzinfo=timezone.utc), last_event_utc=lambda: "x")
        self.assertEqual(bad_last()[0::2], (C.DEGRADED, ErrorCode.CLK_001))

    def test_clock_before_last_event_is_fault(self):
        now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
        last = "2026-11-10T12:00:00.001Z"  # strict: 1 ms in the future of the clock is already FAULT
        self.assertEqual(ClockCheck(now=lambda: now, last_event_utc=lambda: last)()[0::2],
                         (C.FAULT, ErrorCode.CLK_001))
        same = "2026-11-10T12:00:00.000Z"
        self.assertIs(ClockCheck(now=lambda: now, last_event_utc=lambda: same)()[0], C.OK)

    def test_memory_unknown_when_unmeasurable(self):
        self.assertIs(memory_check(lambda: None)()[0], C.UNKNOWN)

    def test_cpu_unknown_when_unmeasurable(self):
        cpu = CpuCheck()
        cpu._times = lambda: None
        self.assertIs(cpu()[0], C.UNKNOWN)

    def test_gpu_absent_failing_ok_and_cached(self):
        self.assertIs(GpuCheck(which=lambda _: None)()[0], C.DISABLED)

        def timeout(*a, **k):
            raise subprocess.TimeoutExpired("nvidia-smi", 3)

        self.assertIs(GpuCheck(which=lambda _: "nvidia-smi", run=timeout)()[0], C.DEGRADED)
        calls = []

        def ok(*a, **k):
            calls.append(1)
            return subprocess.CompletedProcess(a, 0, stdout="RTX 3050 Laptop GPU, 6144, 300, 45\n")

        now = [0.0]
        g = GpuCheck(which=lambda _: "nvidia-smi", run=ok, clock=lambda: now[0])
        self.assertEqual(g()[0], C.OK)
        self.assertIn("6144", g()[1])
        self.assertEqual(len(calls), 1)  # cached
        now[0] += 31
        g()
        self.assertEqual(len(calls), 2)

    def test_real_probes_return_valid_results(self):
        # Smoke test on this host; only checks shape (values vary by session).
        for probe in (memory_check(), CpuCheck(), GpuCheck(), ClockCheck()):
            with self.subTest(probe=type(probe).__name__):
                result = probe()
                self.assertIsInstance(result[0], C)
                self.assertIsInstance(result[1], str)


@unittest.skipUnless(OPENAPI.is_file(), "OpenAPI document not present (docs/ is not versioned)")
class OpenApiShapeTests(HealthyAppCase):
    @staticmethod
    def schema_block(name: str) -> str:
        text = OPENAPI.read_text(encoding="utf-8")
        m = re.search(rf"^    {name}:\n((?:      .*\n)+)", text, re.MULTILINE)
        assert m, name
        return m.group(1)

    @staticmethod
    def enum(block: str, prop: str) -> set[str]:
        m = re.search(rf"{prop}: \{{type: string, enum: \[([^\]]+)\]", block)
        assert m, prop
        return {v.strip() for v in m.group(1).split(",")}

    @staticmethod
    def required(block: str) -> set[str]:
        m = re.search(r"required: \[([^\]]+)\]", block)
        return {v.strip() for v in m.group(1).split(",")}

    def test_component_enums_match_openapi(self):
        block = self.schema_block("ComponentHealth")
        self.assertEqual(self.enum(block, "component"), OPENAPI_COMPONENTS)
        self.assertEqual(self.enum(block, "status"), {s.value for s in C})

    def test_status_document_matches_schema(self):
        st = self.app().status()
        self.assertLessEqual(self.required(self.schema_block("Status")), set(st))
        comp_block = self.schema_block("ComponentHealth")
        for e in st["components"]:
            self.assertLessEqual(self.required(comp_block), set(e))
            self.assertIn(e["component"], OPENAPI_COMPONENTS)
        counters = self.schema_block("Counters")
        for key in re.findall(r"^        (\w+):", counters, re.MULTILINE):
            self.assertIsInstance(st["counters"][key], int)
        self.assertIsInstance(st["fault_latched"], bool)
        self.assertIn(st["state"], self.enum_line("SystemState"))

    def test_health_document_matches_schema(self):
        code, body = self.app().health_document()
        self.assertEqual(code, 200)
        self.assertLessEqual(self.required(self.schema_block("Health")), set(body))
        self.assertIn(body["overall"], self.enum(self.schema_block("Health"), "overall"))

    def enum_line(self, name: str) -> set[str]:
        text = OPENAPI.read_text(encoding="utf-8")
        m = re.search(rf"^    {name}: \{{type: string, enum: \[([^\]]+)\]", text, re.MULTILINE)
        return {v.strip() for v in m.group(1).split(",")}


if __name__ == "__main__":
    unittest.main()
