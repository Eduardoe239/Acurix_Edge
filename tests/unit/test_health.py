import unittest
from pathlib import Path
from typing import Any, cast

from acurix_edge.health.checks import storage_check
from acurix_edge.health.service import ComponentStatus as C
from acurix_edge.health.service import HealthService, host_memory


def _fixed(status):
    return lambda: (status, "x")


def svc(*specs):
    h = HealthService()
    for i, (status, critical) in enumerate(specs):
        h.register(f"c{i}", _fixed(status), critical=critical)
    return h


class HealthTests(unittest.TestCase):
    def test_empty_is_unknown_not_ok(self):
        self.assertIs(HealthService().evaluate().overall, C.UNKNOWN)

    def test_aggregation(self):
        self.assertIs(svc((C.OK, True), (C.OK, False)).evaluate().overall, C.OK)
        self.assertIs(svc((C.OK, True), (C.DEGRADED, False)).evaluate().overall, C.DEGRADED)
        self.assertIs(svc((C.OK, True), (C.UNKNOWN, False)).evaluate().overall, C.DEGRADED)
        self.assertIs(svc((C.DEGRADED, True)).evaluate().overall, C.DEGRADED)
        self.assertIs(svc((C.UNKNOWN, True), (C.OK, False)).evaluate().overall, C.FAULT)
        self.assertIs(svc((C.FAULT, True)).evaluate().overall, C.FAULT)
        self.assertIs(svc((C.FAULT, False)).evaluate().overall, C.DEGRADED)

    def test_raising_check_is_fault(self):
        h = HealthService()
        h.register("db", cast(Any, lambda: 1 / 0), critical=True)  # deliberately invalid check
        r = h.evaluate()
        self.assertIs(r.overall, C.FAULT)
        self.assertIn("ZeroDivisionError", r.component("db").detail)

    def test_invalid_status_type_is_fault(self):
        h = HealthService()
        h.register("x", cast(Any, lambda: ("OK", "string not enum")), critical=False)  # deliberately invalid
        self.assertIs(h.evaluate().component("x").status, C.FAULT)

    def test_duplicate_registration_rejected(self):
        h = HealthService()
        h.register("a", lambda: (C.OK, ""), critical=True)
        with self.assertRaises(ValueError):
            h.register("a", lambda: (C.OK, ""), critical=True)

    def test_storage_threshold(self):
        here = Path(__file__).parent
        self.assertIs(storage_check(here, 0)()[0], C.OK)
        self.assertIs(storage_check(here, 10**12)()[0], C.DEGRADED)

    def test_host_memory_snapshot_shape(self):
        mem = host_memory()
        if mem is not None:
            self.assertGreater(mem["total_gb"], 0)
            self.assertGreaterEqual(mem["available_gb"], 0)


if __name__ == "__main__":
    unittest.main()
