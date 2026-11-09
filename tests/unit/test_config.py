import unittest
from pathlib import Path

from acurix_edge.config.settings import AppConfig, ConfigError, load_config
from tests.support import TempDirTestCase

EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "acurix.example.toml"


class ConfigTests(TempDirTestCase):
    def write(self, text: str) -> Path:
        p = self.tmp / "c.toml"
        p.write_text(text, encoding="utf-8")
        return p

    def test_defaults_valid_and_loopback(self):
        cfg = load_config(None)
        self.assertEqual(cfg.api.bind_host, "127.0.0.1")
        self.assertEqual(cfg.inference.adapter, "noop")

    def test_example_file_loads_and_matches_defaults(self):
        self.assertEqual(load_config(EXAMPLE).sha256, AppConfig().sha256)

    def test_hash_stable_and_sensitive(self):
        a = load_config(self.write('config_version = "0.1.0"\n'))
        b = load_config(self.write('config_version = "0.1.0"\n'))
        c = load_config(self.write('config_version = "0.1.1"\n'))
        self.assertEqual(a.sha256, b.sha256)
        self.assertNotEqual(a.sha256, c.sha256)
        self.assertEqual(len(a.sha256), 64)
        self.assertEqual(a.reference(), {"version": "0.1.0", "sha256": a.sha256})

    def test_rejects_invalid_values_and_reports_all(self):
        bad = self.write(
            'config_version = "one"\nsite_id = "../x"\n'
            '[api]\nbind_host = "0.0.0.0"\nport = 80\n'
            '[storage]\ndata_dir = "../outside"\n'
            '[inference]\nadapter = "anything"\n'
            '[logging]\nlevel = "LOUD"\n'
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(bad)
        joined = " ".join(ctx.exception.problems)
        for frag in ("config_version", "site_id", "bind_host", "api.port", "data_dir", "inference.adapter", "logging.level"):
            self.assertIn(frag, joined)

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ConfigError):
            load_config(self.write('surprise = 1\n'))
        with self.assertRaises(ConfigError):
            load_config(self.write('[api]\nexpose_publicly = true\n'))

    def test_rejects_absolute_and_drive_paths(self):
        for value in ("C:/data", "/abs", "a/../../b"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    load_config(self.write(f'[storage]\ndata_dir = "{value}"\n'))
        with self.assertRaises(ConfigError):
            load_config(self.write('[storage]\ndatabase_file = "sub/x.db"\n'))

    def test_loopback_literals_only(self):
        self.assertEqual(load_config(self.write('[api]\nbind_host = "::1"\n')).api.bind_host, "::1")
        for host in ("localhost", "0.0.0.0", "192.168.0.10"):
            with self.subTest(host=host):
                with self.assertRaises(ConfigError):
                    load_config(self.write(f'[api]\nbind_host = "{host}"\n'))

    def test_rejects_unc_drive_relative_and_reserved_names(self):
        for value in ("\\\\\\\\server\\\\share", "C:rel", "NUL", "data/con.txt", "LPT1"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    load_config(self.write(f'[storage]\ndata_dir = "{value}"\n'))
        with self.assertRaises(ConfigError):
            load_config(self.write('[storage]\ndatabase_file = "nul.db"\n'))

    def test_non_string_values_are_config_errors_not_crashes(self):
        for text in ('[api]\nbind_host = ["127.0.0.1"]\n', '[logging]\nlevel = {a = 1}\n',
                     '[inference]\nadapter = ["noop"]\n', 'site_id = 5\n', '[storage]\ndata_dir = 1\n'):
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    load_config(self.write(text))

    def test_directory_as_config_path(self):
        with self.assertRaises(ConfigError):
            load_config(self.tmp)

    def test_model_policy_locked_default_deny(self):
        cfg = load_config(None)
        self.assertEqual(cfg.models.approved_models, ())
        self.assertIs(cfg.models.real_model_loading, False)
        bad = (
            '[models]\napproved_models = [{id = "m", version = "1", sha256 = "%s"}]\n' % ("0" * 64),
            '[models]\nreal_model_loading = true\n',
            '[models]\nreal_model_loading = "false"\n',
            '[models]\napproved_models = "none"\n',
            '[models]\nallow_any = true\n',
            '[inference]\nadapter = "onnx"\n',
        )
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    load_config(self.write(text))
        ok = load_config(self.write('[models]\napproved_models = []\nreal_model_loading = false\n'))
        self.assertEqual(len(ok.models.approved_models), 0)

    def test_wrong_types(self):
        with self.assertRaises(ConfigError):
            load_config(self.write('[api]\nport = "8765"\n'))
        with self.assertRaises(ConfigError):
            load_config(self.write('api = 5\n'))

    def test_missing_and_malformed_file(self):
        with self.assertRaises(ConfigError):
            load_config(self.tmp / "missing.toml")
        with self.assertRaises(ConfigError):
            load_config(self.write("this is = = not toml"))


if __name__ == "__main__":
    unittest.main()
