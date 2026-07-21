"""Plan A, Part A — custom-backend hot flag (resolve_custom_backend).

Runs against the app3 venv (torch/fastapi/etc. present); skipped elsewhere so it
does not fail on a machine without the heavy inference deps.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import app3
    _IMPORT_ERR = None
except Exception as _e:  # torch/openvino/etc. not installed in this env
    app3 = None
    _IMPORT_ERR = _e


@unittest.skipIf(app3 is None, "app3 import failed (heavy deps absent)")
class TestResolveCustomBackend(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "custom_backend.json")
        self._saved_path = app3.CUSTOM_BACKEND_FLAG_PATH
        self._saved_cache = dict(app3._custom_backend_cache)
        app3.CUSTOM_BACKEND_FLAG_PATH = self.path
        app3._custom_backend_cache["mtime"] = None
        app3._custom_backend_cache["value"] = "fp32"
        self._mtime = 100000.0

    def tearDown(self):
        app3.CUSTOM_BACKEND_FLAG_PATH = self._saved_path
        app3._custom_backend_cache.clear()
        app3._custom_backend_cache.update(self._saved_cache)
        self._tmp.cleanup()

    def _write(self, obj):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(obj if isinstance(obj, str) else json.dumps(obj))
        # Force a distinct mtime so the change is always detected.
        self._mtime += 10.0
        os.utime(self.path, (self._mtime, self._mtime))

    def test_absent_defaults_fp32(self):
        self.assertEqual(app3.resolve_custom_backend(), "fp32")

    def test_int8_flag(self):
        self._write({"backend": "int8"})
        self.assertEqual(app3.resolve_custom_backend(), "int8")

    def test_mtime_cache_then_flip(self):
        self._write({"backend": "int8"})
        self.assertEqual(app3.resolve_custom_backend(), "int8")
        self.assertEqual(app3.resolve_custom_backend(), "int8")  # cached (no mtime change)
        self._write({"backend": "fp32"})
        self.assertEqual(app3.resolve_custom_backend(), "fp32")

    def test_invalid_value_defaults_fp32(self):
        self._write({"backend": "weird"})
        self.assertEqual(app3.resolve_custom_backend(), "fp32")

    def test_delete_reverts_fp32(self):
        self._write({"backend": "int8"})
        self.assertEqual(app3.resolve_custom_backend(), "int8")
        os.remove(self.path)
        self.assertEqual(app3.resolve_custom_backend(), "fp32")

    def test_malformed_json_defaults_fp32(self):
        self._write("{ not json")
        self.assertEqual(app3.resolve_custom_backend(), "fp32")


if __name__ == "__main__":
    unittest.main()
