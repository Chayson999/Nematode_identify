import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import resolve_safe_path


class ResolveSafePathTests(unittest.TestCase):
    def test_allows_files_inside_configured_directory(self):
        base_dir = Path("/tmp/test-root")
        result = resolve_safe_path(base_dir, "demo.mp4")
        self.assertEqual(result, base_dir.resolve() / "demo.mp4")

    def test_rejects_path_traversal(self):
        base_dir = Path("/tmp/test-root")
        with self.assertRaises(ValueError):
            resolve_safe_path(base_dir, "../outside.mp4")

    def test_rejects_absolute_path(self):
        base_dir = Path("/tmp/test-root")
        with self.assertRaises(ValueError):
            resolve_safe_path(base_dir, "/tmp/outside.mp4")


if __name__ == "__main__":
    unittest.main()
