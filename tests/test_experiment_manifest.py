import importlib.util
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("manifest", pathlib.Path(__file__).resolve().parents[1] / "scripts/experiment_manifest.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = pathlib.Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        def g(*args):
            subprocess.run(["git", "-C", str(self.repo), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        g("init")
        g("config", "user.name", "Test")
        g("config", "user.email", "test@example.invalid")
        (self.repo / "source.py").write_text("# source")
        g("add", ".")
        g("commit", "-m", "fixture")
        self.data = self.base / "data"
        self.data.write_bytes(b"input")
        self.items = ["dataset=" + str(self.data)]
        self.out = self.base / "manifest.json"

    def test_roundtrip_and_no_paths(self):
        payload = m.record(self.repo, "exp001-attempt1", self.out, self.items)
        self.assertNotIn(str(self.base), self.out.read_text())
        self.assertEqual(payload, m.verify(self.repo, self.out, self.items))
    def test_dirty_rejected(self):
        (self.repo / "new").write_text("untracked")
        with self.assertRaises(ValueError):
            m.record(self.repo, "exp1", self.out, self.items)
    def test_overwrite_rejected(self):
        m.record(self.repo, "exp1", self.out, self.items)
        with self.assertRaises(FileExistsError):
            m.record(self.repo, "exp1", self.out, self.items)
    def test_in_repo_rejected(self):
        with self.assertRaises(ValueError):
            m.record(self.repo, "exp1", self.repo / "manifest.json", self.items)
    def test_changed_input_rejected(self):
        m.record(self.repo, "exp1", self.out, self.items)
        self.data.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            m.verify(self.repo, self.out, self.items)
    def test_duplicate_label_rejected(self):
        with self.assertRaises(ValueError):
            m.artifacts(self.items + self.items)
    def test_missing_label_rejected(self):
        m.record(self.repo, "exp1", self.out, self.items)
        with self.assertRaises(ValueError):
            m.verify(self.repo, self.out, ["other=" + str(self.data)])

    def test_mutation_during_hash_rejected(self):
        real = m.os.fstat
        calls = []
        def altered(fd):
            calls.append(fd)
            if len(calls) == 2:
                self.data.write_bytes(b"changed")
            return real(fd)
        with patch.object(m.os, "fstat", side_effect=altered):
            with self.assertRaises(ValueError):
                m.artifacts(self.items)
    def test_source_change_during_verify_rejected(self):
        m.record(self.repo, "exp1", self.out, self.items)
        real = m.artifacts
        def changed(items):
            result = real(items)
            (self.repo / "new").write_text("changed during verification")
            return result
        with patch.object(m, "artifacts", side_effect=changed):
            with self.assertRaises(ValueError):
                m.verify(self.repo, self.out, self.items)

if __name__ == "__main__":
    unittest.main()
