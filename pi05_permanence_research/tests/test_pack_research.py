"""Pack-list checks for the research zip."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pack_research as pack


class Classify(unittest.TestCase):
    def test_the_colab_download_notebook_is_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "3final_smaj (1).ipynb"
            path.write_text("{}", encoding="utf-8")
            decision, reason = pack.classify(path)
            self.assertEqual(decision, "keep")
            self.assertIn("notebook", reason)

    def test_summary_json_and_contact_sheet_are_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "outputs" / "permanence" / "tight_cover"
            dest.mkdir(parents=True)
            summary = dest / "summary.json"
            sheet = dest / "tight_cover.png"
            summary.write_text("{}", encoding="utf-8")
            sheet.write_bytes(b"png")
            self.assertEqual(pack.classify(summary)[0], "keep")
            self.assertEqual(pack.classify(sheet)[0], "keep")

    def test_the_layer5_transcoder_is_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "outputs" / "permanence" / "controlled_contrasts"
            dest.mkdir(parents=True)
            weight = dest / "transcoder.pt"
            weight.write_bytes(b"ckpt")
            decision, reason = pack.classify(weight)
            self.assertEqual(decision, "keep")
            self.assertIn("dictionary", reason)

    def test_action_expert_weights_are_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "step_027233.pt"
            path.write_bytes(b"ckpt")
            decision, reason = pack.classify(path)
            self.assertEqual(decision, "skip")
            self.assertIn("weight", reason)

    def test_caches_archives_and_secrets_are_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                Path(folder) / "hf_home" / "config.json",
                Path(folder) / "libero.hdf5",
                Path(folder) / "hf_home.tar",
                Path(folder) / "secrets" / "HF_TOKEN.txt",
                Path(folder) / "outputs" / "permanence" / "layer5_replay" / "layer5_frames.npz",
            ]
            rows[0].parent.mkdir(parents=True)
            rows[3].parent.mkdir(parents=True)
            rows[4].parent.mkdir(parents=True)
            for path in rows:
                path.write_bytes(b"x")
            for path in rows:
                self.assertEqual(pack.classify(path)[0], "skip", path)


class ZipBundle(unittest.TestCase):
    def test_one_zip_holds_the_paper_files_and_skips_the_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "README.md").write_text("run notes\n", encoding="utf-8")
            (root / "conclusion.py").write_text("print(1)\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_conclusion.py").write_text("import unittest\n", encoding="utf-8")
            notebook = root / "3final_smaj (1).ipynb"
            notebook.write_text("{}", encoding="utf-8")
            dest = root / "outputs" / "permanence" / "conclusion"
            dest.mkdir(parents=True)
            (dest / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            (dest / "conclusion.png").write_bytes(b"png")
            (root / "outputs" / "occlusion_contrast").mkdir(parents=True)
            (root / "outputs" / "occlusion_contrast" / "verdict.json").write_text("{}\n", encoding="utf-8")
            (root / "hf_home").mkdir()
            (root / "hf_home" / "model.bin").write_bytes(b"skipme")
            (root / "step_027233.pt").write_bytes(b"skipme")
            (root / "uploads").mkdir()
            (root / "uploads" / "old_smaj.ipynb").write_text("{}", encoding="utf-8")
            zip_path = root / "out" / pack.DEFAULT_ZIP_NAME
            self.assertEqual(pack.main(["--cwd", str(root), "--zip", str(zip_path)]), 0)
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path) as bundle:
                names = set(bundle.namelist())
            self.assertIn("MANIFEST.txt", names)
            self.assertIn("notebooks/3final_smaj (1).ipynb", names)
            self.assertIn("source/conclusion.py", names)
            self.assertIn("tests/test_conclusion.py", names)
            self.assertIn("results/permanence/conclusion/summary.json", names)
            self.assertIn("results/permanence/conclusion/conclusion.png", names)
            self.assertIn("results/occlusion_contrast/verdict.json", names)
            self.assertNotIn("source/step_027233.pt", names)
            self.assertTrue(all("hf_home" not in name for name in names))
            self.assertTrue(all("uploads" not in name for name in names))
            self.assertNotIn("notebooks/old_smaj.ipynb", names)
            manifest = zipfile.ZipFile(zip_path).read("MANIFEST.txt").decode("utf-8")
            self.assertIn("KEEP", manifest)
            self.assertIn("SKIP", manifest)


if __name__ == "__main__":
    unittest.main()
