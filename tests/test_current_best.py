import json
from pathlib import Path
import tempfile
import unittest

from cnn_mamba.current_best import SCHEMA, load_registry, materialize


def registry(root: Path) -> dict[str, object]:
    entries = {}
    for species, evidence in (("human", "rodentia"), ("dmel", "coleoptera")):
        entries[species] = {
            "evidence_source": evidence,
            "selection": {"checkpoint_epoch": 1, "validation_mean_ap": 0.5},
            "prediction_assets": ["prediction"],
            "assets": {
                "annotation": {
                    "source": f"inputs/{species}.gff.gz", "required": True,
                    "link": "annotation.gff.gz", "sha256": True,
                },
                "checkpoint": {
                    "source": f"inputs/{species}.pt", "required": True,
                    "link": "best_model.pt",
                },
                "prediction": {
                    "source": f"outputs/{species}.npy", "required": False,
                    "link": "prediction.npy",
                },
            },
        }
    return {"schema": SCHEMA, "project_root": str(root), "species": entries}


class CurrentBestTests(unittest.TestCase):
    def test_materialize_current_best_links_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            for species in ("human", "dmel"):
                (inputs / f"{species}.gff.gz").write_bytes(species.encode())
                (inputs / f"{species}.pt").write_bytes(b"checkpoint")
            (outputs / "dmel.npy").write_bytes(b"prediction")
            destination = root / "current"
            result = materialize(registry(root), root, destination)

            self.assertTrue((destination / "human/best_model.pt").is_symlink())
            self.assertEqual(
                (destination / "human/best_model.pt").read_bytes(), b"checkpoint"
            )
            self.assertTrue(result["species"]["human"]["model_ready"])
            self.assertFalse(result["species"]["human"]["prediction_ready"])
            self.assertTrue(result["species"]["dmel"]["prediction_ready"])
            self.assertEqual(
                len(result["species"]["human"]["assets"]["annotation"]["sha256"]),
                64,
            )
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["schema"], SCHEMA)

    def test_missing_optional_removes_stale_managed_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            for species in ("human", "dmel"):
                (inputs / f"{species}.gff.gz").write_bytes(b"annotation")
                (inputs / f"{species}.pt").write_bytes(b"checkpoint")
            stale = root / "current/human/prediction.npy"
            stale.parent.mkdir(parents=True)
            stale.symlink_to(inputs / "human.pt")
            materialize(registry(root), root, root / "current")
            self.assertFalse(stale.is_symlink())

    def test_missing_required_is_reported_after_manifest_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                FileNotFoundError, "human: annotation, checkpoint"
            ):
                materialize(registry(root), root, root / "current", {"human"})
            manifest = json.loads((root / "current/manifest.json").read_text())
            self.assertFalse(manifest["species"]["human"]["model_ready"])

    def test_registry_rejects_wrong_evidence_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = registry(root)
            config["species"]["human"]["evidence_source"] = "primates"
            path = root / "registry.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "rodentia"):
                load_registry(path)


if __name__ == "__main__":
    unittest.main()
