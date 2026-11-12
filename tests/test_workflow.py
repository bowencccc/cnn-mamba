import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

from cnn_mamba.build_uniann_emissions import main as build_emissions
from cnn_mamba.workflow import expand, load_workflow


class WorkflowTests(unittest.TestCase):
    def test_expansion_uses_declared_variables(self):
        self.assertEqual(
            expand("${ROOT}/data", {"ROOT": "/tmp/project"}),
            "/tmp/project/data",
        )

    def test_workflow_resolves_paths_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "workflow.json"
            config.write_text(
                """{
  "schema_version": 1,
  "project_root": "project",
  "variables": {"FASTA": "${PROJECT_ROOT}/data/genome.fa"},
  "steps": [
    {"name": "score", "command": "score", "args": ["--fasta", "${FASTA}"]}
  ]
}
"""
            )
            parsed = load_workflow(config)
        self.assertEqual(parsed["project_root"], str((root / "project").resolve()))
        self.assertEqual(parsed["steps"][0]["module"], "cnn_mamba.score_chromosome")
        self.assertEqual(
            parsed["steps"][0]["args"][-1],
            str((root / "project" / "data" / "genome.fa").resolve()),
        )

    def test_packaged_source_has_no_user_home_paths(self):
        source = Path(__file__).resolve().parents[1] / "src" / "cnn_mamba"
        offenders = []
        for path in source.glob("*.py"):
            text = path.read_text()
            if "/home/" in text or "bowencccc" in text or "bchen103" in text:
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_emission_builder_has_no_implicit_second_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probability_dir = root / "probabilities"
            probability_dir.mkdir()
            probabilities = np.asarray(
                [[0.4, 0.3, 0.2, 0.1]] * 6, dtype=np.float32
            )
            probability_path = probability_dir / "joint_frame_probabilities.npy"
            np.save(probability_path, probabilities)
            (probability_dir / "metadata.json").write_text(
                '{"calibration_method":"scalar_temperature","temperature":9.0}\n'
            )
            fasta = root / "sequence.fa"
            fasta.write_text(">chr\nACGTAC\n")
            output = root / "out.ps.txt"
            previous = sys.argv
            try:
                sys.argv = [
                    "build-uniann-emissions",
                    "--probabilities", str(probability_path),
                    "--fasta", str(fasta),
                    "--output", str(output),
                    "--seqid", "chr",
                    "--expected-calibration-method", "scalar_temperature",
                    "--expected-temperature", "9",
                ]
                build_emissions()
            finally:
                sys.argv = previous
            rows = [line.split("\t") for line in output.read_text().splitlines()]
            manifest = Path(str(output) + ".manifest.json").read_text()
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(len(row) == 6 for row in rows))
        self.assertIn('"binary_odds_temperature": null', manifest)
        self.assertIn('"frame_noncoding_gate": false', manifest)


if __name__ == "__main__":
    unittest.main()
