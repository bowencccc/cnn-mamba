import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

from cnn_mamba.build_uniann_emissions import main as build_emissions
from cnn_mamba.export_joint_frame_tsv import main as export_joint_frame
from cnn_mamba.uniann_compat import (
    JOINT_FRAME_TSV_HEADER,
    validate_header_compatible_joint_frame_tsv,
)
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

    def test_exported_joint_frame_tsv_preserves_uniann_header_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probabilities = np.asarray(
                [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
                dtype=np.float32,
            )
            probability_path = root / "joint_frame_probabilities.npy"
            argmax_path = root / "joint_frame_argmax.npy"
            output = root / "chrX_plus_joint_frame_probabilities.tsv"
            np.save(probability_path, probabilities)
            np.save(argmax_path, probabilities.argmax(axis=1).astype(np.uint8))
            previous = sys.argv
            try:
                sys.argv = [
                    "export-joint-frame",
                    "--probabilities", str(probability_path),
                    "--argmax", str(argmax_path),
                    "--output", str(output),
                    "--chrom", "chrX",
                    "--strand", "+",
                ]
                export_joint_frame()
            finally:
                sys.argv = previous
            lines = output.read_text().splitlines()
            contract = validate_header_compatible_joint_frame_tsv(output)
        self.assertEqual(lines[0], JOINT_FRAME_TSV_HEADER)
        self.assertTrue(lines[1].startswith("chrX\t1\t+\t"))
        self.assertEqual(contract["header_rows"], 1)
        self.assertFalse(contract["remove_header_before_uniann"])
        self.assertEqual(contract["rows"], 2)

    def test_joint_frame_header_validator_rejects_stripped_tsv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stripped.tsv"
            path.write_text("chrX\t1\t+\t0.3\t0.2\t0.1\t0.4\tframe0\n")
            with self.assertRaisesRegex(ValueError, "requires the original TSV header"):
                validate_header_compatible_joint_frame_tsv(path)


if __name__ == "__main__":
    unittest.main()
