import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cnn_mamba.common import splice_candidate_masks, start_stop_candidate_masks
from cnn_mamba.evaluate_gated_phase import BinnedBinaryMetrics
from cnn_mamba.prepare_human import parse_cds_phase_records as parse_human_phase
from cnn_mamba.model import SpliceMamba
from cnn_mamba.prepare_droso import (
    build_cds_boundary_mask, build_cds_track, build_phase_track,
)
from cnn_mamba.score_chromosome import keep_bounds
from cnn_mamba.train import (
    PUDataset, boundary_weighted_cds_loss, candidate_loss,
    cosine_learning_rate, masked_phase_loss, group_balanced_joint_frame_loss,
    shared_backbone_gradient_norm,
)
from cnn_mamba.temperature_scaling import (
    calibration_metrics, fit_temperature, joint_frame_targets,
)
from cnn_mamba.uniann_evaluate import parse_stats


class CoreTests(unittest.TestCase):
    def test_cosine_learning_rate_stays_at_floor_after_horizon(self):
        base = 3e-4
        self.assertAlmostEqual(cosine_learning_rate(0, base, 1000, 100), 3e-6)
        self.assertAlmostEqual(cosine_learning_rate(1000, base, 1000, 100), 3e-5)
        self.assertAlmostEqual(cosine_learning_rate(2000, base, 1000, 100), 3e-5)

    def test_human_phase_parser_normalizes_accession_and_chr_names(self):
        eviann = (
            "NC_000001.11\tEviAnn\tCDS\t2\t5\t.\t+\t1\tParent=x\n"
        )
        chess = (
            'chr1\tCHESS\tCDS\t8\t10\t.\t-\t2\ttranscript_id "y";\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            e_path = Path(directory) / "e.gff"
            c_path = Path(directory) / "c.gtf"
            e_path.write_text(eviann)
            c_path.write_text(chess)
            e_records = parse_human_phase(e_path, "eviann")
            c_records = parse_human_phase(c_path, "chess")
        self.assertEqual(e_records["chr1"]["+"], [(1, 5, 1)])
        self.assertEqual(c_records["chr1"]["-"], [(7, 10, 2)])

    def test_candidate_masks(self):
        # A C G T G T A G A T G T A A
        sequence = torch.tensor([[0, 1, 2, 3, 2, 3, 0, 2, 0, 3, 2, 3, 0, 0]])
        donor, acceptor = splice_candidate_masks(sequence)
        start, stop = start_stop_candidate_masks(sequence)
        self.assertEqual(torch.where(donor[0])[0].tolist(), [2, 4, 10])
        self.assertEqual(torch.where(acceptor[0])[0].tolist(), [6])
        self.assertEqual(torch.where(start[0])[0].tolist(), [8])
        self.assertEqual(torch.where(stop[0])[0].tolist(), [5, 11])

    def test_model_shapes_without_mamba_layers(self):
        model = SpliceMamba(d_model=16, d_state=8, n_layers=0, dropout=0.0,
                            architecture="cnn_mamba", cnn_kernel_size=7)
        outputs = model(torch.randint(0, 5, (2, 101)))
        self.assertEqual(outputs[0].shape, (2, 101, 3))
        self.assertEqual(outputs[1].shape, (2, 101, 3))

    def test_frame_branch_and_phase_head_shapes(self):
        model = SpliceMamba(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
            frame_dilation=3, phase_auxiliary=True,
            cds_auxiliary=True,
        )
        outputs = model(
            torch.randint(0, 5, (2, 101)),
            return_phase=True, return_cds=True,
        )
        self.assertEqual(len(outputs), 4)
        for output in outputs[:3]:
            self.assertEqual(output.shape, (2, 101, 3))
        self.assertEqual(outputs[3].shape, (2, 101, 2))

    def test_four_state_joint_frame_head_shape(self):
        model = SpliceMamba(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
            frame_dilation=3, joint_frame_auxiliary=True,
        )
        outputs = model(
            torch.randint(0, 5, (2, 101)), return_joint_frame=True
        )
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[2].shape, (2, 101, 4))

    def test_group_balanced_joint_frame_loss(self):
        logits = torch.zeros(1, 6, 4, requires_grad=True)
        phase = torch.tensor([[0, 1, 2, 0, 0, 0]])
        phase_mask = torch.tensor([[True, True, True, False, False, False]])
        cds = torch.tensor([[True, True, True, False, False, True]])
        loss = group_balanced_joint_frame_loss(
            logits, phase, phase_mask, cds, noncoding_weight=0.1
        )
        self.assertAlmostEqual(loss.item(), 1.1 * np.log(4), places=6)
        loss.backward()
        # The final CDS base has conflicting/unknown phase and is excluded.
        self.assertTrue(torch.equal(logits.grad[0, 5], torch.zeros(4)))
        self.assertIsNotNone(logits.grad)

    def test_joint_frame_targets_exclude_ambiguous_cds(self):
        phase = torch.tensor([[0, 1, 2, 0, 0]])
        phase_mask = torch.tensor([[True, True, True, False, False]])
        cds = torch.tensor([[True, True, True, False, True]])
        targets, valid = joint_frame_targets(phase, phase_mask, cds)
        self.assertEqual(targets.tolist(), [[1, 2, 3, 0, 0]])
        self.assertEqual(valid.tolist(), [[True, True, True, True, False]])

    def test_temperature_scaling_softens_without_changing_argmax(self):
        torch.manual_seed(7)
        base = torch.randn(4000, 4)
        targets = torch.distributions.Categorical(logits=base).sample()
        overconfident = base * 3.0
        temperature = fit_temperature(overconfident, targets)
        before = calibration_metrics(overconfident, targets)
        after = calibration_metrics(overconfident, targets, temperature)
        self.assertGreater(temperature, 1.5)
        self.assertLess(after["nll"], before["nll"])
        self.assertLess(
            after["mean_max_probability"], before["mean_max_probability"]
        )
        self.assertTrue(torch.equal(
            overconfident.argmax(-1), (overconfident / temperature).argmax(-1)
        ))

    def test_phase_tracks_follow_transcript_direction(self):
        plus = build_phase_track([(2, 8, 0)], 10, "+")
        minus = build_phase_track([(2, 8, 0)], 10, "-")
        self.assertEqual(plus[2:8].tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(minus[2:8][::-1].tolist(), [0, 1, 2, 0, 1, 2])

    def test_conflicting_phase_tracks_are_masked(self):
        track = build_phase_track([(0, 6, 0), (0, 6, 1)], 6, "+")
        self.assertTrue((track == -2).all())

    def test_conflicting_phase_tracks_remain_cds(self):
        track = build_cds_track([(0, 6, 0), (0, 6, 1)], 8)
        self.assertEqual(track.tolist(), [True] * 6 + [False] * 2)

    def test_cds_boundary_mask_uses_only_real_chromosome_transitions(self):
        track = np.asarray([True, True, True, False, False, False], dtype=bool)
        mask = build_cds_boundary_mask(track, radius=1)
        self.assertEqual(mask.tolist(), [False, False, True, True, False, False])

    def test_boundary_weighted_cds_loss_separately_averages_regions(self):
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.tensor([[0, 1, 1, 0]])
        boundary = torch.tensor([[False, True, True, False]])
        loss = boundary_weighted_cds_loss(
            logits, labels, boundary, far_weight=0.1
        )
        self.assertAlmostEqual(loss.item(), 1.1 * np.log(2), places=6)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_window_ownership(self):
        self.assertEqual(keep_bounds(0, 3, 10_000, 5_000), (0, 7_500))
        self.assertEqual(keep_bounds(1, 3, 10_000, 5_000), (2_500, 7_500))
        self.assertEqual(keep_bounds(2, 3, 10_000, 5_000), (2_500, 10_000))

    def test_binned_binary_metrics_separates_perfect_scores(self):
        metric = BinnedBinaryMetrics(bins=16)
        metric.add(
            np.asarray([0.01, 0.1, 0.9, 0.99], dtype=np.float32),
            np.asarray([False, False, True, True]),
        )
        result = metric.metrics()
        self.assertAlmostEqual(result["AP"], 1.0)
        self.assertAlmostEqual(result["best_F1"], 1.0)

    def test_joint_phase_cds_dataset_layout_and_backward(self):
        length = 12
        with tempfile.TemporaryDirectory() as directory:
            arrays = {
                "sequence": np.asarray([0, 3, 2, 3] * 3, dtype=np.uint8),
                "labels": np.zeros(length, dtype=np.int8),
                "start_stop_labels": np.zeros(length, dtype=np.int8),
                "chess_labels": np.zeros(length, dtype=np.int8),
                "chess_start_stop_labels": np.zeros(length, dtype=np.int8),
                "phase_labels": np.arange(length, dtype=np.int8) % 3,
                "phase_mask": np.ones(length, dtype=bool),
                "reference_phase_labels": np.arange(length, dtype=np.int8) % 3,
                "reference_phase_mask": np.ones(length, dtype=bool),
                "cds_labels": np.asarray([0, 1] * 6, dtype=bool),
                "reference_cds_labels": np.asarray([1, 0] * 6, dtype=bool),
                "cds_boundary_mask": np.asarray(
                    [False, True, True, False] * 3, dtype=bool
                ),
                "reference_cds_boundary_mask": np.asarray(
                    [True, True, False, False] * 3, dtype=bool
                ),
            }
            np.savez_compressed(Path(directory) / "sample.npz", **arrays)
            batch = PUDataset(
                Path(directory), require_phase=True, require_cds=True,
                require_cds_boundary=True,
            )[0]
        self.assertEqual(len(batch), 15)
        self.assertTrue(torch.equal(batch[11], torch.from_numpy(arrays["cds_labels"])))
        self.assertTrue(torch.equal(
            batch[13], torch.from_numpy(arrays["cds_boundary_mask"])
        ))
        model = SpliceMamba(
            d_model=16, d_state=8, n_layers=0, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
            frame_dilation=3, phase_auxiliary=True, cds_auxiliary=True,
        )
        sequence = batch[0].unsqueeze(0)
        splice, start_stop, phase, cds = model(
            sequence, return_phase=True, return_cds=True
        )
        loss = (
            0.25 * candidate_loss(splice, batch[1].unsqueeze(0), sequence, "splice")
            + candidate_loss(
                start_stop, batch[2].unsqueeze(0), sequence, "start_stop"
            )
            + 0.1 * masked_phase_loss(
                phase, batch[7].unsqueeze(0), batch[8].unsqueeze(0)
            )
            + 0.1 * torch.nn.functional.cross_entropy(
                cds.reshape(-1, 2), batch[11].reshape(-1)
            )
        )
        shared = tuple(
            parameter for name, parameter in model.named_parameters()
            if not name.startswith((
                "splice_head.", "start_stop_head.", "phase_head.",
                "cds_head.", "frame_branch.",
            ))
        )
        cds_norm = shared_backbone_gradient_norm(
            torch.nn.functional.cross_entropy(
                cds.reshape(-1, 2), batch[11].reshape(-1)
            ),
            shared,
        )
        self.assertGreater(cds_norm, 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in shared))
        loss.backward()
        self.assertIsNotNone(model.cds_head.weight.grad)

    def test_stats_parser(self):
        text = (
            "Query mRNAs : 100 in 90 loci (80 multi-exon)\n"
            "Locus level:    75.0     |    80.0\n"
            "Matching loci: 72\nMissed loci: 10/100\nNovel loci: 3/90\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.stats"
            path.write_text(text)
            row = parse_stats(path)
        self.assertEqual(row["f1"], 77.41935483870968)
        self.assertEqual(row["predicted_loci"], 90)


if __name__ == "__main__":
    unittest.main()
