import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from cnn_mamba.common import initialize_like_original
from cnn_mamba.intron_phase_labels import coding_introns_from_gtf, window_intron_tracks
from cnn_mamba.model import SpliceMamba
from cnn_mamba.prepare_droso import build_phase_track
from cnn_mamba.probe_features import (
    BackboneFeatureHooks, FEATURE_NAMES, assert_backbone_unchanged,
    backbone_parameter_snapshot,
)
from cnn_mamba.probe_metrics import (
    fit_linear_probe, four_state_metrics, seven_state_metrics,
)
from cnn_mamba.probe_overcounting import e_fold_lag, segmentwise_autocorrelation


class BeliefProbeTests(unittest.TestCase):
    def test_conditional_phase_metrics_ignore_background_logit(self):
        four_probabilities = np.asarray([
            [0.90, 0.08, 0.01, 0.01],
            [0.90, 0.01, 0.08, 0.01],
            [0.90, 0.01, 0.01, 0.08],
        ])
        four = four_state_metrics(four_probabilities, np.asarray([1, 2, 3]))
        self.assertEqual(four["joint_4class_cds_accuracy"], 0.0)
        self.assertEqual(four["conditional_frame_accuracy_3class"], 1.0)

        seven_probabilities = np.asarray([
            [0.90, 0, 0, 0, 0.08, 0.01, 0.01],
            [0.90, 0, 0, 0, 0.01, 0.08, 0.01],
            [0.90, 0, 0, 0, 0.01, 0.01, 0.08],
        ])
        seven = seven_state_metrics(seven_probabilities, np.asarray([4, 5, 6]))
        self.assertEqual(seven["joint_7class_intron_accuracy"], 0.0)
        self.assertEqual(
            seven["conditional_intron_phase_accuracy_3class"], 1.0,
        )

    def test_segmentwise_autocorrelation_does_not_cross_boundaries(self):
        # The two segments have different means. Per-segment centering removes
        # that difference, and lag two has no within-segment pairs.
        logits = np.asarray([
            [2.0, 0, 0, 0], [2.0, 0, 0, 0],
            [9.0, 0, 0, 0], [9.0, 0, 0, 0],
        ])
        correlation, pairs = segmentwise_autocorrelation(
            logits, [(0, 2), (2, 4)], 0, "logits", 3,
        )
        np.testing.assert_array_equal(pairs, [4, 2, 0, 0])
        np.testing.assert_allclose(correlation, 0.0)
        self.assertEqual(e_fold_lag(np.ones(4), pairs=pairs), (1, True))

    @unittest.skipUnless(torch.cuda.is_available(), "Mamba3 feature hooks require CUDA")
    def test_feature_hooks_shapes_and_frozen_probe_epoch(self):
        model = SpliceMamba(
            d_model=192, d_state=64, n_layers=8, dropout=0.0,
            architecture="cnn_mamba", cnn_kernel_size=7,
        )
        initialize_like_original(model)
        model.cuda().eval().requires_grad_(False)
        before = backbone_parameter_snapshot(model)
        sequence = torch.randint(0, 5, (2, 31), device="cuda")
        with BackboneFeatureHooks(model) as hooks:
            outputs = hooks.collect(sequence, include_concat=True)
        for name in FEATURE_NAMES:
            self.assertEqual(outputs[name].shape, (2, 31, 192))
        self.assertEqual(outputs["concat_all"].shape, (2, 31, 1920))
        x = outputs["final_norm"].reshape(-1, 192).cpu().numpy().astype(np.float16)
        y = np.arange(len(x)) % 4
        fit_linear_probe(
            x, y, 192, 4, 1e-3, epochs=1, batch_size=16,
            device="cuda",
        )
        assert_backbone_unchanged(model, before)

    def test_intron_phase_plus_minus_and_conflict(self):
        text = """\
chrT\tx\tCDS\t101\t150\t.\t+\t0\tgene_id \"g1\"; transcript_id \"plus_a\";
chrT\tx\tCDS\t201\t250\t.\t+\t1\tgene_id \"g1\"; transcript_id \"plus_a\";
chrT\tx\tCDS\t101\t150\t.\t+\t0\tgene_id \"g2\"; transcript_id \"plus_b\";
chrT\tx\tCDS\t201\t250\t.\t+\t2\tgene_id \"g2\"; transcript_id \"plus_b\";
chrT\tx\tCDS\t401\t450\t.\t-\t0\tgene_id \"g3\"; transcript_id \"minus_a\";
chrT\tx\tCDS\t301\t350\t.\t-\t2\tgene_id \"g3\"; transcript_id \"minus_a\";
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gtf"
            path.write_text(text)
            introns = coding_introns_from_gtf(path)
        plus = introns[("chrT", "+")]
        self.assertEqual([(x.start, x.end, x.phase) for x in plus], [
            (150, 200, 1), (150, 200, 2),
        ])
        tracks = window_intron_tracks(plus, 100, 200, "+")
        self.assertTrue(np.all(tracks["intron_phase_labels"][50:100] == -2))
        minus = introns[("chrT", "-")]
        self.assertEqual([(x.start, x.end, x.phase) for x in minus], [(350, 400, 1)])
        tracks = window_intron_tracks(minus, 300, 200, "-")
        valid = np.flatnonzero(tracks["intron_phase_mask"])
        self.assertEqual((valid[0], valid[-1]), (100, 149))
        self.assertEqual(tracks["distance_to_donor"][100], 0)
        self.assertEqual(tracks["distance_to_acceptor"][149], 0)
        self.assertEqual(tracks["intron_phase_labels"][100], 1)
        self.assertEqual(tracks["intron_start"][100], 350)
        self.assertEqual(tracks["intron_end"][100], 400)
        downstream = build_phase_track([(300, 350, 2)], 500, "-")
        # Genomic 349 is the first CDS base after the acceptor in transcript
        # orientation and becomes position 150 after reversing this window.
        self.assertEqual(downstream[349], minus[0].phase)


if __name__ == "__main__":
    unittest.main()
