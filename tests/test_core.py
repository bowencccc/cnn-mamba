import tempfile
import unittest
from pathlib import Path

import torch

from cnn_mamba.common import splice_candidate_masks, start_stop_candidate_masks
from cnn_mamba.model import SpliceMamba
from cnn_mamba.score_chromosome import keep_bounds
from cnn_mamba.uniann_evaluate import parse_stats


class CoreTests(unittest.TestCase):
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

    def test_window_ownership(self):
        self.assertEqual(keep_bounds(0, 3, 10_000, 5_000), (0, 7_500))
        self.assertEqual(keep_bounds(1, 3, 10_000, 5_000), (2_500, 7_500))
        self.assertEqual(keep_bounds(2, 3, 10_000, 5_000), (2_500, 10_000))

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
