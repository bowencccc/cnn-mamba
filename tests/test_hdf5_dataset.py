import copy
from contextlib import contextmanager
import multiprocessing as mp

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cnn_mamba.model import SpliceMamba
from cnn_mamba.train import PUDataset, candidate_loss
from window_test_helpers import pack_windows

pytest.importorskip("h5py")


class _LegacyFixtureRows:
    """Test-only pre-migration input reader, never a production fallback."""

    def __init__(self, directories):
        self.files = sorted(str(p) for d in directories for p in d.glob("*.npz"))

    @contextmanager
    def load(self, index):
        with np.load(self.files[index]) as row:
            yield row


def _make_windows(tmp_path, embedded_weights=False):
    npz_dirs, h5_files = [], []
    weights, propensity = tmp_path / "weights", tmp_path / "propensity"
    for split in ("z_split", "a_split"):
        directory = tmp_path / split
        directory.mkdir()
        (weights / split).mkdir(parents=True)
        (propensity / split).mkdir(parents=True)
        for i, name in enumerate(("b.npz", "a.npz")):
            length = 12
            seq = np.array([0, 3, 2, 3] * 3, dtype=np.uint8)
            seq[0] = i
            labels = np.zeros(length, dtype=np.int8)
            labels[1 + i] = 1
            labels[4 + i] = 2
            arrays = {
                "sequence": seq, "labels": labels,
                "start_stop_labels": np.roll(labels, 2),
                "chess_labels": np.roll(labels, 1),
                "chess_start_stop_labels": np.roll(labels, 3),
                "phase_labels": np.arange(length, dtype=np.int8) % 3,
                "phase_mask": np.ones(length, dtype=bool),
                "reference_phase_labels": np.arange(length, dtype=np.int8) % 3,
                "reference_phase_mask": np.ones(length, dtype=bool),
                "cds_labels": np.arange(length) % 2 == 0,
                "reference_cds_labels": np.arange(length) % 2 == 1,
                "cds_boundary_mask": np.arange(length) % 3 == 0,
                "reference_cds_boundary_mask": np.arange(length) % 3 == 1,
            }
            w = np.linspace(0.3, 1, length, dtype=np.float32)
            if embedded_weights:
                arrays["start_stop_weights"] = w
            np.savez(directory / name, **arrays)
            np.savez(weights / split / name, start_stop_weights=w)
            np.savez(propensity / split / name, splice_propensity=labels,
                     start_stop_propensity=np.roll(labels, 2))
        output = tmp_path / f"{split}.h5"
        pack_windows(directory, output)
        pack_windows(weights / split, weights / f"{split}.h5")
        pack_windows(propensity / split, propensity / f"{split}.h5")
        npz_dirs.append(directory)
        h5_files.append(output)
    options = dict(require_phase=True, require_cds=True, require_cds_boundary=True,
                   require_start_stop_weights=True, start_stop_weight_root=weights,
                   propensity_root=propensity)
    original = PUDataset(h5_files, **options)
    original._windows = _LegacyFixtureRows(npz_dirs)
    original.files = original._windows.files
    return original, PUDataset(h5_files, **options)


def _assert_batch_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and torch.equal(x, y)


@pytest.mark.parametrize("embedded_weights", [False, True])
def test_dataset_equivalence_and_defaults(tmp_path, embedded_weights):
    original, packed = _make_windows(tmp_path, embedded_weights)
    assert original.files == packed.files == sorted(original.files)
    assert len(original) == 4
    for i in range(len(original)):
        a, b = original[i], packed[i]
        _assert_batch_equal(a, b)
        assert len(a) == 20
        assert not a[3].any() and not a[4].any()  # absent ignore masks
        assert torch.all(a[13] == 1) and torch.all(a[14] == 1)  # absent joint masks
        assert torch.equal(a[5], torch.roll(a[1], 1))  # legacy reference alias
        assert torch.equal(a[6], torch.roll(a[1], 3))
    packed._windows.close()


@pytest.mark.parametrize("method", [m for m in ("fork", "spawn") if m in mp.get_all_start_methods()])
def test_persistent_loader_workers(tmp_path, method):
    original, packed = _make_windows(tmp_path)
    _ = packed[0]  # Parent holds open HDF5 handles before workers start.
    expected = list(DataLoader(original, batch_size=2, shuffle=False))
    loader = DataLoader(packed, batch_size=2, shuffle=False, num_workers=2,
                        persistent_workers=True, multiprocessing_context=method)
    try:
        for _ in range(2):
            actual = list(loader)
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                _assert_batch_equal(a, b)
        _assert_batch_equal(original[0], packed[0])
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
        packed._windows.close()


def test_training_step_equivalence(tmp_path):
    original, packed = _make_windows(tmp_path)
    torch.manual_seed(42)
    model = SpliceMamba(d_model=8, d_state=4, n_layers=0, dropout=0.0)
    models = [model, copy.deepcopy(model)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.01) for m in models]
    loaders = [DataLoader(dataset, batch_size=2, shuffle=False) for dataset in (original, packed)]
    for batches in zip(*loaders):
        losses = []
        for batch, net, optimizer in zip(batches, models, optimizers):
            optimizer.zero_grad()
            sequence, labels, start_stop = batch[:3]
            splice, codon = net(sequence)
            loss = (candidate_loss(splice, labels, sequence, "splice", ignore=batch[3])
                    + candidate_loss(codon, start_stop, sequence, "start_stop", ignore=batch[4],
                                     candidate_weights=batch[17]))
            loss.backward()
            losses.append(loss.detach())
        assert torch.equal(*losses)
        for p, q in zip(models[0].parameters(), models[1].parameters()):
            assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)
        for optimizer in optimizers:
            optimizer.step()
        for p, q in zip(models[0].parameters(), models[1].parameters()):
            assert torch.equal(p, q)
    packed._windows.close()
