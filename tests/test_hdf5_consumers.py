import importlib.util
from pathlib import Path
import shutil

import numpy as np
import torch

from cnn_mamba.train_cds_head import CDSDataset
from cnn_mamba.window_storage import WindowStore
from window_test_helpers import pack_windows


def _script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(tmp_path, split):
    source = tmp_path / split
    source.mkdir()
    for start in (20, 0):
        labels = np.zeros(12, dtype=np.uint8)
        labels[4] = 1
        np.savez(source / f'3L_plus_{start}.npz', sequence=np.arange(12),
                 labels=labels, start_stop_labels=labels,
                 phase_labels=np.zeros(12, dtype=np.uint8),
                 phase_mask=np.ones(12, dtype=bool), cds_labels=labels,
                 reference_phase_labels=np.zeros(12, dtype=np.uint8),
                 reference_phase_mask=np.ones(12, dtype=bool),
                 reference_cds_labels=np.zeros(12, dtype=bool))
    pack_windows(source, tmp_path / f'{split}.h5')
    shutil.rmtree(source)


def test_hdf5_cds_and_calibration_replay(tmp_path):
    _split(tmp_path, 'val')
    dataset = CDSDataset(tmp_path / 'val')
    assert [Path(path).name for path in dataset.files] == ['3L_plus_0.npz', '3L_plus_20.npz']
    sequence, cds, reference = dataset[0]
    assert torch.equal(sequence, torch.arange(12))
    assert cds[4] == 1 and not reference.any()
    store = WindowStore(tmp_path / 'val')
    replay = _script('fit_droso_vector_scaling').replay_sample
    blocks, files, dropped = replay(store, np.zeros(8, dtype=np.uint8), 42, 4, 2, 0, 12, 12, 20)
    assert blocks.tolist() == [0] * 4 + [1] * 4
    assert files.tolist() == [0] * 4 + [1] * 4
    assert dropped == 0
    store.close()
    dataset._windows.close()


def test_hdf5_summary_and_structured_cache(tmp_path):
    for split in ('train', 'val', 'test'):
        _split(tmp_path, split)
    summary = _script('summarize_window_gene_coverage')
    windows = summary.load_windows(tmp_path, {'3L': 'chr3L'})
    assert windows[('train', 'chr3L', '+')] == [0, 20]
    structured = _script('train_structured_hmm_calibration')
    examples = structured.choose_examples(tmp_path / 'train', 2, 6, 42)
    assert {path.name for path, _ in examples} == {'3L_plus_0.npz', '3L_plus_20.npz'}
    assert [start for _, start in examples] == [1, 1]

    class Model:
        def __call__(self, sequence, return_joint_frame):
            assert return_joint_frame
            output = sequence.unsqueeze(-1).float()
            return output, output, output

    result = structured.build_cache(Model(), examples, 6, 2, torch.device('cpu'), False)
    assert result['sequence'].shape == (2, 6)
    assert result['labels'][:, 3].tolist() == [1, 1]
    assert torch.equal(result['sequence'], result['frame_logits'].squeeze(-1).long())
