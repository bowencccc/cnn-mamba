import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from cnn_mamba.window_storage import WindowWriter
from cnn_mamba import build_propensity_masks


def _script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _windows(root):
    for split in ('train', 'val', 'test'):
        with WindowWriter(root / f'{split}.h5') as writer:
            for strand in ('plus', 'minus'):
                labels = np.array([0, 1, 0, 2, 0, 0], dtype=np.uint8)
                writer.append(f'1_{strand}_100.npz', {
                    'sequence': np.zeros(6, dtype=np.uint8),
                    'labels': np.zeros(6, dtype=np.uint8),
                    'start_stop_labels': labels,
                })


def _weights(root, name):
    with np.load(root / 'train' / name) as data:
        return data['start_stop_weights']


def test_droso_sidecars_keep_coordinate_and_tier_rules(tmp_path, monkeypatch):
    data, out = tmp_path / 'data', tmp_path / 'out'
    _windows(data)
    module = _script('prepare_droso_lightgbm_start_stop_weights')
    monkeypatch.setattr(module, 'CHROMS', {'1': 'chr1'})
    def read_scores(path):
        if path.name.startswith('start'):
            return {('chr1', '+', 101): 0.5, ('chr1', '-', 102): 0.8}
        return {('chr1', '+', 103): 0.1, ('chr1', '-', 100): 0.8}
    monkeypatch.setattr(module, 'read_scores', read_scores)
    monkeypatch.setattr(sys, 'argv', ['script', '--data-root', str(data),
                                    '--score-dir', str(tmp_path), '--output-root', str(out)])
    module.main()
    np.testing.assert_allclose(_weights(out, '1_plus_100.npz'), [1, .3, 1, 0, 1, 1])
    np.testing.assert_allclose(_weights(out, '1_minus_100.npz'), np.ones(6))
    assert not list(out.rglob('*.h5'))


def test_human_sidecars_filter_passed_and_unmapped(tmp_path, monkeypatch):
    data, out, scores = tmp_path / 'data', tmp_path / 'out', tmp_path / 'scores'
    _windows(data)
    scores.mkdir()
    summary = {'tasks': {task: {'cross_fitted_precision_operating_points': [
        {'target_precision': .9, 'folds': [{'fold': 0, 'threshold_fitted_without_held_fold': .5}]}
    ]} for task in ('start', 'stop')}}
    (scores / 'summary_corrected.json').write_text(json.dumps(summary))
    module = _script('prepare_human_lightgbm_start_stop_filter')
    monkeypatch.setattr(module, 'read_passed_sites', lambda path, thresholds: (
        {('chr1', '+', 101)}, {('chr1', '+', 101), ('chr1', '+', 103)}))
    monkeypatch.setattr(sys, 'argv', ['script', '--data-root', str(data),
                                    '--lightgbm-dir', str(scores), '--output-root', str(out)])
    module.main()
    np.testing.assert_array_equal(_weights(out, '1_plus_100.npz'), [1, 1, 1, 0, 1, 1])
    np.testing.assert_array_equal(_weights(out, '1_minus_100.npz'), [1, 0, 1, 0, 1, 1])
    assert not list(out.rglob('*.h5'))


def test_propensity_sidecars_preserve_selection(tmp_path, monkeypatch):
    data, out = tmp_path / 'data', tmp_path / 'out'
    _windows(data)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {})
    class Model:
        def eval(self):
            return self
        def __call__(self, sequence):
            logits = torch.zeros((*sequence.shape, 3))
            logits[..., 1] = 10
            return logits, logits
    monkeypatch.setattr(build_propensity_masks, 'model_from_checkpoint', lambda *args: Model())
    def masks(sequence):
        return torch.ones_like(sequence, dtype=torch.bool), torch.zeros_like(sequence, dtype=torch.bool)
    monkeypatch.setattr(build_propensity_masks, 'splice_candidate_masks', masks)
    monkeypatch.setattr(build_propensity_masks, 'start_stop_candidate_masks', masks)
    monkeypatch.setattr(sys, 'argv', ['script', '--checkpoint', str(tmp_path / 'model.pt'),
                                    '--data-root', str(data), '--output-root', str(out),
                                    '--splits', 'train,val', '--batch-size', '3'])
    build_propensity_masks.main()
    with np.load(out / 'train' / '1_plus_100.npz') as row:
        np.testing.assert_array_equal(row['splice_propensity'], np.ones(6, dtype=bool))
        np.testing.assert_array_equal(row['start_stop_propensity'], [1, 0, 1, 0, 1, 1])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['files'] == 4
    assert manifest['totals_with_overlapping_windows']['start']['selected_unlabeled'] == 16
    assert not list(out.rglob('*.h5'))
