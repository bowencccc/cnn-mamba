import numpy as np
import pytest

from cnn_mamba import human_window_helpers as helper
from cnn_mamba import prepare_droso, prepare_human
from cnn_mamba.window_storage import WindowStore

pytest.importorskip("h5py")


class TinyFasta:
    def __init__(self, path):
        pass

    def get_reference_length(self, chrom):
        return 8

    def fetch(self, chrom, start, end):
        return "ACGTACGT"[start:end]

    def close(self):
        pass


def _sites(chrom, **positions):
    return {site: {chrom: {"+": positions.get(site, []), "-": []}}
            for site in ("donor", "acceptor", "start", "stop")}


def _read_single(output, split):
    assert sorted(p.name for p in output.iterdir() if p.suffix == ".h5") == [f"{split}.h5"]
    assert not list(output.rglob("*.npz"))
    store = WindowStore(output / split)
    assert len(store) == 1
    with store.load(0) as row:
        result = {key: row[key] for key in row.files}
    store.close()
    return result


def test_droso_generation_writes_hdf5(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_droso.pysam, "FastaFile", TinyFasta)
    monkeypatch.setattr(prepare_droso, "SPLITS", {"train": ["2L"]})
    monkeypatch.setattr(prepare_droso, "CHROMS", {"2L": "tiny"})
    monkeypatch.setattr(prepare_droso, "sha256", lambda path: "unused")
    regions = {"tiny": {"+": [(0, 8)], "-": []}}
    sites = _sites("tiny", donor=[2], start=[3])
    reference = _sites("tiny", acceptor=[5])
    result = prepare_droso.generate_cell(tmp_path, {"id": "tiny", "window_bp": 8, "stride_bp": 4}, regions, sites, reference)
    assert result["summary"]["train"]["windows"] == 1
    assert result["summary"]["train"]["eviann"]["donor"] == 1
    assert result["summary"]["train"]["eviann"]["start"] == 1
    assert result["summary"]["train"]["reference"]["acceptor"] == 1
    row = _read_single(tmp_path, "train")
    np.testing.assert_array_equal(row["sequence"], [0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(row["labels"], [0, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["start_stop_labels"], [0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["chess_labels"], [0, 0, 0, 0, 0, 2, 0, 0])


def test_human_generation_writes_hdf5(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_human.pysam, "FastaFile", TinyFasta)
    monkeypatch.setattr(prepare_human, "WINDOW_SIZE", 8)
    monkeypatch.setattr(prepare_human, "STRIDE", 4)
    regions = {"chr1": {"+": [(0, 8)], "-": []}}
    sites = _sites("chr1", donor=[2], start=[3])
    ignored = _sites("chr1", acceptor=[4])
    reference = _sites("chr1", donor=[5])
    result = prepare_human.generate_split(tmp_path, "train", ["chr1"], regions, sites,
                                          ignored, reference, "unused", 0, 8)
    assert result["windows"] == 1
    assert result["eviann_positive_positions"]["donor"] == 1
    assert result["masked_excluded_eviann_positions"]["acceptor"] == 1
    row = _read_single(tmp_path, "train")
    np.testing.assert_array_equal(row["labels"], [0, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["start_stop_labels"], [0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(row["splice_ignore"], [0, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(row["reference_labels"], [0, 0, 0, 0, 0, 1, 0, 0])
    with pytest.raises(FileExistsError):
        prepare_human.generate_split(tmp_path, "train", ["chr1"], regions, sites,
                                     ignored, reference, "unused", 0, 8)


def test_human_helper_generation_writes_hdf5(tmp_path, monkeypatch):
    monkeypatch.setattr(helper.pysam, "FastaFile", TinyFasta)
    monkeypatch.setattr(helper, "WINDOW_SIZE", 8)
    monkeypatch.setattr(helper, "STRIDE", 4)
    regions = {"chr1": {"+": [(0, 8)], "-": []}}
    splice = {"chr1": {"+": [(2, 5)], "-": []}}
    starts = {"chr1": {"+": [3], "-": []}}
    stops = {"chr1": {"+": [], "-": []}}
    result = helper.generate_split(tmp_path, "train", ["chr1"], regions, splice, starts, stops, "unused")
    assert result["windows"] == 1
    row = _read_single(tmp_path, "train")
    np.testing.assert_array_equal(row["labels"], [0, 0, 1, 0, 0, 2, 0, 0])
    np.testing.assert_array_equal(row["start_stop_labels"], [0, 0, 0, 1, 0, 0, 0, 0])
    assert row["has_any_label"]


def test_empty_split_is_hdf5_without_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(helper.pysam, "FastaFile", TinyFasta)
    result = helper.generate_split(tmp_path, "val", [], {}, {}, {}, {}, "unused")
    assert result["windows"] == 0
    assert (tmp_path / "val.h5").is_file()
    assert not (tmp_path / "val").exists()
    store = WindowStore(tmp_path / "val")
    assert len(store) == 0
    store.close()
