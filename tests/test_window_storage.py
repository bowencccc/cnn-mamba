import multiprocessing as mp
import os
import pickle

import numpy as np
import pytest

from cnn_mamba.window_storage import WindowStore, WindowWriter
from window_test_helpers import pack_windows

h5py = pytest.importorskip("h5py")


def _read_in_child(store, queue):
    with store.load(0) as row:
        queue.put((os.getpid(), int(row["count"]), store._pid))
    store.close()


def _write_windows(path, names=("b.npz", "a.npz")):
    path.mkdir()
    for index, name in enumerate(names):
        np.savez(path / name, sequence=np.arange(6, dtype=np.uint8).reshape(2, 3) + index,
                 count=np.array(index, dtype=np.int16), label=np.array("猫", dtype="U2"),
                 empty=np.empty((0, 2), dtype=np.float32))


@pytest.mark.parametrize("compression", ["gzip", "lzf", "none"])
def test_roundtrip_raw_equality_and_lazy_rows(tmp_path, compression):
    directory = tmp_path / "train"
    _write_windows(directory)
    output = tmp_path / "train.h5"
    pack_windows(directory, output, compression=compression, chunk_windows=2)
    packed = WindowStore(output)
    originals = sorted(directory.glob("*.npz"))
    assert [str(path) for path in originals] == packed.files
    for i, path in enumerate(originals):
        with np.load(path) as expected, packed.load(i) as actual:
            assert "missing" not in actual
            assert all(key in actual for key in expected.files)
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
                assert np.asarray(actual[key]).dtype == expected[key].dtype
            with pytest.raises(KeyError):
                actual["missing"]
    handle = packed._handles[str(output)]
    with packed.load(0) as row:
        assert row._arrays["count"].file.id == handle.id
    packed.close()
    assert not handle.id.valid


def test_global_sort_logical_split_paths_and_empty_split(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_windows(a)
    _write_windows(b)
    pack_windows(a, tmp_path / "a.h5")
    pack_windows(b, tmp_path / "b.h5")
    for f in b.iterdir():
        f.unlink()
    b.rmdir()
    store = WindowStore([b, a])
    assert store.files == sorted(str(d / n) for d in (a, b) for n in ("a.npz", "b.npz"))
    with store.load(2) as row:
        assert int(row["count"]) == 1
    store.close()
    empty = tmp_path / "empty"
    with WindowWriter(tmp_path / "empty.h5"):
        pass
    assert len(WindowStore(empty)) == 0
    with pytest.raises(FileNotFoundError):
        WindowStore(tmp_path / "missing")


def test_legacy_npz_requires_explicit_conversion(tmp_path):
    source = tmp_path / "train"
    _write_windows(source)
    with pytest.raises(FileNotFoundError, match="Generate HDF5 windows"):
        WindowStore(source)
    assert len(list(source.glob("*.npz"))) == 2


def test_streaming_writer_keeps_global_id_order_and_named_lookup(tmp_path):
    path = tmp_path / "train.h5"
    with WindowWriter(path) as writer:
        writer.append("z.npz", {"count": np.array(9)})
        writer.append("a.npz", {"count": np.array(1)})
    store = WindowStore(path)
    assert [os.path.basename(p) for p in store.files] == ["a.npz", "z.npz"]
    with store.load(0) as row, store.load_name("z.npz") as other:
        assert row["count"] == 1 and other["count"] == 9
        assert row.get("missing", 3) == 3
    store.close()


def test_schema_mismatch_atomic_and_existing_file(tmp_path):
    source = tmp_path / "train"
    source.mkdir()
    np.savez(source / "a.npz", x=np.ones(2, dtype=np.int8))
    np.savez(source / "b.npz", x=np.ones(3, dtype=np.int8))
    output = tmp_path / "train.h5"
    with pytest.raises(ValueError, match="b.npz"):
        pack_windows(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".train.h5.*.tmp"))
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        pack_windows(source, output)
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("method", [method for method in ("fork", "spawn") if method in mp.get_all_start_methods()])
def test_process_handles_and_pickle(tmp_path, method):
    directory = tmp_path / "train"
    _write_windows(directory)
    output = tmp_path / "train.h5"
    pack_windows(directory, output)
    store = WindowStore(output)
    with store.load(0) as row:
        expected = int(row["count"])
    original_handle = store._handles[str(output)]
    copy = pickle.loads(pickle.dumps(store))
    assert not copy._handles
    ctx = mp.get_context(method)
    queue = ctx.Queue()
    child = ctx.Process(target=_read_in_child, args=(store, queue))
    child.start()
    pid, value, handle_pid = queue.get(timeout=20)
    child.join(timeout=20)
    assert child.exitcode == 0
    assert pid == handle_pid and pid != os.getpid()
    assert value == expected
    assert original_handle.id.valid
    store.close()
