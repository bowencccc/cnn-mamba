"""Read NPZ windows or pack them into a single HDF5 file per split."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

import numpy as np


_SCHEMA = "cnn_mamba_windows_v1"


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 window storage requires h5py") from exc
    return h5py


def _dataset_name(key):
    # HDF5 treats slashes as paths, while NPZ keys can contain them.
    if key in (".", ".."):
        return "%2E" * len(key)
    return quote(key, safe="") or "%00"


class _H5Row:
    def __init__(self, arrays, row, keys):
        self._arrays = arrays
        self._row = row
        self._keys = keys

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        dataset = self._arrays[key]
        value = dataset[self._row]
        if "numpy_dtype" in dataset.attrs:
            value = np.asarray(value, dtype=np.dtype(dataset.attrs["numpy_dtype"]))
        return value


class WindowStore:
    """Ordered virtual NPZ paths with process-local, reusable HDF5 handles."""

    def __init__(self, directories):
        if isinstance(directories, (str, os.PathLike)):
            directories = [directories]
        entries = []
        for item in directories:
            path = Path(item)
            if path.is_dir():
                entries.extend((str(p), None, None) for p in path.iterdir()
                               if p.name.endswith(".npz"))
                continue
            if path.is_file() and path.suffix != ".h5":
                raise ValueError(f"unsupported window source: {path}")
            h5_path = path if path.suffix == ".h5" else path.with_name(path.name + ".h5")
            if not h5_path.is_file():
                raise FileNotFoundError(f"window directory or HDF5 file not found: {path}")
            with _h5py().File(h5_path, "r") as handle:
                if handle.attrs.get("schema") != _SCHEMA:
                    raise ValueError(f"unsupported window schema in {h5_path}")
                try:
                    names = handle["sample_names"].asstr()[:].tolist()
                    arrays = handle["arrays"]
                    keys = json.loads(arrays.attrs["keys"])
                    if (not names or names != sorted(names) or len(set(names)) != len(names)
                            or any(Path(name).name != name or not name.endswith(".npz") for name in names)
                            or any(_dataset_name(key) not in arrays or arrays[_dataset_name(key)].shape[0] != len(names) for key in keys)):
                        raise ValueError("invalid names or array lengths")
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    raise ValueError(f"invalid window HDF5 file {h5_path}: {exc}") from exc
            virtual_dir = h5_path.parent / h5_path.stem
            keys = tuple(keys)
            entries.extend((str(virtual_dir / name), str(h5_path), (i, keys))
                           for i, name in enumerate(names))
        entries.sort(key=lambda entry: entry[0])
        self.files = [entry[0] for entry in entries]
        self._entries = entries
        self._handles = {}
        self._datasets = {}
        self._pid = os.getpid()

    def __len__(self):
        return len(self.files)

    @contextmanager
    def load(self, index):
        path, h5_path, info = self._entries[index]
        if h5_path is None:
            with np.load(path) as data:
                yield data
            return
        if self._pid != os.getpid():
            self._handles = {}
            self._datasets = {}
            self._pid = os.getpid()
        if h5_path not in self._handles:
            self._handles[h5_path] = _h5py().File(h5_path, "r")
            self._datasets[h5_path] = {
                key: self._handles[h5_path]["arrays"][_dataset_name(key)]
                for key in info[1]
            }
        row, keys = info
        yield _H5Row(self._datasets[h5_path], row, keys)

    def close(self):
        if self._pid == os.getpid():
            for handle in self._handles.values():
                handle.close()
        self._handles = {}
        self._datasets = {}
        self._pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_datasets"] = {}
        return state

    def __del__(self):
        if hasattr(self, "_handles"):
            self.close()


def pack_windows(source_dir, output_file, compression="gzip", compression_opts=6, chunk_windows=1):
    """Stream a directory of consistently shaped NPZ files into one HDF5 file."""
    source = Path(source_dir)
    output = Path(output_file)
    if not source.is_dir():
        raise FileNotFoundError(f"window directory not found: {source}")
    files = sorted(p for p in source.iterdir() if p.name.endswith(".npz"))
    if not files:
        raise ValueError(f"no NPZ windows in {source}")
    if compression not in ("gzip", "lzf", "none", None):
        raise ValueError(f"unsupported compression: {compression}")
    if not isinstance(chunk_windows, int) or chunk_windows <= 0:
        raise ValueError("chunk_windows must be a positive integer")
    if compression == "gzip" and (not isinstance(compression_opts, int) or not 0 <= compression_opts <= 9):
        raise ValueError("gzip level must be an integer from 0 to 9")
    if output.exists():
        raise FileExistsError(output)
    h5py = _h5py()
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        with h5py.File(temp_name, "w") as handle:
            handle.attrs["schema"] = _SCHEMA
            handle.create_dataset("sample_names", data=[p.name for p in files], dtype=h5py.string_dtype("utf-8"))
            arrays = handle.create_group("arrays")
            datasets = {}
            schema = None
            for index, path in enumerate(files):
                with np.load(path) as data:
                    keys = sorted(data.files)
                    values = {key: data[key] for key in keys}
                    current = {key: (value.shape, value.dtype) for key, value in values.items()}
                    if schema is None:
                        schema = current
                        arrays.attrs["keys"] = json.dumps(keys)
                        for key, value in values.items():
                            dtype = value.dtype
                            opaque = dtype.kind in "UmM"
                            stored_dtype = h5py.opaque_dtype(dtype) if opaque else dtype
                            chunks = (min(chunk_windows, len(files)), *(max(1, size) for size in value.shape))
                            kwargs = {"compression": None if compression == "none" else compression}
                            if compression == "gzip":
                                kwargs["compression_opts"] = compression_opts
                            dataset = arrays.create_dataset(
                                _dataset_name(key), shape=(len(files), *value.shape),
                                maxshape=(len(files), *(None if size == 0 else size for size in value.shape)),
                                chunks=chunks, dtype=stored_dtype, **kwargs)
                            if opaque:
                                dataset.attrs["numpy_dtype"] = dtype.str
                            datasets[key] = dataset
                    elif current != schema:
                        raise ValueError(f"window schema mismatch in {path}: expected {schema}, got {current}")
                    for key, value in values.items():
                        dataset = datasets[key]
                        if "numpy_dtype" in dataset.attrs:
                            value = value.view(dataset.dtype)
                        dataset[index] = value
        # link is an atomic no-overwrite publication even if another writer races us.
        os.link(temp_name, output)
    finally:
        os.unlink(temp_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir")
    parser.add_argument("output_file")
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--gzip-level", type=int, default=6)
    parser.add_argument("--chunk-windows", type=int, default=1)
    args = parser.parse_args()
    pack_windows(args.source_dir, args.output_file, args.compression, args.gzip_level, args.chunk_windows)


if __name__ == "__main__":
    main()
