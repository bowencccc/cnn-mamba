"""Stream and read HDF5 training windows with stable sample IDs."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

import numpy as np
import h5py


_SCHEMA = "cnn_mamba_windows_v1"


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
        self.files = keys

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

    def get(self, key, default=None):
        return self[key] if key in self else default


class WindowStore:
    """HDF5 rows sorted by original sample ID, with per-process read handles."""

    def __init__(self, directories):
        if isinstance(directories, (str, os.PathLike)):
            directories = [directories]
        entries = []
        for item in directories:
            path = Path(item)
            h5_path = path if path.suffix == ".h5" else path.with_name(path.name + ".h5")
            if not h5_path.is_file():
                raise FileNotFoundError(
                    f"HDF5 windows not found: {h5_path}. Generate HDF5 windows or "
                    "use the one-time migration snippet in README.md."
                )
            with h5py.File(h5_path, "r") as handle:
                if handle.attrs.get("schema") != _SCHEMA:
                    raise ValueError(f"unsupported window schema in {h5_path}")
                try:
                    names = handle["sample_names"].asstr()[:].tolist()
                    arrays = handle["arrays"]
                    keys = json.loads(arrays.attrs["keys"])
                    if (len(set(names)) != len(names)
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
        self._by_name = None

    def __len__(self):
        return len(self.files)

    @contextmanager
    def load(self, index):
        path, h5_path, info = self._entries[index]
        if self._pid != os.getpid():
            self._handles = {}
            self._datasets = {}
            self._pid = os.getpid()
        if h5_path not in self._handles:
            self._handles[h5_path] = h5py.File(h5_path, "r")
            self._datasets[h5_path] = {
                key: self._handles[h5_path]["arrays"][_dataset_name(key)]
                for key in info[1]
            }
        row, keys = info
        yield _H5Row(self._datasets[h5_path], row, keys)

    def load_name(self, name):
        """Look up a stable sample ID, typically in a single-split sidecar."""
        if self._by_name is None:
            self._by_name = {}
            for index, path in enumerate(self.files):
                basename = Path(path).name
                if basename in self._by_name:
                    raise ValueError(f"ambiguous sample ID: {basename}")
                self._by_name[basename] = index
        try:
            index = self._by_name[Path(name).name]
        except KeyError as exc:
            raise KeyError(f"missing sample ID {name!r}") from exc
        return self.load(index)

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


class WindowWriter:
    """Stream one split to HDF5, publishing atomically only on success."""

    def __init__(self, output_file, compression="gzip", compression_opts=6, chunk_windows=1):
        self.output = Path(output_file)
        if compression not in ("gzip", "lzf", "none", None):
            raise ValueError(f"unsupported compression: {compression}")
        if not isinstance(chunk_windows, int) or chunk_windows <= 0:
            raise ValueError("chunk_windows must be a positive integer")
        if compression == "gzip" and (not isinstance(compression_opts, int) or not 0 <= compression_opts <= 9):
            raise ValueError("gzip level must be an integer from 0 to 9")
        self.compression = None if compression == "none" else compression
        self.compression_opts = compression_opts
        self.chunk_windows = chunk_windows

    def __enter__(self):
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, self._temp = tempfile.mkstemp(prefix=f".{self.output.name}.", suffix=".tmp", dir=self.output.parent)
        os.close(fd)
        self._handle = h5py.File(self._temp, "w")
        self._handle.attrs["schema"] = _SCHEMA
        self._arrays = self._handle.create_group("arrays")
        self._arrays.attrs["keys"] = "[]"
        self._datasets, self._schema = {}, None
        self._names, self._seen = [], set()
        self._capacity = 0
        return self

    def append(self, name, arrays):
        name = str(name)
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError(f"invalid sample ID: {name!r}")
        if name in self._seen:
            raise ValueError(f"duplicate sample ID: {name}")
        values = {key: np.asarray(value) for key, value in arrays.items()}
        schema = {key: (value.shape, value.dtype) for key, value in values.items()}
        if self._schema is None:
            self._schema = schema
            self._arrays.attrs["keys"] = json.dumps(sorted(values))
            for key, value in values.items():
                opaque = value.dtype.kind in "UmM"
                dtype = h5py.opaque_dtype(value.dtype) if opaque else value.dtype
                kwargs = {"compression": self.compression}
                if self.compression == "gzip":
                    kwargs["compression_opts"] = self.compression_opts
                dataset = self._arrays.create_dataset(
                    _dataset_name(key), shape=(0, *value.shape),
                    maxshape=(None, *(None if size == 0 else size for size in value.shape)),
                    chunks=(self.chunk_windows, *(max(1, size) for size in value.shape)),
                    dtype=dtype, **kwargs,
                )
                if opaque:
                    dataset.attrs["numpy_dtype"] = value.dtype.str
                self._datasets[key] = dataset
        elif schema != self._schema:
            raise ValueError(f"window schema mismatch in {name}: expected {self._schema}, got {schema}")
        index = len(self._names)
        if index == self._capacity:
            self._capacity += max(256, self.chunk_windows)
            for dataset in self._datasets.values():
                dataset.resize(self._capacity, axis=0)
        for key, value in values.items():
            dataset = self._datasets[key]
            if "numpy_dtype" in dataset.attrs:
                value = value.view(dataset.dtype)
            dataset[index] = value
        self._names.append(name)
        self._seen.add(name)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                for dataset in self._datasets.values():
                    dataset.resize(len(self._names), axis=0)
                self._handle.create_dataset("sample_names", data=self._names,
                                            dtype=h5py.string_dtype("utf-8"))
            self._handle.close()
            if exc_type is None:
                # Atomic no-overwrite publication, including concurrent writers.
                os.link(self._temp, self.output)
        finally:
            self._handle.close()
            os.unlink(self._temp)
