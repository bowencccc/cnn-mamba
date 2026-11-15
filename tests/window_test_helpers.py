"""Legacy input fixtures, kept outside the HDF5-only runtime."""

from pathlib import Path

import numpy as np

from cnn_mamba.window_storage import WindowWriter


def pack_windows(source_dir, output_file, compression="gzip", compression_opts=6, chunk_windows=1):
    files = sorted(Path(source_dir).glob("*.npz"))
    with WindowWriter(output_file, compression, compression_opts, chunk_windows) as writer:
        for path in files:
            with np.load(path) as data:
                writer.append(path.name, {key: data[key] for key in data.files})
