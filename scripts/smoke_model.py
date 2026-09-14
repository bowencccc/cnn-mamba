#!/usr/bin/env python3
"""CPU import/shape smoke test; does not exercise CUDA Mamba kernels."""

import torch
from cnn_mamba.model import SpliceMamba


def main():
    model = SpliceMamba(d_model=16, d_state=8, n_layers=0, dropout=0.0,
                        architecture="cnn_mamba", cnn_kernel_size=7)
    splice, start_stop = model(torch.randint(0, 5, (2, 101)))
    assert splice.shape == start_stop.shape == (2, 101, 3)
    print("model shape smoke test passed")


if __name__ == "__main__":
    main()
