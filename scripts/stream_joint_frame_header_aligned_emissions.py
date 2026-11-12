#!/usr/bin/env python3
"""Coordinate-safe chunk adapter for the retained-header UniAnn contract.

Whole-chromosome collaborator runs continue to use the unchanged, headered
CNN--Mamba TSV directly. Human chr1 must be decoded in chunks for memory
reasons. For those chunks, this adapter applies the phase offset supplied by
the retained header while producing exactly one valid emission row per local
FASTA base. It does not change the exported TSV or UniAnn.
"""

from pathlib import Path
import sys

# External chunk runners load this file directly with importlib, which does not
# put the script directory on sys.path.  Make the sibling implementation
# importable in that mode as well as during an ordinary command-line run.
script_dir = str(Path(__file__).resolve().parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from stream_joint_frame_uniann_emissions import (  # noqa: F401
    main,
    validate_out_ps_file,
)
import stream_joint_frame_uniann_emissions as implementation


implementation.FRAME_INDEX_OFFSET = 0
implementation.CONVERTER_SCHEMA = "joint_frame_header_aligned_uniann_emissions_v1"


if __name__ == "__main__":
    main()
