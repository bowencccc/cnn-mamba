"""Compatibility contract for CNN--Mamba four-state TSVs and UniAnn.

The exported header is intentionally retained when the TSV is passed directly
to UniAnn commit 2f5c0ab.  Its converter counts input rows rather than reading
the position column, and that one header row supplies the required +1 offset
between CNN--Mamba codon-offset states and UniAnn genomic-frame states.
"""

from __future__ import annotations

import json
from pathlib import Path


JOINT_FRAME_TSV_COLUMNS = (
    "chrom",
    "position_1based",
    "strand",
    "frame0_probability",
    "frame1_probability",
    "frame2_probability",
    "noncoding_probability",
    "predicted_state",
)
JOINT_FRAME_TSV_HEADER = "\t".join(JOINT_FRAME_TSV_COLUMNS)
UNIANN_HEADER_COMPATIBILITY = {
    "contract": "cnn_mamba_joint_frame_tsv_header_v1",
    "header_required": True,
    "header_rows": 1,
    "remove_header_before_uniann": False,
    "model_phase_semantics": "codon nucleotide offset 0/1/2",
    "uniann_position_column_parsed": False,
    "uniann_header_placeholder_rows": 1,
    "effective_row_counter_frame_mapping": "(zero_based_position - model_phase) mod 3",
    "tested_uniann_commit": "2f5c0ab8fdb2e848f5538f3a1f741cb9c4c67e07",
}


def validate_header_compatible_joint_frame_tsv(path: Path) -> dict[str, object]:
    """Fail fast unless *path* preserves the direct-UniAnn TSV contract."""
    path = Path(path)
    with path.open() as handle:
        header = handle.readline().rstrip("\n\r")
        first = handle.readline().rstrip("\n\r")
    if header != JOINT_FRAME_TSV_HEADER:
        raise ValueError(
            "CNN-Mamba/UniAnn compatibility requires the original TSV header; "
            "do not remove it with tail/sed. Expected: " + JOINT_FRAME_TSV_HEADER
        )
    fields = first.split("\t")
    if len(fields) != len(JOINT_FRAME_TSV_COLUMNS):
        raise ValueError("first joint-frame data row does not have eight columns")
    if fields[1] != "1":
        raise ValueError(
            f"first joint-frame position must be one-based position 1, observed {fields[1]!r}"
        )
    if fields[2] not in {"+", "-"}:
        raise ValueError(f"invalid strand in first joint-frame row: {fields[2]!r}")
    probabilities = [float(value) for value in fields[3:7]]
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("first-row joint-frame probabilities must be in [0,1]")
    if abs(sum(probabilities) - 1.0) > 1e-4:
        raise ValueError("first-row joint-frame probabilities do not sum to one")

    sentinel_path = Path(str(path) + ".complete.json")
    sentinel = None
    if sentinel_path.is_file():
        sentinel = json.loads(sentinel_path.read_text())
        if sentinel.get("schema") != "per_base_joint_frame_probability_tsv_v1":
            raise ValueError(f"unexpected joint-frame sentinel schema: {sentinel_path}")
        if tuple(sentinel.get("columns", ())) != JOINT_FRAME_TSV_COLUMNS:
            raise ValueError(f"joint-frame sentinel columns disagree with TSV: {sentinel_path}")
    return {
        **UNIANN_HEADER_COMPATIBILITY,
        "path": str(path.resolve()),
        "header": header,
        "first_position_1based": 1,
        "sentinel": str(sentinel_path.resolve()) if sentinel is not None else None,
        "rows": None if sentinel is None else sentinel.get("rows"),
    }
