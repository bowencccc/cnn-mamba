#!/usr/bin/env python3
"""Compare strand-aware EviAnn and CHESS splice/start/stop site sets."""

from collections import defaultdict
import csv
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pysam


ROOT = Path(__file__).resolve().parents[2]
EVIANN_GFF = (
    ROOT
    / "data/raw/human_eviann_pseudo_label.gff"
)
CHESS_GTF = ROOT / "data/raw/human_chess3.1.3.gtf"
FASTA = ROOT / "data/raw/human_grch38.fa"
OUTPUT = ROOT / "work/human_annotation_overlap"

CHR_TO_NC = {
    "chr1": "NC_000001.11", "chr2": "NC_000002.12",
    "chr3": "NC_000003.12", "chr4": "NC_000004.12",
    "chr5": "NC_000005.10", "chr6": "NC_000006.12",
    "chr7": "NC_000007.14", "chr8": "NC_000008.11",
    "chr9": "NC_000009.12", "chr10": "NC_000010.11",
    "chr11": "NC_000011.10", "chr12": "NC_000012.12",
    "chr13": "NC_000013.11", "chr14": "NC_000014.9",
    "chr15": "NC_000015.10", "chr16": "NC_000016.10",
    "chr17": "NC_000017.11", "chr18": "NC_000018.10",
    "chr19": "NC_000019.10", "chr20": "NC_000020.11",
    "chr21": "NC_000021.9", "chr22": "NC_000022.11",
    "chrX": "NC_000023.11", "chrY": "NC_000024.10",
}
NC_TO_CHR = {accession: chrom for chrom, accession in CHR_TO_NC.items()}
SITE_TYPES = ("donor", "acceptor", "start", "stop")


def parse_gff_attrs(text):
    attrs = {}
    for item in text.strip().split(";"):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            attrs[key] = value
    return attrs


def parse_gtf_attrs(text):
    attrs = {}
    for item in text.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition(" ")
        if value:
            attrs[key] = value.strip().strip('"')
    return attrs


def parse_eviann(path):
    tx_meta = {}
    exons = defaultdict(list)
    cds = defaultdict(list)
    primary = set(NC_TO_CHR)
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in primary:
                continue
            feature = fields[2]
            if feature not in ("mRNA", "exon", "CDS"):
                continue
            chrom = NC_TO_CHR[fields[0]]
            strand = fields[6]
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            attrs = parse_gff_attrs(fields[8])
            if feature == "mRNA":
                tid = attrs.get("ID")
                if tid:
                    tx_meta[tid] = (chrom, strand)
                continue
            parents = attrs.get("Parent", "").split(",")
            for tid in parents:
                tid = tid.strip()
                if not tid:
                    continue
                (exons if feature == "exon" else cds)[tid].append((start0, end0))
    return tx_meta, exons, cds


def parse_chess(path):
    tx_meta = {}
    exons = defaultdict(list)
    cds = defaultdict(list)
    primary = set(CHR_TO_NC)
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in primary:
                continue
            feature = fields[2]
            if feature not in ("transcript", "exon", "CDS"):
                continue
            attrs = parse_gtf_attrs(fields[8])
            tid = attrs.get("transcript_id")
            if not tid:
                continue
            chrom, strand = fields[0], fields[6]
            tx_meta[tid] = (chrom, strand)
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            if feature == "exon":
                exons[tid].append((start0, end0))
            elif feature == "CDS":
                cds[tid].append((start0, end0))
    return tx_meta, exons, cds


def extract_sites(tx_meta, exons, cds, fasta_path):
    sites = {name: set() for name in SITE_TYPES}
    fasta = pysam.FastaFile(str(fasta_path))
    rejected = defaultdict(int)
    for tid, cds_segments in cds.items():
        if not cds_segments or tid not in tx_meta or tid not in exons:
            continue
        chrom, strand = tx_meta[tid]
        accession = CHR_TO_NC[chrom]
        chrom_len = fasta.get_reference_length(accession)
        cds_lo = min(start for start, _ in cds_segments)
        cds_hi = max(end for _, end in cds_segments)

        exon_segments = sorted(set(exons[tid]))
        for (_, left_end), (right_start, _) in zip(exon_segments, exon_segments[1:]):
            if right_start <= left_end:
                continue
            low_site = left_end
            high_site = right_start - 2
            if not (cds_lo <= low_site < cds_hi and cds_lo <= high_site < cds_hi):
                continue
            low_motif = fasta.fetch(accession, low_site, low_site + 2).upper()
            high_motif = fasta.fetch(accession, high_site, high_site + 2).upper()
            canonical = (
                low_motif == "GT" and high_motif == "AG"
                if strand == "+"
                else low_motif == "CT" and high_motif == "AC"
            )
            if not canonical:
                rejected["splice"] += 1
                continue
            if strand == "+":
                sites["donor"].add((chrom, strand, low_site))
                sites["acceptor"].add((chrom, strand, high_site))
            else:
                sites["donor"].add((chrom, strand, high_site))
                sites["acceptor"].add((chrom, strand, low_site))

        segments = sorted(cds_segments)
        start_position = segments[0][0] if strand == "+" else segments[-1][1] - 3
        start_motif = (
            fasta.fetch(accession, start_position, start_position + 3).upper()
            if 0 <= start_position <= chrom_len - 3
            else ""
        )
        if (strand == "+" and start_motif == "ATG") or (
            strand == "-" and start_motif == "CAT"
        ):
            sites["start"].add((chrom, strand, start_position))
        else:
            rejected["start"] += 1

        stop_position = segments[-1][1] - 3 if strand == "+" else segments[0][0]
        stop_motif = (
            fasta.fetch(accession, stop_position, stop_position + 3).upper()
            if 0 <= stop_position <= chrom_len - 3
            else ""
        )
        if (strand == "+" and stop_motif in {"TAA", "TAG", "TGA"}) or (
            strand == "-" and stop_motif in {"TTA", "CTA", "TCA"}
        ):
            sites["stop"].add((chrom, strand, stop_position))
        else:
            rejected["stop"] += 1
    fasta.close()
    return sites, dict(rejected)


def summarize(eviann, chess, scope, chromosomes):
    rows = []
    for name in SITE_TYPES:
        a = {site for site in eviann[name] if site[0] in chromosomes}
        b = {site for site in chess[name] if site[0] in chromosomes}
        overlap = a & b
        union = a | b
        rows.append(
            {
                "scope": scope,
                "site_type": name,
                "eviann_total": len(a),
                "chess_total": len(b),
                "eviann_only": len(a - b),
                "overlap": len(overlap),
                "chess_only": len(b - a),
                "eviann_overlap_pct": 100 * len(overlap) / max(len(a), 1),
                "chess_overlap_pct": 100 * len(overlap) / max(len(b), 1),
                "jaccard_pct": 100 * len(overlap) / max(len(union), 1),
            }
        )
    return rows


def draw_venn(rows, output_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ("#4C78A8", "#F58518")
    for axis, row in zip(axes.flat, rows):
        axis.set_aspect("equal")
        axis.axis("off")
        axis.add_patch(Circle((0.42, 0.5), 0.31, color=colors[0], alpha=0.48))
        axis.add_patch(Circle((0.68, 0.5), 0.31, color=colors[1], alpha=0.48))
        axis.text(0.16, 0.5, f"{row['eviann_only']:,}", ha="center", va="center", fontsize=15)
        axis.text(0.55, 0.5, f"{row['overlap']:,}", ha="center", va="center", fontsize=15, fontweight="bold")
        axis.text(0.94, 0.5, f"{row['chess_only']:,}", ha="center", va="center", fontsize=15)
        axis.text(0.30, 0.16, f"EviAnn\nN={row['eviann_total']:,}", ha="center", va="center", fontsize=11)
        axis.text(0.80, 0.16, f"CHESS 3.1.3\nN={row['chess_total']:,}", ha="center", va="center", fontsize=11)
        axis.set_title(
            f"{row['site_type'].capitalize()}\n"
            f"EviAnn shared={row['eviann_overlap_pct']:.1f}% | "
            f"CHESS shared={row['chess_overlap_pct']:.1f}%",
            fontsize=13,
        )
        axis.set_xlim(-0.2, 1.2)
        axis.set_ylim(0.05, 0.95)
    fig.suptitle(title + "\nExact strand-aware genomic positions", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Parsing EviAnn...", flush=True)
    e_meta, e_exons, e_cds = parse_eviann(EVIANN_GFF)
    print(f"EviAnn transcripts={len(e_meta):,} CDS transcripts={len(e_cds):,}", flush=True)
    print("Parsing CHESS...", flush=True)
    c_meta, c_exons, c_cds = parse_chess(CHESS_GTF)
    print(f"CHESS transcripts={len(c_meta):,} CDS transcripts={len(c_cds):,}", flush=True)
    print("Extracting EviAnn sites...", flush=True)
    e_sites, e_rejected = extract_sites(e_meta, e_exons, e_cds, FASTA)
    print("Extracting CHESS sites...", flush=True)
    c_sites, c_rejected = extract_sites(c_meta, c_exons, c_cds, FASTA)

    scopes = {
        "all_primary": set(CHR_TO_NC),
        "chr1": {"chr1"},
    }
    all_rows = []
    for scope, chromosomes in scopes.items():
        rows = summarize(e_sites, c_sites, scope, chromosomes)
        all_rows.extend(rows)
        draw_venn(
            rows,
            OUTPUT / f"venn_{scope}.png",
            "EviAnn vs CHESS site overlap — "
            + ("all primary chromosomes" if scope == "all_primary" else "chr1"),
        )
        print(f"\n{scope}", flush=True)
        for row in rows:
            print(
                f"{row['site_type']:8s} EviAnn={row['eviann_total']:,} "
                f"shared={row['overlap']:,} CHESS={row['chess_total']:,} "
                f"Jaccard={row['jaccard_pct']:.1f}%",
                flush=True,
            )

    with open(OUTPUT / "overlap_summary.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    metadata = {
        "definition": "exact (chromosome, strand, 0-based motif-start position); canonical CDS-internal splice sites",
        "eviann_rejected_motif_counts": e_rejected,
        "chess_rejected_motif_counts": c_rejected,
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT / "metadata.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in metadata.items()) + "\n"
    )
    print(f"\nFinished in {(time.time() - started) / 60:.1f} min: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
