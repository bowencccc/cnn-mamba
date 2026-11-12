#!/usr/bin/env bash
# Rebuild the UniAnn strict-evaluation inputs for Drosophila chrX+ from the
# full NCBI RefSeq GTF. These are evaluation references, not training labels.
#
#   dmel_chrX_plus_CDS_reference.gtf  (== NC_004354.4.p.CDS.uniq.gtf)
#     sha256 cc0bb477a4d89d745555d1e98f4d095ba286ac31ce03ad984ec2eb894c588ee5
#   dmel_chrX.fa                      (== NC_004354.4.fa)
#     sha256 8d4513bdf36b879de0b9e5b4fa5e54d5c6cf1ddaf66b73feee4de1fa6e4ab979
#
# CDS reference: chrX (NC_004354.4), plus strand, CDS + stop_codon records
# relabelled as exon (so the stop codon joins the terminal coding exon and the
# structure spans start..stop, no UTRs), then `gffread -M` merges transcripts
# with identical CDS structure: 2,696 transcript IDs / 17,017 records ->
# 1,783 unique transcripts / 9,718 exons (1,070 gffcompare loci).
#
# Byte-identical output was verified with gffread 0.12.8 (UniAnn/bin) and
# samtools 1.15.1 (default 60-column FASTA wrapping).
#
# Usage: scripts/make_dmel_chrx_uniann_reference.sh [output_dir]
#   env: GFFREAD (default $UNIANN_ROOT/bin/gffread), SAMTOOLS (default samtools)

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
raw="$repo_dir/data/raw"
out_dir="${1:-$raw}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
gffread="${GFFREAD:-$uniann_root/bin/gffread}"
samtools="${SAMTOOLS:-samtools}"
gtf="$raw/dmel_reference.gtf"
fasta="$raw/dmel_genome.fa"

mkdir -p "$out_dir"
cds_exons="$out_dir/NC_004354.4.plus.refseq.cds_as_exons.gtf"
reference="$out_dir/NC_004354.4.p.CDS.uniq.gtf"
chrx="$out_dir/NC_004354.4.fa"

awk 'BEGIN{FS=OFS="\t"}
     /^#/ {print; next}
     $1=="NC_004354.4" &&
     $7=="+" &&
     ($3=="CDS" || $3=="stop_codon") {
         $3="exon";
         $8=".";
         print
     }' "$gtf" > "$cds_exons"

"$gffread" -M -T "$cds_exons" -o "$reference"
"$samtools" faidx "$fasta" NC_004354.4 > "$chrx"

if [[ "$out_dir" != "$raw" ]]; then
    ln -sfn "$reference" "$raw/dmel_chrX_plus_CDS_reference.gtf"
    ln -sfn "$chrx" "$raw/dmel_chrX.fa"
else
    mv "$reference" "$raw/dmel_chrX_plus_CDS_reference.gtf"
    mv "$chrx" "$raw/dmel_chrX.fa"
fi

sha256sum "$raw/dmel_chrX_plus_CDS_reference.gtf" "$raw/dmel_chrX.fa"
