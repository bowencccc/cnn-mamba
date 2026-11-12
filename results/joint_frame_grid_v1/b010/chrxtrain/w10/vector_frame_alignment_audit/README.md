# Vector-scaling frame-alignment audit

The CNN-Mamba joint-frame target assigns model phases 0, 1, and 2 to the
first, second, and third nucleotide of a translated codon. At zero-based
genomic position `n`, model phase `i` therefore belongs to UniAnn genomic
exon frame `(n - i) mod 3`.

UniAnn commit `2f5c0ab` uses `(n - i - 1) mod 3` in
`convert_coding_frame.pl`. That formula is appropriate only if the upstream
phase convention has the corresponding one-base offset. Applying it directly
to CNN-Mamba phases rotates every exon-frame emission by one state and causes
the low strict locus result recorded as `literal_builtin_header_removed`.

Passing the exported TSV with its header to the original Perl converter
increments the converter row counter before the first nucleotide. This cancels
the frame rotation, but the converter also emits a placeholder row for the
header and does not read `position_1based`, so the probability track follows
its historical one-row-offset behavior. The professor-facing workflow depends
on this exact interface and its validated result; the TSV output and UniAnn are
therefore left unchanged. New experiments must retain exactly one header row.

The coordinate-aware adapter shown here uses `(n - i) mod 3`, skips the TSV
header, and does not move the per-base probabilities. It is retained only as a
diagnostic: it reproduced the rounded locus metrics but is not the delivered
interface. All conditions otherwise use the same vector calibration, UniAnn
binary-odds temperature 3, non-CDS gate, four site-score file, FASTA,
reference, EviAnn annotation, UniAnn commit, and strict comparison.
