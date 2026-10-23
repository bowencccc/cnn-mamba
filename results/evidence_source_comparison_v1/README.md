# Protein-evidence source comparison

Both models use chrX-in-training, 10-kb windows, 5-kb stride, the four-state beta=0.1 head, and the same chrX+ RefSeq evaluation.
UniAnn locus results use joint-softmax T=9 with unscaled donor/acceptor/start/stop scores.

## Mean candidate AUPRC

- Coleoptera protein evidence: 0.9070
- Drosophila protein evidence: 0.9329

## Strict locus F1

- Drosophila protein evidence; EviAnn: 86.12
- Drosophila protein evidence; UniAnn: 78.89
- Drosophila protein evidence; EviAnn+UniAnn: 90.40
- Coleoptera protein evidence; EviAnn: 73.60
- Coleoptera protein evidence; UniAnn: 72.39
- Coleoptera protein evidence; EviAnn+UniAnn: 83.59
