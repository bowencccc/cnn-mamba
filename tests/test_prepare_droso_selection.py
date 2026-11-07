import tempfile
import unittest
from pathlib import Path

from cnn_mamba.prepare_droso import parse_cds_phase_records, parse_eviann


class EviAnnTranscriptSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.gff = Path(self.temporary.name) / "labels.gff"
        records = ["##gff-version 3"]

        def add(tid, locus, evidence, class_code, cds_parts):
            start = min(part[0] for part in cds_parts)
            end = max(part[1] for part in cds_parts)
            attrs = (
                f"ID={tid};Parent={locus};evidence={evidence};"
                f"class={class_code};geneID={locus}"
            )
            records.append(
                f"NC_004354.4\tEviAnn\tmRNA\t{start}\t{end}\t.\t+\t.\t{attrs}"
            )
            for index, (left, right) in enumerate(cds_parts, start=1):
                records.append(
                    "NC_004354.4\tEviAnn\texon\t"
                    f"{left}\t{right}\t.\t+\t.\tParent={tid};ID={tid}:exon:{index}"
                )
                records.append(
                    "NC_004354.4\tEviAnn\tCDS\t"
                    f"{left}\t{right}\t.\t+\t0\tParent={tid};ID={tid}:CDS:{index}"
                )

        # Exact complete transcripts outrank a longer class-k transcript.
        add("tx_equal_short", "geneA", "complete", "=", [(101, 200)])
        add("tx_equal_long", "geneA", "complete", "=", [(301, 450)])
        add("tx_k_longest", "geneA", "complete", "k", [(501, 750)])
        # If exact is absent, select the longest complete class-k transcript.
        add("tx_k_short", "geneB", "complete", "k", [(801, 850)])
        add("tx_k_long", "geneB", "complete", "k", [(901, 1020)])
        # Neither incomplete exact nor complete class-j is eligible.
        add("tx_incomplete_equal", "geneC", "protein_only", "=", [(1101, 1300)])
        add("tx_complete_j", "geneD", "complete", "j", [(1401, 1600)])
        self.gff.write_text("\n".join(records) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_priority_and_fallback_selection(self):
        meta, exons, cds, summary = parse_eviann(
            self.gff, "longest-complete-equal-else-k"
        )
        self.assertEqual(set(meta), {"tx_equal_long", "tx_k_long"})
        self.assertEqual(set(exons), set(meta))
        self.assertEqual(set(cds), set(meta))
        self.assertEqual(summary["input_loci"], 4)
        self.assertEqual(summary["selected_complete_class_equal"], 1)
        self.assertEqual(summary["selected_complete_class_k"], 1)
        self.assertEqual(summary["skipped_loci_without_complete_equal_or_k"], 2)

    def test_default_keeps_all_transcripts(self):
        meta, _exons, _cds, summary = parse_eviann(self.gff)
        self.assertEqual(len(meta), 7)
        self.assertEqual(summary["mode"], "all_transcripts")

    def test_longest_cds_per_locus_ignores_evidence_and_class(self):
        meta, _exons, _cds, summary = parse_eviann(
            self.gff, "longest-cds-per-locus"
        )
        self.assertEqual(
            set(meta),
            {
                "tx_k_longest", "tx_k_long", "tx_incomplete_equal",
                "tx_complete_j",
            },
        )
        self.assertEqual(summary["input_loci"], 4)
        self.assertEqual(summary["selected_loci"], 4)
        self.assertEqual(summary["skipped_loci_without_cds"], 0)

    def test_complete_equal_else_k_falls_back_per_locus(self):
        meta, _exons, _cds, summary = parse_eviann(
            self.gff, "complete-equal-else-k-with-fallback"
        )
        self.assertEqual(
            set(meta),
            {
                "tx_equal_long", "tx_k_long", "tx_incomplete_equal",
                "tx_complete_j",
            },
        )
        self.assertEqual(summary["selected_complete_class_equal"], 1)
        self.assertEqual(summary["selected_complete_class_k"], 1)
        self.assertEqual(summary["selected_longest_cds_fallback"], 2)
        self.assertEqual(summary["selected_loci"], 4)

    def test_phase_records_are_limited_to_selected_transcripts(self):
        records = parse_cds_phase_records(
            self.gff, allowed_transcripts={"tx_equal_long", "tx_k_long"}
        )
        selected = records["NC_004354.4"]["+"]
        self.assertEqual(len(selected), 2)
        self.assertIn((300, 450, 0), selected)
        self.assertIn((900, 1020, 0), selected)


if __name__ == "__main__":
    unittest.main()
