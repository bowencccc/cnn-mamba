import itertools

import torch

from cnn_mamba.structured_hmm import (
    NEG_INF,
    N_STATES,
    STATE,
    StructuredGeneHMM,
    build_partial_constraints,
)


def encode(sequence):
    table = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    return torch.tensor([[table[base] for base in sequence]])


def test_event_emissions_use_expected_motif_anchors():
    sequence = encode("ATGGTAGTAA")
    length = sequence.shape[1]
    splice = torch.zeros(1, length, 3)
    start_stop = torch.zeros(1, length, 3)
    frame = torch.zeros(1, length, 4)
    hmm = StructuredGeneHMM()
    emissions = hmm.emissions(sequence, splice, start_stop, frame)

    assert emissions[0, 0, STATE["start"]] > NEG_INF / 2
    assert emissions[0, 1, STATE["start"]] == NEG_INF
    assert emissions[0, 3, STATE["donor0"]] > NEG_INF / 2
    # AG begins at 5, so its internal acceptor state is on G at position 6.
    assert emissions[0, 6, STATE["acceptor0"]] > NEG_INF / 2
    # TAA begins at 7, so STOP is on its third base at position 9.
    assert emissions[0, 9, STATE["stop"]] > NEG_INF / 2


def test_log_partition_matches_brute_force_for_short_sequence():
    torch.manual_seed(4)
    hmm = StructuredGeneHMM()
    emissions = torch.randn(1, 3, N_STATES)
    actual = hmm.log_partition(emissions)[0]
    transition = hmm.transitions
    initial = torch.full((N_STATES,), NEG_INF)
    initial[STATE["intergenic"]] = 0
    initial[STATE["intron0"]:STATE["exon2"] + 1] = 0
    path_scores = []
    for path in itertools.product(range(N_STATES), repeat=3):
        score = initial[path[0]] + emissions[0, 0, path[0]]
        score = score + transition[path[0], path[1]] + emissions[0, 1, path[1]]
        score = score + transition[path[1], path[2]] + emissions[0, 2, path[2]]
        path_scores.append(score)
    expected = torch.logsumexp(torch.stack(path_scores), dim=0)
    torch.testing.assert_close(actual, expected)


def test_unconstrained_partial_loss_is_zero():
    torch.manual_seed(8)
    hmm = StructuredGeneHMM()
    emissions = torch.randn(2, 5, N_STATES)
    allowed = torch.ones_like(emissions, dtype=torch.bool)
    torch.testing.assert_close(
        hmm.partial_nll(emissions, allowed), torch.tensor(0.0), atol=1e-6, rtol=0
    )


def test_partial_constraints_preserve_a_complete_valid_gene_path():
    sequence = encode("ATGAAAGTCCAGAAATAA")
    length = sequence.shape[1]
    splice = torch.zeros(1, length, dtype=torch.long)
    splice[0, 6] = 1
    splice[0, 10] = 2
    start_stop = torch.zeros_like(splice)
    start_stop[0, 0] = 1
    start_stop[0, 15] = 2
    phase = torch.zeros_like(splice)
    phase[0, 0:6] = torch.tensor([0, 1, 2, 0, 1, 2])
    phase[0, 12:18] = torch.tensor([0, 1, 2, 0, 1, 2])
    cds = torch.zeros_like(splice, dtype=torch.bool)
    cds[0, 0:6] = True
    cds[0, 12:18] = True
    phase_mask = cds.clone()
    allowed, summary = build_partial_constraints(
        splice, start_stop, phase, phase_mask, cds
    )
    path_names = (
        "start", "exon1", "exon2", "exon0", "exon1", "exon2",
        "donor0", "intron0", "intron0", "intron0", "intron0", "acceptor0",
        "exon0", "exon1", "exon2", "exon0", "exon1", "stop",
    )
    path = torch.tensor([[STATE[name] for name in path_names]])
    assert allowed.gather(2, path.unsqueeze(-1)).all()
    assert summary.starts == summary.stops == summary.donors == summary.acceptors == 1

    logits3 = torch.zeros(1, length, 3)
    logits4 = torch.zeros(1, length, 4)
    hmm = StructuredGeneHMM()
    emissions = hmm.emissions(sequence, logits3, logits3, logits4)
    assert torch.isfinite(hmm.partial_nll(emissions, allowed))
