"""A small gene-structure CRF/HMM on top of the four CNN--Mamba heads.

The module deliberately keeps the neural backbone frozen.  Its trainable
parameters calibrate only the joint-frame logits; splice and start/stop event
scores retain their original scale.  Partial-label likelihood can therefore
learn from EviAnn positives without treating every unannotated base as a
negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


STATE_NAMES = (
    "intergenic",
    "intron0", "intron1", "intron2",
    "exon0", "exon1", "exon2",
    "start",
    "stop",
    "donor0", "donor1", "donor2",
    "acceptor0", "acceptor1", "acceptor2",
)
STATE = {name: index for index, name in enumerate(STATE_NAMES)}
N_STATES = len(STATE_NAMES)
NEG_INF = -1.0e4


@dataclass
class ConstraintSummary:
    constrained_bases: int
    starts: int
    stops: int
    donors: int
    acceptors: int
    coding_bases: int


def _transition_scores(device=None, dtype=torch.float32):
    """Return the fixed biologically valid 15-state transition graph."""
    scores = torch.full((N_STATES, N_STATES), NEG_INF, device=device, dtype=dtype)

    def allow(source, target):
        scores[STATE[source], STATE[target]] = 0.0

    allow("intergenic", "intergenic")
    allow("intergenic", "start")
    allow("start", "exon1")

    for phase in range(3):
        next_phase = (phase + 1) % 3
        allow(f"exon{phase}", f"exon{next_phase}")
        allow(f"exon{phase}", f"donor{next_phase}")
        allow(f"donor{phase}", f"intron{phase}")
        allow(f"intron{phase}", f"intron{phase}")
        allow(f"intron{phase}", f"acceptor{phase}")
        allow(f"acceptor{phase}", f"exon{phase}")

    # STOP is placed on the third nucleotide of a stop codon.  The first two
    # nucleotides remain exon0/exon1, matching the phase labels in the NPZs.
    allow("exon1", "stop")
    allow("stop", "intergenic")
    return scores


class StructuredGeneHMM(nn.Module):
    """Fifteen-state linear-chain model with HMM-aware frame calibration."""

    def __init__(self):
        super().__init__()
        # exp(log_scale) is a single temperature-like multiplier.  Centering
        # the biases removes their unidentifiable common offset.
        self.frame_log_scale = nn.Parameter(torch.zeros(()))
        self.frame_bias = nn.Parameter(torch.zeros(4))
        self.register_buffer("transitions", _transition_scores())

    @property
    def frame_scale(self):
        return self.frame_log_scale.exp()

    def calibrated_frame_log_probs(self, frame_logits):
        bias = self.frame_bias - self.frame_bias.mean()
        return F.log_softmax(frame_logits * self.frame_scale + bias, dim=-1)

    def emissions(self, sequence, splice_logits, start_stop_logits, frame_logits):
        """Map neural logits to motif-masked state emission potentials.

        Sequence encoding is A=0, C=1, G=2, T=3, N=4.  Donor and start states
        are anchored at the first motif nucleotide.  Acceptor is represented
        on the G of AG, and STOP on the third nucleotide of the stop codon.
        """
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape [batch, length]")
        batch, length = sequence.shape
        expected = (batch, length)
        if splice_logits.shape[:2] != expected or splice_logits.shape[-1] != 3:
            raise ValueError("splice_logits must have shape [batch, length, 3]")
        if start_stop_logits.shape[:2] != expected or start_stop_logits.shape[-1] != 3:
            raise ValueError("start_stop_logits must have shape [batch, length, 3]")
        if frame_logits.shape[:2] != expected or frame_logits.shape[-1] != 4:
            raise ValueError("frame_logits must have shape [batch, length, 4]")

        splice = F.log_softmax(splice_logits, dim=-1)
        start_stop = F.log_softmax(start_stop_logits, dim=-1)
        frame = self.calibrated_frame_log_probs(frame_logits)
        result = frame_logits.new_full((batch, length, N_STATES), NEG_INF)

        result[..., STATE["intergenic"]] = frame[..., 0]
        for phase in range(3):
            result[..., STATE[f"intron{phase}"]] = frame[..., 0]
            result[..., STATE[f"exon{phase}"]] = frame[..., phase + 1]

        is_gt = torch.zeros_like(sequence, dtype=torch.bool)
        is_ag_second = torch.zeros_like(sequence, dtype=torch.bool)
        is_atg = torch.zeros_like(sequence, dtype=torch.bool)
        is_stop_third = torch.zeros_like(sequence, dtype=torch.bool)
        if length >= 2:
            is_gt[:, :-1] = (sequence[:, :-1] == 2) & (sequence[:, 1:] == 3)
            is_ag_second[:, 1:] = (sequence[:, :-1] == 0) & (sequence[:, 1:] == 2)
        if length >= 3:
            is_atg[:, :-2] = (
                (sequence[:, :-2] == 0)
                & (sequence[:, 1:-1] == 3)
                & (sequence[:, 2:] == 2)
            )
            first = sequence[:, :-2]
            second = sequence[:, 1:-1]
            third = sequence[:, 2:]
            stop = (first == 3) & (
                ((second == 0) & ((third == 0) | (third == 2)))
                | ((second == 2) & (third == 0))
            )
            is_stop_third[:, 2:] = stop

        start_potential = start_stop[..., 1] + frame[..., 1]
        shifted_stop = frame_logits.new_full((batch, length), NEG_INF)
        if length >= 3:
            shifted_stop[:, 2:] = start_stop[:, :-2, 2]
        stop_potential = shifted_stop + frame[..., 3]
        result[..., STATE["start"]] = torch.where(
            is_atg, start_potential, result[..., STATE["start"]]
        )
        result[..., STATE["stop"]] = torch.where(
            is_stop_third, stop_potential, result[..., STATE["stop"]]
        )

        shifted_acceptor = frame_logits.new_full((batch, length), NEG_INF)
        if length >= 2:
            shifted_acceptor[:, 1:] = splice[:, :-1, 2]
        for phase in range(3):
            donor_value = splice[..., 1] + frame[..., 0]
            acceptor_value = shifted_acceptor + frame[..., 0]
            result[..., STATE[f"donor{phase}"]] = torch.where(
                is_gt, donor_value, result[..., STATE[f"donor{phase}"]]
            )
            result[..., STATE[f"acceptor{phase}"]] = torch.where(
                is_ag_second,
                acceptor_value,
                result[..., STATE[f"acceptor{phase}"]],
            )
        return result

    def log_partition(self, emissions, allowed=None):
        """Compute log sum of all paths, optionally under state constraints."""
        if emissions.ndim != 3 or emissions.shape[-1] != N_STATES:
            raise ValueError(f"emissions must have shape [batch, length, {N_STATES}]")
        if emissions.shape[1] == 0:
            raise ValueError("empty sequences are not supported")
        if allowed is not None:
            if allowed.shape != emissions.shape or allowed.dtype != torch.bool:
                raise ValueError("allowed must be a boolean tensor matching emissions")
            emissions = emissions.masked_fill(~allowed, NEG_INF)

        # Crops can begin inside an exon/intron.  Boundary event states are not
        # valid initial states, but all ordinary context states are.
        initial = emissions.new_full((N_STATES,), NEG_INF)
        initial[STATE["intergenic"]] = 0.0
        initial[STATE["intron0"]:STATE["exon2"] + 1] = 0.0
        alpha = initial + emissions[:, 0]
        transitions = self.transitions.to(dtype=emissions.dtype)
        for position in range(1, emissions.shape[1]):
            alpha = torch.logsumexp(
                alpha.unsqueeze(2) + transitions.unsqueeze(0), dim=1
            ) + emissions[:, position]
        return torch.logsumexp(alpha, dim=1)

    def partial_nll(self, emissions, allowed):
        all_paths = self.log_partition(emissions)
        compatible_paths = self.log_partition(emissions, allowed)
        loss = all_paths - compatible_paths
        if not torch.isfinite(loss).all():
            raise RuntimeError("constraints admit no finite HMM path")
        return loss.mean()

    def viterbi(self, emissions):
        """Return best state path and its score (intended for short regions)."""
        initial = emissions.new_full((N_STATES,), NEG_INF)
        initial[STATE["intergenic"]] = 0.0
        initial[STATE["intron0"]:STATE["exon2"] + 1] = 0.0
        score = initial + emissions[:, 0]
        backpointers = []
        transitions = self.transitions.to(dtype=emissions.dtype)
        for position in range(1, emissions.shape[1]):
            candidates = score.unsqueeze(2) + transitions.unsqueeze(0)
            score, previous = candidates.max(dim=1)
            score = score + emissions[:, position]
            backpointers.append(previous)
        best_score, state = score.max(dim=1)
        path = [state]
        for previous in reversed(backpointers):
            state = previous.gather(1, state[:, None]).squeeze(1)
            path.append(state)
        return torch.stack(list(reversed(path)), dim=1), best_score

    def calibration_dict(self):
        scale = float(self.frame_scale.detach().cpu())
        bias = self.frame_bias.detach().cpu()
        bias = bias - bias.mean()
        names = ("noncoding", "frame0", "frame1", "frame2")
        return {
            "schema": "structured_hmm_frame_calibration_v1",
            "method": "scalar frame scale and class bias fitted by partial HMM likelihood",
            "class_order": list(names),
            "vector_scale": {name: scale for name in names},
            "vector_bias": {name: float(value) for name, value in zip(names, bias)},
        }


def build_partial_constraints(
    splice_labels, start_stop_labels, phase_labels, phase_mask, cds_labels,
):
    """Create compatible state sets from positive-only EviAnn supervision.

    Unannotated positions remain unconstrained.  Known CDS phase bases and
    positive boundary events are enforced.  Acceptor and stop labels are moved
    to their internal HMM state anchors (+1 and +2 bp respectively).
    """
    tensors = (splice_labels, start_stop_labels, phase_labels, phase_mask, cds_labels)
    if any(item.ndim != 2 for item in tensors):
        raise ValueError("all label tensors must have shape [batch, length]")
    if any(item.shape != splice_labels.shape for item in tensors[1:]):
        raise ValueError("all label tensors must have identical shapes")
    batch, length = splice_labels.shape
    device = splice_labels.device
    allowed = torch.ones((batch, length, N_STATES), dtype=torch.bool, device=device)

    valid_phase = phase_mask.bool() & cds_labels.bool()
    for phase in range(3):
        mask = valid_phase & (phase_labels == phase)
        if mask.any():
            allowed[mask] = False
            allowed[..., STATE[f"exon{phase}"]][mask] = True

    def restrict(mask, states):
        if mask.any():
            allowed[mask] = False
            for state in states:
                allowed[..., state][mask] = True

    starts = start_stop_labels == 1
    restrict(starts, [STATE["start"]])

    stops_source = start_stop_labels == 2
    stops = torch.zeros_like(stops_source)
    if length >= 3:
        stops[:, 2:] = stops_source[:, :-2]
    restrict(stops, [STATE["stop"]])

    donors = splice_labels == 1
    for phase in range(3):
        donor_phase = torch.zeros_like(donors)
        if length >= 2:
            donor_phase[:, 1:] = (
                valid_phase[:, :-1] & (phase_labels[:, :-1] == phase)
            )
        mask = donors & donor_phase
        restrict(mask, [STATE[f"donor{(phase + 1) % 3}"]])
    donor_unknown = donors.clone()
    if length >= 2:
        donor_unknown[:, 1:] &= ~valid_phase[:, :-1]
    restrict(donor_unknown, [STATE[f"donor{phase}"] for phase in range(3)])

    acceptor_source = splice_labels == 2
    acceptors = torch.zeros_like(acceptor_source)
    if length >= 2:
        acceptors[:, 1:] = acceptor_source[:, :-1]
    acceptor_known = torch.zeros_like(acceptors)
    if length >= 3:
        for phase in range(3):
            mask = torch.zeros_like(acceptors)
            mask[:, 1:-1] = (
                acceptors[:, 1:-1]
                & valid_phase[:, 2:]
                & (phase_labels[:, 2:] == phase)
            )
            restrict(mask, [STATE[f"acceptor{phase}"]])
            acceptor_known |= mask
    restrict(
        acceptors & ~acceptor_known,
        [STATE[f"acceptor{phase}"] for phase in range(3)],
    )

    constrained = ~allowed.all(dim=-1)
    summary = ConstraintSummary(
        constrained_bases=int(constrained.sum()),
        starts=int(starts.sum()),
        stops=int(stops.sum()),
        donors=int(donors.sum()),
        acceptors=int(acceptors.sum()),
        coding_bases=int(valid_phase.sum()),
    )
    return allowed, summary
