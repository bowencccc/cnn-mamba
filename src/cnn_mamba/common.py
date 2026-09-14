"""Candidate masks, initialization and bounded-memory validation metrics."""

import numpy as np
import torch
import torch.nn as nn

A, G, T = 0, 2, 3


def splice_candidate_masks(sequence):
    gt = torch.zeros_like(sequence, dtype=torch.bool)
    ag = torch.zeros_like(sequence, dtype=torch.bool)
    gt[..., :-1] = (sequence[..., :-1] == G) & (sequence[..., 1:] == T)
    ag[..., :-1] = (sequence[..., :-1] == A) & (sequence[..., 1:] == G)
    return gt, ag


def start_stop_candidate_masks(sequence):
    atg = torch.zeros_like(sequence, dtype=torch.bool)
    stop = torch.zeros_like(sequence, dtype=torch.bool)
    atg[..., :-2] = ((sequence[..., :-2] == A) &
                     (sequence[..., 1:-1] == T) &
                     (sequence[..., 2:] == G))
    x0, x1, x2 = sequence[..., :-2], sequence[..., 1:-1], sequence[..., 2:]
    stop[..., :-2] = (((x0 == T) & (x1 == A) & ((x2 == A) | (x2 == G))) |
                      ((x0 == T) & (x1 == G) & (x2 == A)))
    return atg, stop


def initialize_like_original(model):
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    model.embedding.apply(initialize)
    model.input_proj.apply(initialize)
    if hasattr(model, "norms"):
        model.norms.apply(initialize)
    model.final_norm.apply(initialize)
    model.splice_head.apply(initialize)
    model.start_stop_head.apply(initialize)


class BinnedPR:
    def __init__(self, bins=4096):
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)

    def add(self, scores, truth):
        indices = torch.clamp((scores.detach() * (self.bins - 1)).long(), 0, self.bins - 1)
        self.positive += torch.bincount(indices[truth], minlength=self.bins).cpu().numpy()
        self.negative += torch.bincount(indices[~truth], minlength=self.bins).cpu().numpy()

    def metrics(self):
        positives = int(self.positive.sum())
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(positives, 1)
        ap = float(np.sum(precision * np.diff(np.r_[0.0, recall])))
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        best = int(np.argmax(f1))
        return {"AP": ap, "F1": float(f1[best]), "P": float(precision[best]),
                "R": float(recall[best]),
                "threshold": float((self.bins - 1 - best) / (self.bins - 1)),
                "positives": positives,
                "candidates": positives + int(self.negative.sum())}


def format_metrics(metrics):
    lines = []
    for name in ("donor", "acceptor", "start", "stop"):
        row = metrics[name]
        lines.append(f"{name:8s} AP={row['AP']:.4f} F1={row['F1']:.4f} "
                     f"P={row['P']:.4f} R={row['R']:.4f} thr={row['threshold']:.4f} "
                     f"pos={row['positives']:,}/{row['candidates']:,}")
    return "\n".join(lines)
