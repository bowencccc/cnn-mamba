"""Scalar temperature scaling utilities for four-state joint-frame logits."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


CLASS_ORDER = ("non_CDS", "frame_0", "frame_1", "frame_2")


def joint_frame_targets(phase_labels, phase_mask, cds_labels):
    """Return four-state targets and a mask excluding ambiguous CDS phase."""
    phase_mask = phase_mask.bool()
    cds_mask = cds_labels.bool()
    valid = (~cds_mask) | phase_mask
    targets = torch.zeros_like(phase_labels, dtype=torch.long)
    coding = cds_mask & phase_mask
    targets[coding] = phase_labels[coding].long() + 1
    return targets, valid


def temperature_objective(
    logits, targets, log_temperature, objective="empirical", noncoding_weight=0.1,
):
    """Cross entropy after dividing logits by a positive scalar temperature."""
    temperature = log_temperature.exp()
    per_base = F.cross_entropy(logits / temperature, targets, reduction="none")
    if objective == "empirical":
        return per_base.mean()
    if objective != "group_balanced":
        raise ValueError(f"unknown calibration objective: {objective}")
    coding = targets != 0
    loss = per_base.sum() * 0.0
    if coding.any():
        loss = loss + per_base[coding].mean()
    if noncoding_weight and (~coding).any():
        loss = loss + noncoding_weight * per_base[~coding].mean()
    return loss


def fit_temperature(
    logits, targets, objective="empirical", noncoding_weight=0.1,
    minimum=0.05, maximum=100.0,
):
    """Fit one scalar temperature with bounded LBFGS optimization."""
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("expected logits with shape (N, 4)")
    if targets.shape != logits.shape[:1]:
        raise ValueError("target shape does not match logits")
    if not (0 < minimum <= maximum):
        raise ValueError("temperature bounds must satisfy 0 < minimum <= maximum")
    logits = logits.detach().float().cpu()
    targets = targets.detach().long().cpu()
    log_minimum, log_maximum = math.log(minimum), math.log(maximum)
    parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [parameter], lr=0.25, max_iter=80, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        bounded = parameter.clamp(log_minimum, log_maximum).float()
        loss = temperature_objective(
            logits, targets, bounded, objective, noncoding_weight
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    log_temperature = float(parameter.detach().clamp(log_minimum, log_maximum))
    return float(math.exp(log_temperature))


def calibration_metrics(logits, targets, temperature=1.0, bins=20,
                        noncoding_weight=0.1):
    """Compute multiclass calibration and confidence summaries."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits.detach().float().cpu()
    targets = targets.detach().long().cpu()
    probabilities = (logits / temperature).softmax(-1)
    confidence, prediction = probabilities.max(-1)
    correct = prediction.eq(targets)
    one_hot = F.one_hot(targets, num_classes=4).float()
    nll = F.nll_loss(probabilities.clamp_min(1e-30).log(), targets)
    brier = ((probabilities - one_hot) ** 2).sum(-1).mean()
    coding = targets != 0
    per_base = F.cross_entropy(logits / temperature, targets, reduction="none")
    group_nll = per_base.sum() * 0.0
    if coding.any():
        group_nll = group_nll + per_base[coding].mean()
    if noncoding_weight and (~coding).any():
        group_nll = group_nll + noncoding_weight * per_base[~coding].mean()

    boundaries = torch.linspace(0, 1, bins + 1)
    ece = torch.zeros(())
    reliability = []
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (
            confidence >= lower if index == 0 else confidence > lower
        ) & (confidence <= upper)
        count = int(selected.sum())
        if count:
            bin_confidence = confidence[selected].mean()
            bin_accuracy = correct[selected].float().mean()
            ece += selected.float().mean() * (bin_confidence - bin_accuracy).abs()
            reliability.append({
                "lower": float(lower), "upper": float(upper), "count": count,
                "mean_confidence": float(bin_confidence),
                "accuracy": float(bin_accuracy),
            })
        else:
            reliability.append({
                "lower": float(lower), "upper": float(upper), "count": 0,
                "mean_confidence": None, "accuracy": None,
            })
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    return {
        "temperature": float(temperature),
        "samples": int(targets.numel()),
        "coding_samples": int(coding.sum()),
        "noncoding_samples": int((~coding).sum()),
        "accuracy": float(correct.float().mean()),
        "nll": float(nll),
        "group_balanced_nll": float(group_nll),
        "brier": float(brier),
        "ece": float(ece),
        "mean_max_probability": float(confidence.mean()),
        "median_max_probability": float(confidence.median()),
        "mean_entropy": float(entropy.mean()),
        "fraction_max_ge_0.90": float((confidence >= 0.90).float().mean()),
        "fraction_max_ge_0.99": float((confidence >= 0.99).float().mean()),
        "fraction_max_ge_0.999": float((confidence >= 0.999).float().mean()),
        "reliability_bins": reliability,
    }


def apply_temperature_numpy(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Stable NumPy softmax used by lightweight downstream checks."""
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=-1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=-1, keepdims=True)
