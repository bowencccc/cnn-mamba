"""Linear-probe fitting and metrics for belief-state experiments."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, classes)

    def forward(self, x):
        return self.linear(x)


def fit_linear_probe(
    train_x, train_y, input_dim, classes, weight_decay, seed=42,
    epochs=8, batch_size=8192, lr=1e-3, device="cuda",
):
    torch.manual_seed(seed)
    model = LinearProbe(input_dim, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.as_tensor(train_x)
    y = torch.as_tensor(train_y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        order = torch.randperm(len(y), generator=generator)
        model.train()
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            xb = x[idx].to(device=device, dtype=torch.float32, non_blocking=True)
            yb = y[idx].to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict_probabilities(model, features, batch_size=32768, device="cuda"):
    x = torch.as_tensor(features)
    rows = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size].to(device=device, dtype=torch.float32))
        rows.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(rows) if rows else np.empty((0, model.linear.out_features))


def expected_calibration_error(probabilities, labels, bins=15):
    confidence = probabilities.max(1)
    predicted = probabilities.argmax(1)
    correct = predicted == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(labels))
    ece = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index + 1 < bins else confidence <= 1.0
        )
        if mask.any():
            ece += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def four_state_metrics(probabilities, labels):
    labels = np.asarray(labels, dtype=np.int64)
    predicted = probabilities.argmax(1)
    coding = labels > 0
    eps = np.finfo(np.float64).tiny
    result = {
        "positions": int(len(labels)),
        "coding_positions": int(coding.sum()),
        "balanced_accuracy_4class": float(balanced_accuracy_score(labels, predicted)),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], eps)).mean()),
        "ece15": expected_calibration_error(probabilities, labels, 15),
        "coding_auprc": float(average_precision_score(coding, 1.0 - probabilities[:, 0])),
    }
    if coding.any():
        # This is deliberately named as a joint-head metric: a noncoding
        # prediction at a CDS position is counted as an error.
        result["joint_4class_cds_accuracy"] = float(
            (predicted[coding] == labels[coding]).mean()
        )
        result["joint_4class_cds_macro_f1"] = float(
            f1_score(labels[coding], predicted[coding], labels=[1, 2, 3], average="macro")
        )
        conditional = probabilities[coding, 1:4].argmax(1)
        phase = labels[coding] - 1
        result["conditional_frame_accuracy_3class"] = float(
            (conditional == phase).mean()
        )
        result["conditional_frame_balanced_accuracy_3class"] = float(
            balanced_accuracy_score(phase, conditional)
        )
        result["conditional_frame_macro_f1_3class"] = float(
            f1_score(phase, conditional, labels=[0, 1, 2], average="macro")
        )
    else:
        for name in (
            "joint_4class_cds_accuracy", "joint_4class_cds_macro_f1",
            "conditional_frame_accuracy_3class",
            "conditional_frame_balanced_accuracy_3class",
            "conditional_frame_macro_f1_3class",
        ):
            result[name] = math.nan
    return result


def seven_state_metrics(probabilities, labels):
    labels = np.asarray(labels, dtype=np.int64)
    predicted = probabilities.argmax(1)
    intron = labels >= 4
    result = {
        "positions": int(len(labels)),
        "intron_positions": int(intron.sum()),
        "balanced_accuracy_7class": float(balanced_accuracy_score(labels, predicted)),
        "nll": float(-np.log(np.maximum(
            probabilities[np.arange(len(labels)), labels], np.finfo(np.float64).tiny
        )).mean()),
        "ece15": expected_calibration_error(probabilities, labels, 15),
    }
    if intron.any():
        result["joint_7class_intron_accuracy"] = float(
            (predicted[intron] == labels[intron]).mean()
        )
        phase = labels[intron] - 4
        conditional = probabilities[intron, 4:7].argmax(1)
        result["conditional_intron_phase_accuracy_3class"] = float(
            (conditional == phase).mean()
        )
        result["conditional_intron_phase_balanced_accuracy_3class"] = float(
            balanced_accuracy_score(phase, conditional)
        )
        result["conditional_intron_phase_macro_f1_3class"] = float(
            f1_score(phase, conditional, labels=[0, 1, 2], average="macro")
        )
        counts = np.bincount(labels[intron] - 4, minlength=3)
        result["intron_majority_baseline"] = float(counts.max() / counts.sum())
        result["intron_random_baseline"] = 1.0 / 3.0
    else:
        for name in (
            "joint_7class_intron_accuracy",
            "conditional_intron_phase_accuracy_3class",
            "conditional_intron_phase_balanced_accuracy_3class",
            "conditional_intron_phase_macro_f1_3class",
            "intron_majority_baseline", "intron_random_baseline",
        ):
            result[name] = math.nan
    return result


def three_state_metrics(probabilities, labels):
    """Metrics for a probe trained and evaluated only on phase 0/1/2."""
    labels = np.asarray(labels, dtype=np.int64)
    predicted = probabilities.argmax(1)
    counts = np.bincount(labels, minlength=3)
    eps = np.finfo(np.float64).tiny
    return {
        "positions": int(len(labels)),
        "accuracy_3class": float((predicted == labels).mean()),
        "balanced_accuracy_3class": float(
            balanced_accuracy_score(labels, predicted)
        ),
        "macro_f1_3class": float(
            f1_score(labels, predicted, labels=[0, 1, 2], average="macro")
        ),
        "nll_3class": float(-np.log(np.maximum(
            probabilities[np.arange(len(labels)), labels], eps,
        )).mean()),
        "ece15_3class": expected_calibration_error(probabilities, labels, 15),
        "majority_baseline": float(counts.max() / counts.sum()),
        "random_baseline": 1.0 / 3.0,
        "class0_positions": int(counts[0]),
        "class1_positions": int(counts[1]),
        "class2_positions": int(counts[2]),
    }
