from __future__ import annotations

import math
import torch


def contact_metrics(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> dict:
    """Compute upper-triangle RNA contact metrics for batched contact maps."""
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    lengths = lengths.detach().cpu().long()
    rows = []
    for bsz in range(pred.shape[0]):
        length = int(lengths[bsz])
        p = pred[bsz].squeeze()[:length, :length] > 0.5
        y = target[bsz].squeeze()[:length, :length] > 0.5
        idx = torch.arange(length)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
        p = p[mask]
        y = y[mask]
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        tn = int((~p & ~y).sum())
        # Handle edge case: gt_pairs=0 and pred_pairs=0 → perfect (F1=1)
        if (tp + fn) == 0 and (tp + fp) == 0:
            precision, recall, f1, mcc = 1.0, 1.0, 1.0, 1.0
        elif (tp + fn) == 0 and (tp + fp) > 0:
            # GT has no pairs but model predicted some → all FP
            precision, recall, f1, mcc = 0.0, 1.0, 0.0, 0.0
        else:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
            mcc = ((tp * tn) - (fp * fn)) / denom
        rows.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "gt_pairs": tp + fn,
            "pred_pairs": tp + fp,
        })
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "n": 0}
    keys = ["precision", "recall", "f1", "mcc", "gt_pairs", "pred_pairs"]
    out = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    out["n"] = len(rows)
    return out
