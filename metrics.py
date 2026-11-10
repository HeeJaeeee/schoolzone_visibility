from types import SimpleNamespace

import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import box_iou


def match_counts(predictions, targets):
    """Count class-aware one-to-one matches at IoU .5 after confidence filtering.

    Predictions: cls,conf,x1,y1,x2,y2. Targets: cls,x1,y1,x2,y2.
    Use the installed validator's matcher, with its IoU threshold fixed at .5.
    """
    if not predictions or not targets:
        return {"tp": 0, "fp": len(predictions), "fn": len(targets)}
    pred = torch.tensor(predictions, dtype=torch.float32)
    true = torch.tensor(targets, dtype=torch.float32)
    matched = DetectionValidator.match_predictions(
        SimpleNamespace(iouv=torch.tensor([0.5])), pred[:, 0], true[:, 0],
        box_iou(true[:, 1:5], pred[:, 2:6]),
    )
    tp = int(matched[:, 0].sum())
    return {"tp": tp, "fp": len(predictions) - tp, "fn": len(targets) - tp}
