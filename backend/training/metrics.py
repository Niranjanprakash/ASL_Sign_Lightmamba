import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Computes top-1, top-5 accuracy, precision, recall, and F1-scores.
    outputs shape: [B, num_classes]
    targets shape: [B]
    """
    preds = torch.argmax(outputs, dim=-1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Top-1 Accuracy
    top1_acc = accuracy_score(targets_np, preds)
    
    # Top-5 Accuracy (if num_classes >= 5)
    num_classes = outputs.shape[-1]
    if num_classes >= 5:
        _, top5_preds = torch.topk(outputs, k=5, dim=-1)
        top5_correct = torch.eq(top5_preds, targets.unsqueeze(-1)).any(dim=-1)
        top5_acc = top5_correct.float().mean().item()
    else:
        top5_acc = top1_acc # Fallback

    # Precision, Recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets_np, preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets_np, preds, average='weighted', zero_division=0
    )

    return {
        "accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted)
    }
