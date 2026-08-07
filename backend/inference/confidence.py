import torch
from backend.config import CONFIDENCE_THRESHOLD

def calibrate_prediction(logits: torch.Tensor, threshold: float = CONFIDENCE_THRESHOLD) -> tuple:
    """
    Applies softmax to logits, fetches class prediction, and compares against threshold.
    Returns:
        - pred_class_idx: int (or -1 if uncertain)
        - confidence: float
        - top_k_probs: list of dicts {"class_id": int, "probability": float}
    """
    probs = torch.softmax(logits, dim=-1).squeeze(0) # [num_classes]
    
    # Sort probabilities
    top_probs, top_indices = torch.topk(probs, k=min(3, len(probs)))
    
    confidence = float(top_probs[0].item())
    pred_idx = int(top_indices[0].item())
    
    top_k_list = []
    for p, idx in zip(top_probs, top_indices):
        top_k_list.append({
            "class_id": int(idx.item()),
            "probability": float(p.item())
        })
        
    if confidence < threshold:
        return -1, confidence, top_k_list
    else:
        return pred_idx, confidence, top_k_list
