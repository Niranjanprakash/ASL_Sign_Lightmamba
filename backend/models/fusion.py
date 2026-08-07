import torch
import torch.nn as nn
from backend.config import RGB_FEATURE_DIM, LANDMARK_EMBED_DIM, FUSION_DIM, DROPOUT, USE_RELIABILITY_FUSION

class MultimodalFusion(nn.Module):
    """
    Fuses RGB spatial features with landmark embeddings.
    Supports both baseline Concatenation+Projection and Reliability-Aware Gated Fusion.
    """
    def __init__(self, rgb_dim: int = RGB_FEATURE_DIM, landmark_dim: int = LANDMARK_EMBED_DIM, fusion_dim: int = FUSION_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.use_reliability = USE_RELIABILITY_FUSION

        # Projection layers to align dimensions
        self.rgb_project = nn.Linear(rgb_dim, fusion_dim)
        self.landmark_project = nn.Linear(landmark_dim, fusion_dim)

        if self.use_reliability:
            # Gating networks to compute dynamic weights (reliability scores)
            self.rgb_gate = nn.Sequential(
                nn.Linear(rgb_dim, 1),
                nn.Sigmoid()
            )
            # Mask (validity mask info) can also be used here
            self.landmark_gate = nn.Sequential(
                nn.Linear(landmark_dim + 3, 1), # plus 3 dims for left hand, right hand, pose masks
                nn.Sigmoid()
            )
            self.post_gate_project = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            # Baseline: Concatenate projected features and project again
            self.concat_project = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

    def forward(self, rgb_feats: torch.Tensor, landmark_emb: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        rgb_feats: [B, T, RGB_DIM]
        landmark_emb: [B, T, LANDMARK_EMBED_DIM]
        mask: [B, T, 3] (landmark validity mask)
        Returns: [B, T, FUSION_DIM]
        """
        # Project both to fusion_dim
        p_rgb = self.rgb_project(rgb_feats)          # [B, T, FUSION_DIM]
        p_land = self.landmark_project(landmark_emb)  # [B, T, FUSION_DIM]

        if self.use_reliability:
            # Compute gates
            alpha = self.rgb_gate(rgb_feats) # [B, T, 1]
            
            if mask is not None:
                # Concatenate landmark embeddings and mask
                gate_inputs = torch.cat([landmark_emb, mask], dim=-1)
            else:
                gate_inputs = torch.cat([landmark_emb, torch.ones(landmark_emb.shape[0], landmark_emb.shape[1], 3, device=landmark_emb.device)], dim=-1)
                
            beta = self.landmark_gate(gate_inputs) # [B, T, 1]
            
            # Gated fusion
            fused = alpha * p_rgb + beta * p_land # [B, T, FUSION_DIM]
            fused = self.post_gate_project(fused)
        else:
            # Concatenation fusion
            concated = torch.cat([p_rgb, p_land], dim=-1) # [B, T, FUSION_DIM * 2]
            fused = self.concat_project(concated)         # [B, T, FUSION_DIM]

        return fused
