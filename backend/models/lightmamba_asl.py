import torch
import torch.nn as nn
from backend.config import NUM_CLASSES, MAMBA_HIDDEN_DIM, DROPOUT
from backend.models.mobilenet_branch import MobileNetBranch
from backend.models.landmark_branch import LandmarkBranch
from backend.models.fusion import MultimodalFusion
from backend.models.hms_mamba import HMSMamba

class LightMambaASL(nn.Module):
    """
    End-to-End LightMamba-ASL Multimodal Video Model.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        
        # Branches
        self.rgb_branch = MobileNetBranch(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.landmark_branch = LandmarkBranch()
        
        # Fusion
        self.fusion = MultimodalFusion()
        
        # HMS-Mamba Sequence Model
        self.temporal_model = HMSMamba()
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(MAMBA_HIDDEN_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(MAMBA_HIDDEN_DIM, num_classes)
        )

    def forward(self, rgb_input: torch.Tensor, landmark_input: torch.Tensor, mask: torch.Tensor = None, preextracted_rgb: bool = False) -> torch.Tensor:
        """
        rgb_input shape: [B, T, C, H, W] (if preextracted_rgb=False) OR [B, T, 576] (if preextracted_rgb=True)
        landmark_input shape: [B, T, LANDMARK_DIM]
        mask shape: [B, T, 3]
        """
        # 1. RGB Branch
        rgb_feats = self.rgb_branch(rgb_input, preextracted=preextracted_rgb) # [B, T, 576]
        
        # 2. Landmark Branch
        land_emb = self.landmark_branch(landmark_input) # [B, T, 256]
        
        # 3. Fusion
        fused = self.fusion(rgb_feats, land_emb, mask) # [B, T, 256]
        
        # 4. HMS-Mamba Sequence modeling
        temporal_rep = self.temporal_model(fused) # [B, 256]
        
        # 5. Classifier
        logits = self.classifier(temporal_rep) # [B, num_classes]
        
        return logits
