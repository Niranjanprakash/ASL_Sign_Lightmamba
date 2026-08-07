import torch
import torch.nn as nn
from backend.config import MAMBA_HIDDEN_DIM, FUSION_DIM, DROPOUT
from backend.models.mamba_block import MambaBlock

class HMSMamba(nn.Module):
    """
    Hierarchical Multi-Scale Mamba (HMS-Mamba) temporal model.
    Captures temporal representations at three scales:
      - Scale 1 (Fine): T = 32
      - Scale 2 (Intermediate): T = 16
      - Scale 3 (Global): T = 8
    """
    def __init__(self, in_dim: int = FUSION_DIM, hidden_dim: int = MAMBA_HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        
        # Level 1 (Fine) Processing
        self.fine_mamba = MambaBlock(d_model=in_dim)
        
        # Downsampling layers: sequence length reduction
        # T=32 -> T=16
        self.downsample_1_to_2 = nn.Conv1d(
            in_channels=in_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        # Level 2 (Intermediate) Processing
        self.intermediate_mamba = MambaBlock(d_model=hidden_dim)
        
        # Downsampling layers: sequence length reduction
        # T=16 -> T=8
        self.downsample_2_to_3 = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        # Level 3 (Global) Processing
        self.global_mamba = MambaBlock(d_model=hidden_dim)
        
        # Pooling layers to capture sequence-level representations
        self.pool_fine = nn.AdaptiveAvgPool1d(1)
        self.pool_inter = nn.AdaptiveAvgPool1d(1)
        self.pool_global = nn.AdaptiveAvgPool1d(1)
        
        # Multi-scale Fusion
        # Concatenates: Fine (in_dim) + Intermediate (hidden_dim) + Global (hidden_dim)
        concat_dim = in_dim + hidden_dim + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [B, T=32, FUSION_DIM]
        Returns: [B, MAMBA_HIDDEN_DIM]
        """
        B, T, D = x.shape
        
        # 1. Level 1 (Fine Scale)
        fine_feats = self.fine_mamba(x) # [B, 32, FUSION_DIM]
        fine_rep = self.pool_fine(fine_feats.transpose(1, 2)).squeeze(-1) # [B, FUSION_DIM]
        
        # 2. Downsample: 32 -> 16
        x_conv1 = fine_feats.transpose(1, 2) # [B, FUSION_DIM, 32]
        x_down1 = self.downsample_1_to_2(x_conv1).transpose(1, 2) # [B, 16, MAMBA_HIDDEN_DIM]
        
        # 3. Level 2 (Intermediate Scale)
        inter_feats = self.intermediate_mamba(x_down1) # [B, 16, MAMBA_HIDDEN_DIM]
        inter_rep = self.pool_inter(inter_feats.transpose(1, 2)).squeeze(-1) # [B, MAMBA_HIDDEN_DIM]
        
        # 4. Downsample: 16 -> 8
        x_conv2 = inter_feats.transpose(1, 2) # [B, MAMBA_HIDDEN_DIM, 16]
        x_down2 = self.downsample_2_to_3(x_conv2).transpose(1, 2) # [B, 8, MAMBA_HIDDEN_DIM]
        
        # 5. Level 3 (Global Scale)
        global_feats = self.global_mamba(x_down2) # [B, 8, MAMBA_HIDDEN_DIM]
        global_rep = self.pool_global(global_feats.transpose(1, 2)).squeeze(-1) # [B, MAMBA_HIDDEN_DIM]
        
        # 6. Multi-Scale Fusion
        combined = torch.cat([fine_rep, inter_rep, global_rep], dim=-1) # [B, FUSION_DIM + MAMBA_HIDDEN_DIM * 2]
        video_level_representation = self.fusion(combined) # [B, MAMBA_HIDDEN_DIM]
        
        return video_level_representation
