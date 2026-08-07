import torch
import torch.nn as nn
from backend.utils import logger

# Try native Mamba import
NATIVE_MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba as NativeMamba
    NATIVE_MAMBA_AVAILABLE = True
    logger.info("[MAMBA] Native mamba_ssm imported successfully.")
except ImportError as e:
    logger.warning(
        f"[MAMBA] Native mamba_ssm could not be imported ({e}). "
        "This is common on Windows due to CUDA compilation requirements. "
        "Using PyTorch-based custom MambaBlock FALLBACK."
    )

class MambaBlockFallback(nn.Module):
    """
    A PyTorch-only fallback mimicking a simplified selective SSM block.
    Acts as a Drop-in replacement for NativeMamba.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        
        # Projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM parameters projections
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # SSM states initialization
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [B, L, D]
        Returns: [B, L, D]
        """
        B, L, D = x.shape
        
        # Project inputs
        projected = self.in_proj(x) # [B, L, d_inner * 2]
        x_branch, res_branch = torch.chunk(projected, 2, dim=-1)
        
        # Conv branch
        x_conv = x_branch.transpose(1, 2) # [B, d_inner, L]
        # Padding & Conv
        x_conv = self.conv(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2) # [B, L, d_inner]
        
        # Non-linearity
        x_conv = torch.nn.functional.silu(x_conv)
        
        # SSM parameters prediction
        ssm_params = self.x_proj(x_conv) # [B, L, d_state * 2 + d_inner]
        dt, B_matrix, C_matrix = torch.split(ssm_params, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        dt = torch.nn.functional.softplus(self.dt_proj(dt)) # [B, L, d_inner]
        
        # Selective Scan simulation (Recurrent loop for fallback simplicity and numerical stability)
        A = -torch.exp(self.A_log) # [d_inner, d_state]
        
        # Run scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y = []
        for t in range(L):
            # dt_t: [B, d_inner], B_t: [B, d_state], C_t: [B, d_state], x_t: [B, d_inner]
            dt_t = dt[:, t, :].unsqueeze(-1) # [B, d_inner, 1]
            B_t = B_matrix[:, t, :].unsqueeze(1) # [B, 1, d_state]
            C_t = C_matrix[:, t, :].unsqueeze(-1) # [B, d_state, 1]
            x_t = x_conv[:, t, :].unsqueeze(-1) # [B, d_inner, 1]
            
            # Discretization
            dA = torch.exp(dt_t * A.unsqueeze(0)) # [B, d_inner, d_state]
            dB = dt_t * B_t # [B, d_inner, d_state]
            
            # Update hidden state
            h = dA * h + dB * x_t # [B, d_inner, d_state]
            
            # Compute output
            y_t = torch.matmul(h, C_t).squeeze(-1) # [B, d_inner]
            y.append(y_t)
            
        y = torch.stack(y, dim=1) # [B, L, d_inner]
        
        # Multiply by D skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gated output
        gated = y * torch.nn.functional.silu(res_branch)
        
        # Out projection
        out = self.out_proj(gated)
        return out

class MambaBlock(nn.Module):
    """
    Abstractions wrapper matching native or fallback block.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.native = NATIVE_MAMBA_AVAILABLE
        
        if self.native:
            self.block = NativeMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.block = MambaBlockFallback(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
