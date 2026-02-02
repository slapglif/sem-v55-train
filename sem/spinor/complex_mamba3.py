"""Complex-valued Mamba-3 State Space Model.

Implements a complex-valued selective state space model based on Mamba
architecture principles, extended with:
- Complex state (a+bi, c+di) for rotational dynamics
- MIMO (Multi-Input Multi-Output) for hardware parallelism
- Spinor block-diagonal projections for 100x speedup
- Phase-preserving normalization

The key insight: meaning is encoded in the PHASE of the complex state.
Context changes ROTATE the phase while preserving magnitude (information density).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from .spinor_block import SpinorGate
from ..utils.complex_layernorm import ComplexRMSNorm
from ..utils.fused_complex_linear import FusedComplexLinear
from ..utils.complex_ops import safe_complex


class ComplexSSMState(nn.Module):
    """Complex-valued SSM state update with selective scan.

    Implements: h_t = A * h_{t-1} + B * x_t
                y_t = C * h_t

    where A, B, C are complex-valued and input-dependent (selective).
    A is parameterized in log-polar form with stability guarantee:
        |A| = exp(-softplus(raw_A) * dt_mag), ensuring |A| < 1 always.
    """

    def __init__(self, d_input: int, state_dim: int, mimo_groups: int = 1):
        """
        Args:
            d_input: Input dimension per MIMO group
            state_dim: SSM state dimension N
            mimo_groups: Number of MIMO groups G
        """
        super().__init__()
        self.d_input = d_input
        self.state_dim = state_dim
        self.mimo_groups = mimo_groups

        # A matrix: diagonal, complex, in log-polar form
        # log_magnitude and angle are learnable
        # Raw A magnitude parameter (passed through -softplus for guaranteed stability)
        # After -softplus: effective log_A_mag in [-1.31, -0.47] → |A| in [0.27, 0.63]
        self.log_A_mag = nn.Parameter(
            torch.rand(mimo_groups, state_dim) * 0.5
            + 0.5  # softplus(0.5..1.0) → -[0.97, 1.31]
        )
        self.A_phase = nn.Parameter(
            torch.randn(mimo_groups, state_dim) * 0.1  # Small initial phase
        )

        # B projection: input -> state (complex linear, fused)
        self.B_proj = FusedComplexLinear(d_input, state_dim, bias=False)

        # C projection: state -> output (complex linear, fused)
        self.C_proj = FusedComplexLinear(state_dim, d_input, bias=False)

        # SEOP Fix 11: Decoupled dt for magnitude/phase independence
        # Single dt couples memory horizon τ with rotation speed ω via τ·ω = const.
        # This forces a tradeoff: long memory ↔ slow rotation. Decoupling allows
        # independent control of how long to remember and how fast to rotate.
        # Fused projection: one Linear → [dt_mag, dt_phase] (halves kernel launches)
        self.dt_proj = nn.Linear(d_input * 2, 2, bias=True)
        self.dt_proj.bias.data.fill_(0.0)  # both dt ≈ exp(0) = 1.0

    def forward(self, x: Tensor) -> Tensor:
        """Run complex selective scan.

        Args:
            x: [B, S, G, d_input] complex64 (G = mimo_groups)
        Returns:
            y: [B, S, G, d_input] complex64
        """
        B, S, G, D = x.shape
        N = self.state_dim
        device = x.device

        # SEOP Fix 14: Log-space dt parameterization
        # softplus compresses Gaussian left tail. exp() maps Gaussian→LogNormal,
        # the natural parameterization for positive timescales.
        x_real = torch.cat([x.real, x.imag], dim=-1)  # [B, S, G, 2D]
        dt_both = torch.clamp(
            torch.exp(self.dt_proj(x_real)), min=1e-4, max=2.0
        )  # [B, S, G, 2]
        dt_mag = dt_both[..., :1]  # [B, S, G, 1]
        dt_phase = dt_both[..., 1:]  # [B, S, G, 1]

        # Discretize A with independent magnitude/phase control:
        #   |A_bar| = exp(dt_mag * log_A_mag)   — memory horizon
        #   ∠A_bar = dt_phase * A_phase          — rotation speed
        # Hard-constrain log_A_mag to be negative (ensures |A| < 1 always)
        neg_log_A = -F.softplus(self.log_A_mag.view(1, 1, G, N))  # Always negative
        A_mag = (dt_mag * neg_log_A).exp()  # |A| = exp(dt_mag * neg_log_A) < 1 always
        A_angle = dt_phase * self.A_phase.view(1, 1, G, N)

        # Pre-calculate real-block isomorphism components using cos/sin to avoid complex ops
        A_bar_r = A_mag * torch.cos(A_angle)
        A_bar_i = A_mag * torch.sin(A_angle)

        # Compute B * x (project input to state space) - fused 1 kernel
        Bx = self.B_proj(x)  # [B, S, G, N]
        Bx = Bx * dt_mag.to(Bx.dtype)  # Scale by dt_mag (controls input injection rate)

        # Sequential scan using explicit real-valued recurrence (Real-Block Isomorphism)
        #   h_r = A_bar_r * h_r - A_bar_i * h_i + Bx_r
        #   h_i = A_bar_r * h_i + A_bar_i * h_r + Bx_i
        # Initial state h must be float32 for XPU compatibility.
        h_r = torch.zeros(B, G, N, dtype=torch.float32, device=device)
        h_i = torch.zeros(B, G, N, dtype=torch.float32, device=device)

        Bx_r, Bx_i = Bx.real, Bx.imag
        all_h_r = torch.empty(B, S, G, N, dtype=torch.float32, device=device)
        all_h_i = torch.empty(B, S, G, N, dtype=torch.float32, device=device)

        for t in range(S):
            # Fetch components for time t
            ar, ai = A_bar_r[:, t], A_bar_i[:, t]
            br, bi = Bx_r[:, t], Bx_i[:, t]

            # Real-block update (complex multiply expansion)
            # No torch.complex64 tensors used inside this loop.
            new_h_r = ar * h_r - ai * h_i + br
            new_h_i = ar * h_i + ai * h_r + bi

            h_r, h_i = new_h_r, new_h_i

            # Periodic state normalization to prevent numerical drift
            if (t + 1) % 256 == 0:
                h_norm = torch.sqrt(h_r * h_r + h_i * h_i + 1e-8)
                h_scale = torch.clamp(h_norm, max=100.0) / h_norm
                h_r = h_r * h_scale
                h_i = h_i * h_scale

            all_h_r[:, t], all_h_i[:, t] = h_r, h_i

        # Reconstruct complex state for output projection
        all_h = safe_complex(all_h_r, all_h_i)

        # Apply C projection to all states at once (batched, fused 1 kernel)
        y = self.C_proj(all_h)  # [B, S, G, d_input]

        return y


class ComplexMamba3Layer(nn.Module):
    """Single Complex Mamba-3 layer with spinor gating.

    Architecture:
    1. ComplexRMSNorm
    2. SpinorGate (block-diagonal rotation with selective gating)
    3. ComplexSSMState (complex-valued selective scan)
    4. Residual connection
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 64,
        mimo_groups: int = 8,
        block_size: int = 8,
        d_conv: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mimo_groups = mimo_groups
        self.group_dim = hidden_dim // mimo_groups

        assert hidden_dim % mimo_groups == 0, (
            f"hidden_dim {hidden_dim} must be divisible by mimo_groups {mimo_groups}"
        )

        # Pre-norm
        self.norm = ComplexRMSNorm(hidden_dim)

        # Spinor gate (block-diagonal rotation)
        self.spinor_gate = SpinorGate(hidden_dim, block_size)

        # SEOP Fix 8+10: Fused complex depthwise convolution
        # Single Conv1d with groups=D, 2 input channels (re,im) and 2 output channels
        # per group implements the complex product matrix [[h_r,-h_i],[h_i,h_r]].
        # 1 kernel launch instead of 4, same parameter count (2*D*K).
        # Input layout: interleaved [re_0,im_0,re_1,im_1,...] for memory locality.
        self.conv = nn.Conv1d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=hidden_dim,  # Each group: 2 in → 2 out (complex multiply)
        )
        # Initialize: h_real = variance-preserving (Fix 10), h_imag = zero
        # Conv weight shape: [2D, 2, K] (out_channels, in_channels/groups, kernel)
        # Per group: [[w_rr, w_ri], [w_ir, w_ii]] kernels
        # For complex conv at init: w_rr=w_ii=h_real, w_ri=-h_imag=0, w_ir=h_imag=0
        with torch.no_grad():
            self.conv.weight.zero_()
            # Set diagonal blocks (rr and ii) to variance-preserving init
            for d in range(hidden_dim):
                # out_channel 2*d (real out), in_channel 0 (real in) = h_real
                self.conv.weight[2 * d, 0, :].normal_(std=1.0 / math.sqrt(d_conv))
                # out_channel 2*d+1 (imag out), in_channel 1 (imag in) = h_real (same)
                self.conv.weight[2 * d + 1, 1, :] = self.conv.weight[2 * d, 0, :]
            # Off-diagonal blocks (ri, ir) start at zero → pure real conv at init
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # Complex SSM
        self.ssm = ComplexSSMState(
            d_input=self.group_dim, state_dim=state_dim, mimo_groups=mimo_groups
        )

        # Output projection (complex, fused)
        self.out_proj = FusedComplexLinear(hidden_dim, hidden_dim, bias=False)

        # Learnable magnitude gate threshold (SEOP Fix 12: χ²(2)-CDF matched)
        # For complex Gaussian z, |z|² ~ Exponential(1/2σ²). Using CDF gate
        # 1-exp(-|z|²·β) instead of sigmoid gives uniformly distributed gate values,
        # maximizing gradient information. Initialize β=exp(0)=1 for unit-variance match.
        self.activation_threshold = nn.Parameter(torch.tensor(0.0))

        # SEOP Fix 17: Per-dimension SSM output scaling
        # Scalar scale applies uniform compensation, but different state dimensions
        # decay at different rates (|A|^S varies per dim). Per-dimension scaling lets
        # each channel learn its own attenuation compensation.
        self.ssm_output_scale = nn.Parameter(torch.full((hidden_dim,), 2.5))

        # SEOP Fix 22: Learnable residual scaling initialized to 1/√(2L)
        # Without scaling, variance grows as 1 + L·σ² across L layers.
        # DeepSeek/GPT-2 style: scale residual additions to maintain unit variance.
        self.residual_scale = nn.Parameter(
            torch.tensor(1.0 / math.sqrt(2.0 * num_layers))
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Complex Mamba-3 layer.

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64 (with residual)
        """
        B, S, D = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Spinor gated rotation
        x = self.spinor_gate(x)

        # SEOP Fix 8: Fused complex depthwise conv (1 kernel launch, not 4)
        # Interleave real/imag: [B, S, D] complex → [B, 2D, S] interleaved
        x_interleaved = torch.stack([x.real, x.imag], dim=-1)  # [B, S, D, 2]
        x_interleaved = x_interleaved.reshape(B, S, 2 * D).transpose(1, 2)  # [B, 2D, S]
        conv_out = self.conv(x_interleaved)[:, :, :S]  # [B, 2D, S]
        conv_out = conv_out.transpose(1, 2).reshape(B, S, D, 2)  # [B, S, D, 2]
        x = safe_complex(conv_out[..., 0], conv_out[..., 1])  # [B, S, D]

        # SEOP Fix 12: χ²(2)-CDF magnitude gate
        # For complex Gaussian z, |z|² ~ Exp(1/2σ²). The CDF 1-exp(-|z|²·β)
        # gives uniformly distributed gate values, maximizing gradient information.
        mag_sq = x.real**2 + x.imag**2  # |z|^2
        gate = 1.0 - torch.exp(-mag_sq * self.activation_threshold.exp())
        x = x * gate  # phase preserved, χ²-matched gate

        # Reshape for MIMO SSM
        x_groups = x.view(B, S, self.mimo_groups, self.group_dim)

        # Complex SSM scan
        y_groups = self.ssm(x_groups)  # [B, S, G, group_dim]
        y = y_groups.reshape(B, S, D)  # [B, S, D]

        # Output projection (complex linear, fused) - 1 kernel instead of 4
        out = self.out_proj(y)

        # SEOP Fix 5: Scale SSM output to match residual magnitude
        out = out * self.ssm_output_scale

        # Residual with learnable scaling (SEOP Fix 22)
        return residual + self.residual_scale * out
