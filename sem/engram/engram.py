"""Engram - N-gram hash-based embedding augmentation module.

Augments hidden states with n-gram context via hash-based embeddings.
Uses conv-gated mixing to selectively incorporate n-gram information.

Architecture:
1. Generate n-gram hash IDs from input tokens
2. Lookup multi-head embeddings for each hash
3. Flatten embeddings across n-gram sizes and heads
4. Conv-gated mixing: learned per-channel gates + short convolution
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .hash_mapping import NgramHashMapping
from .multi_head_embedding import MultiHeadEmbedding
from ..utils.complex_ops import safe_complex


@dataclass
class EngramConfig:
    """Engram configuration.

    Attributes:
        tokenizer: Tokenizer instance (required)
        engram_vocab_size: Base vocab sizes for each n-gram size [bigram_vocab, trigram_vocab, ...]
        max_ngram_size: Maximum n-gram size (2 = bigrams only, 3 = bigrams + trigrams)
        n_embed_per_ngram: Embedding dimension per n-gram (divided among heads)
        n_head_per_ngram: Number of hash heads per n-gram size
        layer_ids: Layer IDs where Engram is inserted
        pad_id: Padding token ID
        seed: Random seed for hash multipliers
        kernel_size: Convolution kernel size
        hc_mult: Hierarchical capacity multiplier (number of gated channels)
    """

    tokenizer: Any = None
    engram_vocab_size: list[int] = field(
        default_factory=lambda: [129280 * 5, 129280 * 5]
    )
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: list[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    hc_mult: int = 4  # Hierarchical capacity multiplier


class ShortConv(nn.Module):
    """Short convolution for local smoothing.

    Applies grouped 1D convolution across time dimension with RMSNorm
    and optional SiLU activation.

    Attributes:
        hc_mult: Number of channels (hierarchical capacity multiplier)
        activation: Whether to apply SiLU activation
        conv: Grouped 1D convolution
        norms: RMSNorm for each channel
        act_fn: SiLU activation function
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        """Initialize short convolution.

        Args:
            hidden_size: Hidden dimension
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            norm_eps: RMSNorm epsilon
            hc_mult: Number of channels
            activation: Whether to apply SiLU activation
        """
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult

        # Grouped convolution (each channel processed separately)
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        # RMSNorm for each channel
        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)]
        )

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply short convolution.

        Args:
            x: [B, T, hc_mult, D] input tensor

        Returns:
            output: [B, T, hc_mult, D] convolved tensor
        """
        B, T, G, C = x.shape
        assert G == self.hc_mult, f"Expected {self.hc_mult} channels, got {G}"

        # Apply RMSNorm to each channel separately
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]  # [B, T, C]
            normed_chunks.append(self.norms[i](chunk))

        # Concatenate channels: [B, T, G*C]
        x_norm = torch.cat(normed_chunks, dim=-1)

        # Transpose to [B, G*C, T] for Conv1d
        x_bct = x_norm.transpose(1, 2)

        # Apply convolution
        y_bct = self.conv(x_bct)

        # Truncate padding
        y_bct = y_bct[..., :T]

        # Apply activation
        if self.activation:
            y_bct = self.act_fn(y_bct)

        # Transpose back to [B, T, G*C] and reshape to [B, T, G, C]
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

        return y


class Engram(nn.Module):
    """Engram module - n-gram hash-based embedding augmentation.

    Augments hidden states with n-gram context via:
    1. N-gram hash embeddings (multi-head, multi-layer)
    2. Conv-gated mixing (learned per-channel scalar gates)
    3. Short convolution for local smoothing

    Attributes:
        layer_id: Layer ID for this Engram instance
        hidden_size: Hidden dimension
        hc_mult: Hierarchical capacity multiplier
        hash_mapping: N-gram hash mapping generator
        multi_head_embedding: Multi-head embedding lookup
        short_conv: Short convolution module
        value_proj: Linear projection for n-gram embeddings
        gate_logits: Learned per-channel scalar gates (logits)
    """

    # [SEOP] Cross-layer CPU hash caching.
    # Hashing runs in NumPy on CPU; if repeated per Engram call it forces a GPU->CPU sync
    # and stalls the forward pass (entropy transfer collapses to a host roundtrip).
    # We keep a single-entry cache shared across Engram instances so multiple Engram
    # layers in the same model forward can reuse the same CPU hash results.
    _global_hash_cache_key: Optional[tuple[Any, ...]] = None
    _global_hash_cache_val: Optional[dict[int, Tensor]] = None

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        config: EngramConfig,
    ):
        """Initialize Engram module.

        Args:
            layer_id: Layer ID for this instance
            hidden_size: Hidden dimension (must match encoder output)
            config: EngramConfig instance
        """
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hc_mult = config.hc_mult

        # [SEOP] Phase-preserving complex injection is default-on.
        # Engram itself produces real-valued augmentation; downstream integration should
        # modulate complex magnitude without breaking global-phase symmetry.
        self.phase_preserving = True

        if config.tokenizer is None:
            raise ValueError("EngramConfig.tokenizer must be provided")

        # Hash mapping (shared across all layers for efficiency)
        # In SEM V8.0, we'll create one instance and share it
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=config.engram_vocab_size,
            max_ngram_size=config.max_ngram_size,
            n_embed_per_ngram=config.n_embed_per_ngram,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.layer_ids,
            tokenizer=config.tokenizer,
            pad_id=config.pad_id,
            seed=config.seed,
        )

        # Multi-head embedding
        # Flatten all n-gram sizes and heads into single list
        list_of_N = [
            x
            for y in self.hash_mapping.vocab_size_across_layers[self.layer_id]
            for x in y
        ]
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=config.n_embed_per_ngram // config.n_head_per_ngram,
        )

        # Short convolution
        self.short_conv = ShortConv(
            hidden_size=hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
            hc_mult=config.hc_mult,
        )

        # Projection layers
        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)

        # Learned per-channel gates (replaces similarity-based gating)
        # Initialize at 0.0 so sigmoid gives 0.5 (moderate mixing)
        self.gate_logits = nn.Parameter(torch.zeros(config.hc_mult))

        # [SEOP] Cache hash results to avoid repeated GPU->CPU sync.
        # Per-instance cache holds the device-resident hash IDs for this layer.
        # Keyed by tensor identity + pointer + shape to avoid recomputing when the same
        # input_ids are reused (e.g., across multiple Engram layers / recomputation).
        self._hash_cache_key: Optional[tuple[Any, ...]] = None
        self._hash_cache_val: Optional[Tensor] = None

    def inject_into_complex(self, psi: Tensor, engram_out: Tensor) -> Tensor:
        """Inject Engram output into a complex state.

        Phase-preserving injection modulates magnitude without breaking complex phase.

        Notes:
            - [SEOP] Preserving global-phase symmetry avoids injecting information into
              only one quadrature (real) which can create systematic phase bias.
            - The legacy behavior (real-only injection) is kept for backward compatibility
              when `self.phase_preserving` is disabled.

        Args:
            psi: Complex tensor state, e.g. [B, L, D] complex.
            engram_out: Real-valued augmentation, broadcastable to psi.real.

        Returns:
            Complex tensor with Engram injection applied.
        """
        if not self.phase_preserving:
            # Legacy: add to real part only (breaks Re/Im symmetry).
            return safe_complex(psi.real + engram_out, psi.imag)

        # Phase-preserving: scale magnitude along psi direction.
        # psi_out = psi * (1 + engram_out / (|psi| + eps))
        mag = psi.abs().clamp(min=1e-8)
        scale = 1.0 + engram_out / mag
        return psi * scale

    def forward(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        """Apply Engram augmentation to hidden states.

        Args:
            hidden_states: [B, L, D] encoder output (REAL tensor, not complex)
            input_ids: [B, L] token IDs

        Notes:
            - [SEOP] `hidden_states` is used for device placement only.
              Engram computation is conditioned solely on `input_ids` via hash-based
              n-gram addressing (it is intentionally unconditional on activations).

        Returns:
            output: [B, L, D] augmented hidden states (REAL tensor)
        """
        B, L = input_ids.shape

        # Generate n-gram hash IDs
        # hash_input_ids: [B, L, num_heads_total]
        # [SEOP] Avoid repeated NumPy hashing + device sync.
        # Hashing requires `input_ids.cpu().numpy()` which can force a GPU->CPU sync.
        # We cache the *CPU* hash results globally (shared across Engram instances) and
        # cache the *device* tensor per instance.
        cache_key = (id(input_ids), input_ids.data_ptr(), input_ids.shape)
        if Engram._global_hash_cache_key != cache_key:
            # NOTE: NgramHashMapping.hash() is expected to return a dict keyed by layer_id.
            hash_out = self.hash_mapping.hash(input_ids.cpu().numpy())
            Engram._global_hash_cache_val = {
                k: torch.from_numpy(v) for k, v in hash_out.items()
            }
            Engram._global_hash_cache_key = cache_key

        local_cache_key = (cache_key, hidden_states.device)
        if self._hash_cache_key != local_cache_key:
            cached = Engram._global_hash_cache_val
            assert cached is not None
            cpu_hash_ids = cached[self.layer_id]
            self._hash_cache_val = cpu_hash_ids.to(hidden_states.device)
            self._hash_cache_key = local_cache_key

        hash_input_ids = self._hash_cache_val
        assert hash_input_ids is not None

        # Lookup embeddings and flatten across heads
        # embeddings: [B, L, num_heads_total, D_per_head]
        # -> flatten to [B, L, engram_hidden_size]
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        # Project n-gram embeddings to hidden size
        value = self.value_proj(embeddings)  # [B, L, D]

        # Expand to hc_mult channels with learned gates
        # gate_logits: [hc_mult] learned parameters, sigmoid gives (0,1) range
        # [SEOP] Scalar per-channel gates are intentionally non-adaptive (position-independent).
        # Per-token gates would require computing attention/MLP over hash embeddings, adding
        # O(T*D) cost and extra memory traffic; the scalar gate is a coarse, fast on/off switch.
        gates = torch.softmax(self.gate_logits, dim=0)  # [hc_mult], sum=1

        # Expand value to [B, L, hc_mult, D] and apply gates
        value_expanded = value.unsqueeze(2).expand(
            -1, -1, self.hc_mult, -1
        )  # [B, L, hc_mult, D]
        gated_value = value_expanded * gates.view(
            1, 1, self.hc_mult, 1
        )  # [B, L, hc_mult, D]

        # Short convolution for local smoothing
        output = gated_value + self.short_conv(gated_value)  # [B, L, hc_mult, D]

        # Reduce hc_mult dimension
        return output.sum(dim=2)  # [B, L, D]
