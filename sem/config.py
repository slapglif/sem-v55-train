"""Configuration dataclasses for SEM V5.5."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 8
    vocab_size: int = 50262
    max_seq_length: int = 2048
    model_version: str = "v55"


@dataclass
class EncoderConfig:
    sdr_sparsity: int = 16
    sdr_candidates: int = 128
    sinkhorn_epsilon: float = 0.07
    sinkhorn_max_iter: int = 90
    sinkhorn_tol: float = 1e-3
    sinkhorn_auto_epsilon: bool = True  # Scale ε to median(cost) at runtime
    sinkhorn_auto_epsilon_scale: float = 0.04  # ε = scale * median(cost)
    soft_sparse: bool = (
        True  # SEOP Fix 48: Enable gradient flow to all codebook entries
    )
    soft_sparse_temp: float = 0.1  # Temperature for soft-sparse softmax weighting
    simple_mode: bool = False  # SEOP Fix 52: Bypass MESH-SDR, keep Re(z) in embedding space for weight tying


@dataclass
class SpinorConfig:
    block_size: int = 8
    num_blocks: int = 32
    state_dim: int = 64
    mimo_groups: int = 8
    d_conv: int = 4
    memory_horizon_ratio: float = (
        0.0  # τ = ratio * max_seq_length; 0 = use default init (-4.55 ≈ S/e for S=256)
    )


@dataclass
class PropagatorConfig:
    cayley_dt: float = 0.02
    num_layers: int = 1  # SEOP Fix: Decoupled from model depth (was 8). 1 layer is sufficient for diffusion.
    cg_max_iter: int = 5  # 5 iterations sufficient for training; use 20 for inference
    cg_tol: float = 1e-6
    nonlinear_alpha: float = 0.12
    laplacian_sparsity: int = 6
    lazy_cg: bool = True
    lazy_cg_tol: float = 1e-6  # Residual gate tolerance for lazy CG
    direct_solve: bool = False
    pit_gamma: float = 0.05  # Soliton envelope width (SEOP Fix 29)
    adaptive_cg_tol: bool = False
    cg_tol_warmup: float = 1e-4  # Loose tolerance during warmup
    cg_tol_mid: float = 1e-5  # Mid-training tolerance
    cg_tol_late: float = 1e-6  # Tight tolerance for convergence
    cg_tol_warmup_end: int = 2000  # Step to switch warmup → mid
    cg_tol_mid_end: int = 50000  # Step to switch mid → late
    use_chebyshev_kpm: bool = (
        True  # Use KPM Chebyshev expansion instead of CG/direct solve
    )
    chebyshev_degree: int = 16  # Number of Chebyshev polynomial terms for KPM


@dataclass
class QuantizerConfig:
    codebook_size: int = 256
    group_size: int = 2
    fisher_ema_decay: float = 0.99
    outlier_percentile: float = 0.01
    dead_code_threshold: int = 100


@dataclass
class SamplerConfig:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.92

    # Modern sampling methods (SEOP Fix 59: composable LogitsProcessor chain)
    min_p: float = 0.0  # Discard tokens with P < min_p * P_max. 0.0 = disabled
    typical_p: float = 0.92  # Typical sampling: keep tokens near expected information content. 1.0 = disabled
    repetition_penalty: float = (
        1.43  # Multiplicative penalty for tokens already in context. 1.0 = disabled
    )
    frequency_penalty: float = (
        0.0  # Additive penalty proportional to token count in context. 0.0 = disabled
    )
    presence_penalty: float = (
        0.0  # Additive one-time penalty for tokens present in context. 0.0 = disabled
    )
    no_repeat_ngram_size: int = 0  # Ban repeating n-grams of this size. 0 = disabled
    top_a: float = (
        0.0  # Top-A sampling: keep tokens with P >= top_a * P_max². 0.0 = disabled
    )
    epsilon_cutoff: float = (
        0.0  # Truncation: discard tokens with P < epsilon. 0.0 = disabled
    )
    eta_cutoff: float = 0.0  # Eta sampling: entropy-adaptive cutoff. 0.0 = disabled
    temperature_last: bool = (
        False  # Apply temperature AFTER filtering (llama.cpp style). Default: before
    )
    sampler_seed: Optional[int] = (
        None  # Fixed seed for reproducible sampling. None = random
    )


@dataclass
class TrainingConfig:
    # Existing fields
    batch_size: int = 32
    learning_rate: float = 8.6e-4
    weight_decay: float = 0.002
    encoder_lr_scale: float = (
        0.016  # SEOP Fix 41: Encoder LR = base_lr * this (balance gradient flow)
    )
    warmup_steps: int = 2000  # Changed from 1000
    max_steps: int = 100000
    gradient_clip: float = 17.0  # SEOP Fix 35: raised from 1.0 — too aggressive, clips useful NLL gradients
    dtype: str = "complex64"

    # New fields
    micro_batch_size: int = 4
    gradient_checkpointing: bool = (
        False  # Default disabled - causes tensor count mismatch bug
    )
    unitary_lambda: float = (
        0.006  # SEOP Fix 35: reduced from 0.1 — Cayley enforces unitarity structurally
    )
    unitary_clamp_min: float = (
        0.01  # SEOP Fix 35: lower bound for psi energy norm clamping
    )
    unitary_clamp_max: float = (
        6.9  # SEOP Fix 35: upper bound for psi energy norm clamping
    )
    low_vram_mode: bool = False
    born_chunk_size: int = 2048

    # Logging
    log_interval: int = 10
    health_check_interval: int = 100
    checkpoint_interval: int = 5000
    keep_checkpoints: int = 3

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    tokenizer_path: str = "tokenizer/"
    num_workers: int = 0
    shuffle_buffer_size: int = 1000

    # WSD Scheduler
    scheduler_type: str = "wsd"
    stable_steps: int = 0
    decay_steps: int = 5000
    lr_min_ratio: float = 0.1

    # wandb
    wandb_project: str = "sem-v55-lean-crystal"
    wandb_enabled: bool = True

    # Timing and profiling
    timing_enabled: bool = False
    timing_log_interval: int = 10

    no_compile: bool = False
    compile_mode: str = "default"
    no_amp: bool = False

    # SEOP Fix 56: Label smoothing for noisy small-batch training
    label_smoothing: float = 0.058


@dataclass
class CurriculumConfig:
    enabled: bool = True
    stages: list = field(
        default_factory=lambda: [
            {"min_score": 2, "seq_len": 512, "min_steps": 20000},
            {"min_score": 3, "seq_len": 1024, "min_steps": 30000},
            {"min_score": 3, "seq_len": 2048, "min_steps": 50000},
        ]
    )
    transition_check_interval: int = 500
    loss_plateau_threshold: float = 0.01
    loss_plateau_window: int = 1000
    lr_decay_per_stage: float = 0.7
    stage_warmup_steps: int = 500
    unitary_stability_threshold: float = 0.05


@dataclass
class DistillationConfig:
    enabled: bool = True
    alpha: float = 0.7
    ema_decay_start: float = 0.999
    ema_decay_end: float = 0.9999
    ema_decay_ramp_steps: int = 10000
    enable_at_stage: int = 2
    temperature: float = 2.0


@dataclass
class V8Config:
    """V8.0 model-specific configuration."""

    use_lindblad: bool = True
    use_hybrid_automata: bool = False
    use_quaternionic: bool = True
    use_mhc: bool = True
    mhc_streams: int = 8
    mhc_num_iters: int = 10
    mhc_tau: float = 0.031
    lindblad_gamma: float = 0.007
    num_lindblad_ops: int = 4
    curvature_threshold: float = 0.64
    condition_threshold: float = 62.0


@dataclass
class SEMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    spinor: SpinorConfig = field(default_factory=SpinorConfig)
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    v8: V8Config = field(default_factory=V8Config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SEMConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        return config
