import math

import pytest
import torch

from sem.config import (
    SEMConfig,
    ModelConfig,
    EncoderConfig,
    SpinorConfig,
    PropagatorConfig,
    SamplerConfig,
    V8Config,
)
from sem.encoder.mesh_sdr import MESHEncoder
from sem.spinor.complex_mamba3 import ComplexMamba3Layer
from sem.spinor.lindblad import LindbladDissipation
from sem.spinor.hybrid_automata import HybridAutomata
from sem.spinor.quaternion import QuaternionicEscape
from sem.propagator.cayley_soliton import CayleySolitonPropagator
from sem.sampler.born_collapse import BornCollapseSampler
from sem.hyper_connections.mhc import mhc_residual
from sem.hyper_connections.sinkhorn import sinkhorn_log_complex
from sem.model import ComplexMamba3LayerV8


def make_small_config():
    return SEMConfig(
        model=ModelConfig(
            hidden_dim=64, num_layers=2, vocab_size=256, max_seq_length=32
        ),
        encoder=EncoderConfig(sdr_sparsity=8, sdr_candidates=32, soft_sparse=False),
        spinor=SpinorConfig(
            block_size=8, num_blocks=8, state_dim=16, mimo_groups=4, d_conv=4
        ),
        propagator=PropagatorConfig(cg_max_iter=10, laplacian_sparsity=3),
        sampler=SamplerConfig(temperature=1.0, top_k=0, top_p=1.0),
        v8=V8Config(
            use_lindblad=True,
            use_hybrid_automata=True,
            use_quaternionic=True,
            use_mhc=True,
        ),
    )


class TestMESHEncoderQuality:
    """Quality tests for MESHEncoder."""

    def test_output_is_complex(self):
        torch.manual_seed(42)
        config = make_small_config()
        encoder = MESHEncoder(
            vocab_size=config.model.vocab_size,
            hidden_dim=config.model.hidden_dim,
            sdr_sparsity=config.encoder.sdr_sparsity,
            sdr_candidates=config.encoder.sdr_candidates,
            sinkhorn_epsilon=config.encoder.sinkhorn_epsilon,
            sinkhorn_max_iter=config.encoder.sinkhorn_max_iter,
            sinkhorn_tol=config.encoder.sinkhorn_tol,
            max_seq_length=config.model.max_seq_length,
            soft_sparse=config.encoder.soft_sparse,
        )
        tokens = torch.randint(0, config.model.vocab_size, (2, 8))
        z = encoder(tokens)
        assert z.is_complex()
        assert z.dtype == torch.complex64

    def test_sdr_sparsity(self):
        torch.manual_seed(42)
        config = make_small_config()
        encoder = MESHEncoder(
            vocab_size=config.model.vocab_size,
            hidden_dim=config.model.hidden_dim,
            sdr_sparsity=config.encoder.sdr_sparsity,
            sdr_candidates=config.encoder.sdr_candidates,
            sinkhorn_epsilon=config.encoder.sinkhorn_epsilon,
            sinkhorn_max_iter=config.encoder.sinkhorn_max_iter,
            sinkhorn_tol=config.encoder.sinkhorn_tol,
            max_seq_length=config.model.max_seq_length,
            soft_sparse=False,
        )
        tokens = torch.randint(0, config.model.vocab_size, (2, 6))
        _ = encoder(tokens)
        sparse = encoder._last_sdr_sparse
        active = (sparse.abs() > 1e-6).sum(dim=-1).float().mean().item()
        expected = config.encoder.sdr_sparsity
        assert abs(active - expected) <= max(1.0, 0.2 * expected)

    def test_sinkhorn_convergence(self):
        torch.manual_seed(42)
        config = make_small_config()
        encoder = MESHEncoder(
            vocab_size=config.model.vocab_size,
            hidden_dim=config.model.hidden_dim,
            sdr_sparsity=config.encoder.sdr_sparsity,
            sdr_candidates=config.encoder.sdr_candidates,
            sinkhorn_epsilon=config.encoder.sinkhorn_epsilon,
            sinkhorn_max_iter=config.encoder.sinkhorn_max_iter,
            sinkhorn_tol=config.encoder.sinkhorn_tol,
            max_seq_length=config.model.max_seq_length,
            soft_sparse=config.encoder.soft_sparse,
        )
        tokens = torch.randint(0, config.model.vocab_size, (1, 4))
        embeddings = encoder.embedding(tokens)
        cost = encoder.cost(embeddings)
        transport = encoder.sinkhorn(cost)
        row_sums = transport.sum(dim=-1)
        col_sums = transport.sum(dim=-2)
        expected_row = torch.full_like(row_sums, 1.0 / transport.shape[-2])
        expected_col = torch.full_like(col_sums, 1.0 / transport.shape[-1])
        assert torch.allclose(row_sums, expected_row, atol=1e-2)
        assert torch.allclose(col_sums, expected_col, atol=1e-2)

    def test_embedding_gradient_flow(self):
        torch.manual_seed(42)
        config = make_small_config()
        encoder = MESHEncoder(
            vocab_size=config.model.vocab_size,
            hidden_dim=config.model.hidden_dim,
            sdr_sparsity=config.encoder.sdr_sparsity,
            sdr_candidates=config.encoder.sdr_candidates,
            sinkhorn_epsilon=config.encoder.sinkhorn_epsilon,
            sinkhorn_max_iter=config.encoder.sinkhorn_max_iter,
            sinkhorn_tol=config.encoder.sinkhorn_tol,
            max_seq_length=config.model.max_seq_length,
            soft_sparse=config.encoder.soft_sparse,
        )
        tokens = torch.randint(0, config.model.vocab_size, (1, 6))
        z = encoder(tokens)
        loss = z.abs().sum()
        loss.backward()
        grad = encoder.embedding.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    def test_different_tokens_produce_different_sdrs(self):
        torch.manual_seed(42)
        config = make_small_config()
        encoder = MESHEncoder(
            vocab_size=config.model.vocab_size,
            hidden_dim=config.model.hidden_dim,
            sdr_sparsity=config.encoder.sdr_sparsity,
            sdr_candidates=config.encoder.sdr_candidates,
            sinkhorn_epsilon=config.encoder.sinkhorn_epsilon,
            sinkhorn_max_iter=config.encoder.sinkhorn_max_iter,
            sinkhorn_tol=config.encoder.sinkhorn_tol,
            max_seq_length=config.model.max_seq_length,
            soft_sparse=config.encoder.soft_sparse,
        )
        tokens_a = torch.tensor([[1, 2, 3, 4]])
        tokens_b = torch.tensor([[4, 3, 2, 1]])
        z_a = encoder(tokens_a)
        z_b = encoder(tokens_b)
        assert not torch.allclose(z_a, z_b, atol=1e-4)


class TestComplexMamba3LayerQuality:
    """Quality tests for ComplexMamba3Layer."""

    def test_output_is_complex(self):
        torch.manual_seed(42)
        layer = ComplexMamba3Layer(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
        )
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        y = layer(x)
        assert y.shape == x.shape
        assert y.is_complex()
        assert y.dtype == torch.complex64

    def test_residual_connection_preserves_signal(self):
        torch.manual_seed(42)
        layer = ComplexMamba3Layer(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
        )
        x = torch.randn(1, 6, 64, dtype=torch.complex64)
        y = layer(x)
        correlation = (x.real * y.real).sum() / (x.abs().sum() * y.abs().sum() + 1e-8)
        assert correlation > 0

    def test_sequential_scan_order_matters(self):
        torch.manual_seed(42)
        layer = ComplexMamba3Layer(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
        )
        a = torch.randn(1, 1, 64, dtype=torch.complex64)
        b = torch.randn(1, 1, 64, dtype=torch.complex64)
        c = torch.randn(1, 1, 64, dtype=torch.complex64)
        seq_abc = torch.cat([a, b, c], dim=1)
        seq_cba = torch.cat([c, b, a], dim=1)
        out_abc = layer(seq_abc)
        out_cba = layer(seq_cba)
        diff = (out_abc[:, -1] - out_cba[:, -1]).abs().mean().item()
        assert diff > 1e-4

    def test_near_identity_at_init(self):
        torch.manual_seed(42)
        layer = ComplexMamba3Layer(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
        )
        x = torch.randn(1, 8, 64, dtype=torch.complex64)
        y = layer(x)
        relative_change = (y - x).abs().mean() / (x.abs().mean() + 1e-8)
        assert relative_change < 0.5

    def test_gradient_stability(self):
        torch.manual_seed(42)
        layer = ComplexMamba3Layer(
            hidden_dim=16,
            state_dim=8,
            mimo_groups=2,
            block_size=4,
            d_conv=2,
            num_layers=2,
            max_seq_length=32,
        )
        x = torch.randn(1, 8, 16, dtype=torch.complex64)
        y = layer(x)
        loss = (y.real + y.imag).sum()
        loss.backward()
        grad_norms = [
            p.grad.norm().item() for p in layer.parameters() if p.grad is not None
        ]
        assert grad_norms
        assert max(grad_norms) < 100
        assert all(math.isfinite(g) for g in grad_norms)


class TestLindbladDissipationQuality:
    """Quality tests for LindbladDissipation."""

    def test_output_shape_preserved(self):
        torch.manual_seed(42)
        layer = LindbladDissipation(dim=64, num_lindblad_ops=4, gamma=0.01)
        psi = torch.randn(2, 5, 64, dtype=torch.complex64)
        out = layer(psi)
        assert out.shape == psi.shape

    def test_dissipation_reduces_energy(self):
        torch.manual_seed(42)
        layer = LindbladDissipation(dim=64, num_lindblad_ops=4, gamma=0.05)
        psi = torch.randn(2, 5, 64, dtype=torch.complex64)
        out = layer(psi)
        norm_in = (psi.abs() ** 2).sum(dim=-1).mean()
        norm_out = (out.abs() ** 2).sum(dim=-1).mean()
        assert norm_out <= norm_in + 1e-6

    def test_gamma_controls_dissipation_strength(self):
        torch.manual_seed(42)
        psi = torch.randn(2, 5, 64, dtype=torch.complex64)
        low = LindbladDissipation(dim=64, num_lindblad_ops=4, gamma=0.01)
        high = LindbladDissipation(dim=64, num_lindblad_ops=4, gamma=0.1)
        high.L_real.data.copy_(low.L_real.data)
        high.L_imag.data.copy_(low.L_imag.data)
        high.log_gamma.data.copy_(torch.tensor(math.log(0.1)))
        out_low = low(psi)
        out_high = high(psi)
        norm_low = (out_low.abs() ** 2).sum(dim=-1).mean()
        norm_high = (out_high.abs() ** 2).sum(dim=-1).mean()
        assert norm_high < norm_low

    def test_gradient_flow(self):
        torch.manual_seed(42)
        layer = LindbladDissipation(dim=64, num_lindblad_ops=4, gamma=0.02)
        psi = torch.randn(1, 4, 64, dtype=torch.complex64, requires_grad=True)
        out = layer(psi)
        loss = out.abs().sum()
        loss.backward()
        assert psi.grad is not None
        assert layer.L_real.grad is not None
        assert layer.L_imag.grad is not None
        assert layer.log_gamma.grad is not None


class TestHybridAutomataQuality:
    """Quality tests for HybridAutomata."""

    def test_curvature_detection(self):
        torch.manual_seed(42)
        automata = HybridAutomata(
            dim=32, curvature_threshold=0.1, learnable_threshold=True
        )
        h_prev = torch.randn(2, 32, 32, dtype=torch.complex64)
        h_t = torch.randn(2, 32, 32, dtype=torch.complex64) * 5
        curvature = automata.compute_curvature(h_t, h_prev)
        assert (curvature > automata.log_threshold.exp()).all()

    def test_no_jump_for_smooth_input(self):
        torch.manual_seed(42)
        automata = HybridAutomata(
            dim=32, curvature_threshold=0.5, learnable_threshold=True
        )
        psi = torch.randn(2, 4, 32, dtype=torch.complex64)
        h_prev = torch.randn(2, 32, 32, dtype=torch.complex64) * 0.01
        h_t = h_prev + torch.randn(2, 32, 32, dtype=torch.complex64) * 1e-4
        _, jump_weight = automata(psi, h_t, h_prev)
        assert jump_weight.mean().item() < 0.1

    def test_output_preserves_shape(self):
        torch.manual_seed(42)
        automata = HybridAutomata(
            dim=32, curvature_threshold=0.1, learnable_threshold=True
        )
        psi = torch.randn(2, 4, 32, dtype=torch.complex64)
        h_prev = torch.randn(2, 32, 32, dtype=torch.complex64)
        h_t = torch.randn(2, 32, 32, dtype=torch.complex64)
        out, _ = automata(psi, h_t, h_prev)
        assert out.shape == psi.shape

    def test_learnable_threshold(self):
        torch.manual_seed(42)
        automata = HybridAutomata(
            dim=32, curvature_threshold=0.1, learnable_threshold=True
        )
        psi = torch.randn(2, 4, 32, dtype=torch.complex64)
        h_prev = torch.randn(2, 32, 32, dtype=torch.complex64)
        h_t = torch.randn(2, 32, 32, dtype=torch.complex64) * 2
        _, jump_weight = automata(psi, h_t, h_prev)
        loss = jump_weight.sum()
        loss.backward()
        grad = automata.log_threshold.grad
        assert grad is not None
        assert torch.isfinite(grad).all()


class TestQuaternionicEscapeQuality:
    """Quality tests for QuaternionicEscape."""

    def test_escape_on_ill_conditioned_hamiltonian(self):
        torch.manual_seed(42)
        escape = QuaternionicEscape(dim=16, condition_threshold=10.0, learnable=False)
        psi = torch.randn(2, 4, 16, dtype=torch.complex64)
        h = torch.zeros(2, 16, 16)
        h[:, 0, 0] = 1e6
        psi_out, mask = escape(psi, h)
        assert mask.any()
        assert psi_out.is_complex()

    def test_no_escape_on_well_conditioned_hamiltonian(self):
        torch.manual_seed(42)
        escape = QuaternionicEscape(dim=16, condition_threshold=100.0, learnable=False)
        psi = torch.randn(2, 4, 16, dtype=torch.complex64)
        h = torch.eye(16).unsqueeze(0).repeat(2, 1, 1) * 0.01
        _, mask = escape(psi, h)
        assert not mask.any()

    def test_output_is_complex(self):
        torch.manual_seed(42)
        escape = QuaternionicEscape(dim=16, condition_threshold=10.0, learnable=False)
        psi = torch.randn(1, 3, 16, dtype=torch.complex64)
        h = torch.eye(16).unsqueeze(0)
        psi_out, _ = escape(psi, h)
        assert psi_out.is_complex()
        assert psi_out.dtype == torch.complex64

    def test_svd_fallback(self):
        torch.manual_seed(42)
        escape = QuaternionicEscape(dim=8, condition_threshold=10.0, learnable=False)
        h = torch.eye(8).unsqueeze(0)
        monkeypatch = pytest.MonkeyPatch()

        def raise_svd(_):
            raise RuntimeError("svd failed")

        monkeypatch.setattr(torch.linalg, "svdvals", raise_svd)
        _, condition = escape.detect_singularity(h)
        monkeypatch.undo()
        assert torch.isfinite(condition).all()


class TestCayleyPropagatorQuality:
    """Quality tests for CayleySolitonPropagator."""

    def test_unitarity_preservation(self):
        torch.manual_seed(42)
        prop = CayleySolitonPropagator(
            dim=32,
            dt=0.1,
            nonlinear_alpha=0.1,
            cg_max_iter=15,
            cg_tol=1e-6,
            laplacian_sparsity=3,
            num_scales=1,
        )
        psi = torch.randn(2, 6, 32, dtype=torch.complex64)
        out = prop(psi)
        norm_in = (psi.abs() ** 2).sum(dim=-1)
        norm_out = (out.abs() ** 2).sum(dim=-1)
        ratio = norm_out / (norm_in + 1e-12)
        assert (ratio - 1.0).abs().max() < 0.1

    def test_output_is_complex(self):
        torch.manual_seed(42)
        prop = CayleySolitonPropagator(dim=32, num_scales=1)
        psi = torch.randn(1, 4, 32, dtype=torch.complex64)
        out = prop(psi)
        assert out.is_complex()
        assert out.dtype == torch.complex64

    def test_laplacian_is_symmetric(self):
        torch.manual_seed(42)
        prop = CayleySolitonPropagator(dim=32, num_scales=1)
        h = prop.hamiltonian.get_hamiltonian_dense()
        assert torch.allclose(h, h.transpose(-2, -1), atol=1e-5)

    def test_gradient_through_cg_solver(self):
        torch.manual_seed(42)
        prop = CayleySolitonPropagator(dim=16, num_scales=1, cg_max_iter=10)
        psi = torch.randn(1, 3, 16, dtype=torch.complex64, requires_grad=True)
        out = prop(psi)
        loss = out.abs().sum()
        loss.backward()
        assert psi.grad is not None
        param_grads = [p.grad for p in prop.parameters() if p.grad is not None]
        assert param_grads


class TestBornCollapseQuality:
    """Quality tests for BornCollapseSampler."""

    def test_logits_are_real(self):
        torch.manual_seed(42)
        sampler = BornCollapseSampler(hidden_dim=32, vocab_size=64)
        psi = torch.randn(2, 4, 32, dtype=torch.complex64)
        logits = sampler.compute_logits(psi)
        assert not logits.is_complex()

    def test_probability_normalization(self):
        torch.manual_seed(42)
        sampler = BornCollapseSampler(hidden_dim=32, vocab_size=64, top_k=0, top_p=1.0)
        psi = torch.randn(1, 3, 32, dtype=torch.complex64)
        logits = sampler.compute_logits(psi)
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_born_rule_applied(self):
        torch.manual_seed(42)
        sampler = BornCollapseSampler(hidden_dim=4, vocab_size=2, top_k=0, top_p=1.0)
        with torch.no_grad():
            sampler.proj_real.weight.zero_()
            sampler.proj_imag.weight.zero_()
            sampler.output_bias.zero_()
            sampler.proj_real.weight[0, 0] = 1.0
        psi_small = torch.zeros(1, 1, 4, dtype=torch.complex64)
        psi_large = torch.zeros(1, 1, 4, dtype=torch.complex64)
        psi_small[..., 0] = 0.1 + 0j
        psi_large[..., 0] = 1.0 + 0j
        logit_small = sampler.compute_logits(psi_small)[0, 0, 0].item()
        logit_large = sampler.compute_logits(psi_large)[0, 0, 0].item()
        assert logit_large > logit_small

    def test_temperature_effect(self):
        torch.manual_seed(42)
        sampler = BornCollapseSampler(hidden_dim=32, vocab_size=64, top_k=0, top_p=1.0)
        psi = torch.randn(1, 1, 32, dtype=torch.complex64)
        result_t1 = sampler(psi, temperature=1.0, top_k=0, top_p=1.0, sample=True)
        result_t01 = sampler(psi, temperature=0.1, top_k=0, top_p=1.0, sample=True)
        p1 = result_t1["probs"]
        p01 = result_t01["probs"]
        entropy_t1 = -(p1 * (p1 + 1e-12).log()).sum(dim=-1)
        entropy_t01 = -(p01 * (p01 + 1e-12).log()).sum(dim=-1)
        assert entropy_t1.mean() > entropy_t01.mean()


class TestMHCResidualQuality:
    """Quality tests for mhc_residual."""

    def test_s1_fast_path(self):
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, dtype=torch.complex64)
        branch = torch.randn(2, 4, 16, dtype=torch.complex64)
        h_logits = torch.zeros(1, 1)
        out = mhc_residual(x, branch, h_logits, complex_mode=True)
        assert torch.allclose(out, x + branch, atol=1e-5)

    def test_multi_stream_mixing(self):
        torch.manual_seed(42)
        logits_real = torch.randn(4, 4)
        logits_imag = torch.randn(4, 4)
        h_real, h_imag = sinkhorn_log_complex(
            logits_real, logits_imag, num_iters=20, tau=0.05
        )
        h_mag = torch.sqrt(h_real**2 + h_imag**2)
        row_sums = h_mag.sum(dim=-1)
        col_sums = h_mag.sum(dim=-2)
        assert torch.allclose(row_sums, torch.ones(4), atol=3e-2)
        assert torch.allclose(col_sums, torch.ones(4), atol=3e-2)

    def test_complex_mode(self):
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, dtype=torch.complex64)
        branch = torch.randn(2, 4, 16, dtype=torch.complex64)
        h_logits = torch.zeros(1, 1)
        out = mhc_residual(x, branch, h_logits, complex_mode=True)
        assert out.is_complex()
        assert out.dtype == torch.complex64

    def test_gradient_no_explosion(self):
        torch.manual_seed(42)
        x = torch.randn(2, 4, 16, dtype=torch.complex64, requires_grad=True)
        branch = torch.randn(2, 4, 16, dtype=torch.complex64, requires_grad=True)
        h_logits = torch.zeros(1, 1)
        out = mhc_residual(x, branch, h_logits, complex_mode=True)
        loss = out.abs().sum()
        loss.backward()
        grad_norm = x.grad.norm().item()
        assert grad_norm < 100
        assert math.isfinite(grad_norm)


class TestV8IntegrationQuality:
    """Quality tests for ComplexMamba3LayerV8."""

    def test_all_v8_features_together(self):
        torch.manual_seed(42)
        layer = ComplexMamba3LayerV8(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
            use_mhc=True,
            use_lindblad=True,
            use_hybrid_automata=True,
            use_quaternionic=True,
        )
        x = torch.randn(2, 6, 64, dtype=torch.complex64)
        out = layer(x)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()

    def test_v8_off_matches_base(self):
        torch.manual_seed(42)
        base = ComplexMamba3Layer(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
        )
        v8 = ComplexMamba3LayerV8(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
            use_mhc=False,
            use_lindblad=False,
            use_hybrid_automata=False,
            use_quaternionic=False,
        )
        v8.base_layer.load_state_dict(base.state_dict())
        x = torch.randn(1, 5, 64, dtype=torch.complex64)
        out_base = base(x)
        out_v8 = v8(x)
        assert torch.allclose(out_v8, out_base, atol=1e-5)

    def test_gradient_norms_reasonable(self):
        torch.manual_seed(42)
        layer = ComplexMamba3LayerV8(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            num_layers=2,
            max_seq_length=32,
            use_mhc=True,
            use_lindblad=True,
            use_hybrid_automata=True,
            use_quaternionic=True,
        )
        x = torch.randn(1, 4, 64, dtype=torch.complex64, requires_grad=True)
        out = layer(x)
        loss = out.abs().sum()
        loss.backward()
        grad_norms = [
            p.grad.norm().item() for p in layer.parameters() if p.grad is not None
        ]
        assert grad_norms
        assert max(grad_norms) < 100
        assert all(math.isfinite(g) for g in grad_norms)
