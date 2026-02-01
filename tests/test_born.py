"""Tests for Born Collapse Sampler.

Validates:
- Probability normalization (P sums to 1)
- Temperature scaling behavior
- Top-k/top-p filtering
- Differentiability for training
- Born rule: P ∝ |ψ|²
"""
import torch
import pytest


class TestBornCollapse:
    """Test the Born Collapse Sampler."""

    def test_output_shape(self):
        """Sampler should produce correct output shapes."""
        from sem.sampler.born_collapse import BornCollapseSampler

        sampler = BornCollapseSampler(
            hidden_dim=16, vocab_size=100,
            temperature=1.0, top_k=50, top_p=0.95
        )

        psi = torch.randn(2, 8, 16, dtype=torch.complex64)
        result = sampler(psi, sample=True)

        assert 'logits' in result
        assert 'log_probs' in result
        assert 'tokens' in result
        assert result['logits'].shape == (2, 8, 100)
        assert result['tokens'].shape == (2, 8)

    def test_probability_normalization(self):
        """Probabilities should sum to 1 along vocab dimension."""
        from sem.sampler.born_collapse import BornCollapseSampler

        sampler = BornCollapseSampler(
            hidden_dim=16, vocab_size=50, top_k=0, top_p=1.0
        )

        psi = torch.randn(1, 4, 16, dtype=torch.complex64)
        result = sampler(psi, sample=True)

        prob_sums = result['probs'].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            f"Probabilities don't sum to 1: {prob_sums}"

    def test_temperature_effect(self):
        """Higher temperature should produce higher entropy (more uniform)."""
        from sem.sampler.born_collapse import BornCollapseSampler

        sampler_low = BornCollapseSampler(hidden_dim=16, vocab_size=50)
        sampler_high = BornCollapseSampler(hidden_dim=16, vocab_size=50)

        # Share weights
        sampler_high.load_state_dict(sampler_low.state_dict())

        psi = torch.randn(1, 1, 16, dtype=torch.complex64)

        result_low = sampler_low(psi, temperature=0.1, top_k=0, top_p=1.0, sample=True)
        result_high = sampler_high(psi, temperature=2.0, top_k=0, top_p=1.0, sample=True)

        # Entropy: H = -sum(p * log(p))
        p_low = result_low['probs']
        p_high = result_high['probs']

        entropy_low = -(p_low * (p_low + 1e-12).log()).sum(dim=-1)
        entropy_high = -(p_high * (p_high + 1e-12).log()).sum(dim=-1)

        assert entropy_high.mean() > entropy_low.mean(), \
            f"Higher temp should have higher entropy: {entropy_high.mean():.4f} vs {entropy_low.mean():.4f}"

    def test_training_forward(self):
        """Training forward should return differentiable log_probs."""
        from sem.sampler.born_collapse import BornCollapseSampler

        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=50)

        psi = torch.randn(2, 4, 16, dtype=torch.complex64, requires_grad=True)
        log_probs = sampler.training_forward(psi)

        assert log_probs.shape == (2, 4, 50)

        # Should be differentiable
        loss = log_probs.sum()
        loss.backward()
        assert psi.grad is not None

    def test_top_k_filtering(self):
        """Top-k should zero out all but k highest logits."""
        from sem.sampler.born_collapse import BornCollapseSampler

        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=20)

        psi = torch.randn(1, 1, 16, dtype=torch.complex64)
        result = sampler(psi, top_k=5, top_p=1.0, sample=True)

        probs = result['probs']
        nonzero = (probs > 1e-8).sum(dim=-1)
        assert nonzero.max() <= 5, f"Top-k=5 but {nonzero.max()} tokens have nonzero prob"
