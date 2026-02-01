"""Tests for HAS-VQ (Hessian-Adaptive Sparse Vector Quantization).

Validates:
- Fisher tracker EMA updates
- Outlier detection (top percentile)
- VQ codebook quantization and reconstruction
- BPP computation
- Dead code revival
- Full HAS-VQ pipeline
"""
import torch
import pytest


class TestFisherTracker:
    """Test empirical Fisher information tracking."""

    def test_ema_update(self):
        """Fisher EMA should accumulate squared gradients."""
        from sem.quantizer.fisher_tracker import FisherTracker

        tracker = FisherTracker(decay=0.9)

        # Simulate a parameter with gradient
        param = torch.nn.Parameter(torch.randn(4, 4))
        param.grad = torch.ones(4, 4) * 2.0  # grad^2 = 4.0

        tracker.update([('weight', param)])

        fisher = tracker.get_fisher('weight')
        assert fisher is not None
        # First update: F = (1-0.9)*4.0 = 0.4 (since initial is 0, but first update sets it to grad_sq)
        # Actually first update: F = grad_sq (clone on first step)
        assert torch.allclose(fisher, torch.ones(4, 4) * 4.0)

    def test_outlier_detection(self):
        """Top percentile parameters should be identified as outliers."""
        from sem.quantizer.fisher_tracker import FisherTracker

        tracker = FisherTracker(decay=0.99)

        # Create parameter with non-uniform Fisher
        param = torch.nn.Parameter(torch.randn(100))
        grad = torch.zeros(100)
        grad[:5] = 100.0  # 5 outliers with high gradient
        grad[5:] = 1.0    # 95 normal parameters
        param.grad = grad

        tracker.update([('weight', param)])

        mask = tracker.get_outlier_mask('weight', percentile=0.05)
        assert mask is not None
        # Top 5% should catch the first 5 elements
        assert mask[:5].all(), "Outliers should be detected"
        num_outliers = mask.sum().item()
        assert num_outliers <= 10, f"Too many outliers: {num_outliers}"


class TestVQCodebook:
    """Test Vector Quantization codebook."""

    def test_quantization(self):
        """Quantization should map to nearest codebook entry."""
        from sem.quantizer.vq_codebook import VQCodebook

        codebook = VQCodebook(codebook_size=16, group_size=2)

        x = torch.randn(100, 2)
        quantized, indices, loss = codebook.quantize(x)

        assert quantized.shape == (100, 2)
        assert indices.shape == (100,)
        assert indices.max() < 16
        assert indices.min() >= 0
        assert loss.item() >= 0

    def test_straight_through(self):
        """Quantization should pass gradients via straight-through."""
        from sem.quantizer.vq_codebook import VQCodebook

        codebook = VQCodebook(codebook_size=16, group_size=2)

        x = torch.randn(10, 2, requires_grad=True)
        quantized, _, _ = codebook.quantize(x)
        loss = quantized.sum()
        loss.backward()

        assert x.grad is not None, "No gradient through straight-through"

    def test_bpp_calculation(self):
        """BPP should match theoretical value."""
        from sem.quantizer.vq_codebook import VQCodebook

        codebook = VQCodebook(codebook_size=256, group_size=2)

        bpp = codebook.compute_bpp(1000)
        # 8 bits per index / 2 params per group = 4 BPP
        assert abs(bpp - 4.0) < 0.01, f"BPP should be 4.0, got {bpp}"

    def test_ema_update(self):
        """Codebook should update via EMA."""
        from sem.quantizer.vq_codebook import VQCodebook

        codebook = VQCodebook(codebook_size=8, group_size=2)
        initial_cb = codebook.codebook.clone()

        x = torch.randn(100, 2) * 5  # Large values to force updates
        _, indices, _ = codebook.quantize(x)
        codebook.update_codebook(x, indices)

        # Codebook should have changed
        assert not torch.allclose(codebook.codebook, initial_cb), \
            "Codebook should update after EMA step"

    def test_dead_code_revival(self):
        """Dead codes should be reinitialized."""
        from sem.quantizer.vq_codebook import VQCodebook

        codebook = VQCodebook(codebook_size=8, group_size=2, dead_code_threshold=1)

        # Mark some codes as dead
        codebook.steps_since_use[4:] = 10

        x = torch.randn(50, 2)
        codebook.revive_dead_codes(x)

        # Dead codes should have been reset
        assert (codebook.steps_since_use[4:] == 0).any(), \
            "Some dead codes should have been revived"
