"""Tests for adaptive chunk sizing in complex parallel scan.

Validates:
- Optimal chunk size computation based on hardware and tensor dimensions
- Backward compatibility with fixed chunk_size
- Correctness across different chunk sizing strategies
- Cache efficiency heuristics
"""
import torch
import pytest
import math


class TestOptimalChunkSize:
    """Test the chunk size computation logic."""

    def test_get_optimal_chunk_size_auto(self):
        """Auto strategy should return reasonable chunk sizes."""
        from sem.spinor.complex_pscan import get_optimal_chunk_size, _npo2

        # Test with typical tensor shapes
        test_cases = [
            # (B, S, D) -> expected to be power of 2 and reasonable
            (1, 1024, 512),
            (8, 256, 64),
            (4, 2048, 256),
            (16, 512, 128),
        ]

        for B, S, D in test_cases:
            A_r = torch.randn(B, S, D)
            chunk = get_optimal_chunk_size(A_r, strategy="auto")

            # Should be power of 2
            assert chunk == _npo2(chunk), f"Chunk {chunk} not power of 2"

            # Should be in reasonable range
            assert 64 <= chunk <= 2048, f"Chunk {chunk} out of range [64, 2048]"

            # Should not exceed sequence length
            assert chunk <= max(64, _npo2(S)), f"Chunk {chunk} exceeds padded seq len"

    def test_get_optimal_chunk_size_fixed(self):
        """Fixed strategy should respect provided chunk_size."""
        from sem.spinor.complex_pscan import get_optimal_chunk_size, _npo2

        A_r = torch.randn(4, 512, 128)

        # Should round to power of 2
        assert get_optimal_chunk_size(A_r, chunk_size=100, strategy="fixed") == _npo2(100)
        assert get_optimal_chunk_size(A_r, chunk_size=256, strategy="fixed") == 256
        assert get_optimal_chunk_size(A_r, chunk_size=300, strategy="fixed") == _npo2(300)

    def test_get_optimal_chunk_size_cache(self):
        """Cache strategy should optimize for L2 cache."""
        from sem.spinor.complex_pscan import get_optimal_chunk_size

        # Small tensors should allow larger chunks
        A_small = torch.randn(1, 1024, 64)
        chunk_small = get_optimal_chunk_size(A_small, strategy="cache")

        # Larger tensors should use smaller chunks
        A_large = torch.randn(16, 1024, 512)
        chunk_large = get_optimal_chunk_size(A_large, strategy="cache")

        # Larger B*D should result in smaller or equal chunks
        assert chunk_large <= chunk_small, \
            f"Larger tensor should have smaller chunk: {chunk_large} vs {chunk_small}"

    def test_get_optimal_chunk_size_balanced(self):
        """Balanced strategy should target sqrt(S) chunks."""
        from sem.spinor.complex_pscan import get_optimal_chunk_size

        # For S=1024, sqrt(S) = 32, so ~32 chunks of ~32 each
        A_r = torch.randn(4, 1024, 128)
        chunk = get_optimal_chunk_size(A_r, strategy="balanced")

        # Should be around S / sqrt(S) = sqrt(S)
        expected_target = int(math.sqrt(1024))  # ~32
        assert 32 <= chunk <= 128, \
            f"Balanced chunk {chunk} too far from sqrt(S)={expected_target}"

    def test_chunk_size_scales_with_batch(self):
        """Chunk size should decrease as batch size increases."""
        from sem.spinor.complex_pscan import get_optimal_chunk_size

        S, D = 1024, 256

        chunks = []
        for B in [1, 4, 8, 16]:
            A_r = torch.randn(B, S, D)
            chunk = get_optimal_chunk_size(A_r, strategy="auto")
            chunks.append(chunk)

        # Chunks should be non-increasing as B increases
        for i in range(len(chunks) - 1):
            assert chunks[i] >= chunks[i + 1], \
                f"Chunk increased with larger B: {chunks}"


class TestChunkedScanCorrectness:
    """Test that chunked scan produces correct results with different strategies."""

    def _sequential_scan(self, A_r, A_i, X_r, X_i):
        """Reference sequential implementation."""
        B, S, D = A_r.shape
        H_r = torch.zeros_like(X_r)
        H_i = torch.zeros_like(X_i)

        H_r[:, 0] = X_r[:, 0]
        H_i[:, 0] = X_i[:, 0]

        for t in range(1, S):
            # Complex multiply: (a + bi) * (h + gi) = (ah - bg) + (ag + bh)i
            H_r[:, t] = A_r[:, t] * H_r[:, t-1] - A_i[:, t] * H_i[:, t-1] + X_r[:, t]
            H_i[:, t] = A_r[:, t] * H_i[:, t-1] + A_i[:, t] * H_r[:, t-1] + X_i[:, t]

        return H_r, H_i

    @pytest.mark.parametrize("strategy", ["auto", "cache", "balanced", "fixed"])
    def test_chunked_matches_sequential(self, strategy):
        """Chunked scan should match sequential reference for all strategies."""
        from sem.spinor.complex_pscan import complex_parallel_scan_chunked

        torch.manual_seed(42)
        B, S, D = 2, 128, 32

        # Use stable A (|A| < 1)
        A_r = torch.randn(B, S, D) * 0.5
        A_i = torch.randn(B, S, D) * 0.3
        X_r = torch.randn(B, S, D)
        X_i = torch.randn(B, S, D)

        # Reference
        ref_H_r, ref_H_i = self._sequential_scan(A_r, A_i, X_r, X_i)

        # Chunked with strategy
        chunk_size = 64 if strategy == "fixed" else None
        H_r, H_i = complex_parallel_scan_chunked(
            A_r.clone(), A_i.clone(), X_r.clone(), X_i.clone(),
            chunk_size=chunk_size,
            chunk_strategy=strategy
        )

        # Should match within tolerance
        assert torch.allclose(H_r, ref_H_r, rtol=1e-4, atol=1e-5), \
            f"H_r mismatch with strategy={strategy}"
        assert torch.allclose(H_i, ref_H_i, rtol=1e-4, atol=1e-5), \
            f"H_i mismatch with strategy={strategy}"

    def test_backward_compatible_fixed_chunk(self):
        """Fixed chunk_size=256 should work as before."""
        from sem.spinor.complex_pscan import complex_parallel_scan_chunked

        torch.manual_seed(42)
        B, S, D = 2, 512, 64

        A_r = torch.randn(B, S, D) * 0.5
        A_i = torch.randn(B, S, D) * 0.3
        X_r = torch.randn(B, S, D)
        X_i = torch.randn(B, S, D)

        # Old style call (positional chunk_size)
        H_r_old, H_i_old = complex_parallel_scan_chunked(
            A_r.clone(), A_i.clone(), X_r.clone(), X_i.clone(),
            chunk_size=256,
            chunk_strategy="fixed"
        )

        # Reference
        ref_H_r, ref_H_i = self._sequential_scan(A_r, A_i, X_r, X_i)

        assert torch.allclose(H_r_old, ref_H_r, rtol=1e-4, atol=1e-5)
        assert torch.allclose(H_i_old, ref_H_i, rtol=1e-4, atol=1e-5)

    def test_auto_chunk_no_nan(self):
        """Auto chunking should not produce NaN."""
        from sem.spinor.complex_pscan import complex_parallel_scan_chunked

        torch.manual_seed(123)

        # Various tensor shapes
        test_cases = [
            (1, 64, 16),
            (4, 256, 32),
            (2, 1024, 64),
            (8, 512, 128),
        ]

        for B, S, D in test_cases:
            A_r = torch.randn(B, S, D) * 0.5
            A_i = torch.randn(B, S, D) * 0.3
            X_r = torch.randn(B, S, D)
            X_i = torch.randn(B, S, D)

            H_r, H_i = complex_parallel_scan_chunked(
                A_r, A_i, X_r, X_i,
                chunk_size=None,
                chunk_strategy="auto"
            )

            assert not torch.isnan(H_r).any(), f"NaN in H_r for shape {(B, S, D)}"
            assert not torch.isnan(H_i).any(), f"NaN in H_i for shape {(B, S, D)}"


class TestHardwareDetection:
    """Test hardware cache size detection."""

    def test_cpu_cache_size(self):
        """CPU should return reasonable cache size."""
        from sem.spinor.complex_pscan import _get_device_cache_size

        cpu_device = torch.device("cpu")
        cache_size = _get_device_cache_size(cpu_device)

        # Should be in reasonable range (1MB - 32MB)
        assert 1 * 1024 * 1024 <= cache_size <= 32 * 1024 * 1024, \
            f"CPU cache size {cache_size} out of range"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_cache_size(self):
        """CUDA should return L2 cache size for detected GPU."""
        from sem.spinor.complex_pscan import _get_device_cache_size

        cuda_device = torch.device("cuda:0")
        cache_size = _get_device_cache_size(cuda_device)

        # Should be in reasonable range (4MB - 100MB)
        assert 4 * 1024 * 1024 <= cache_size <= 100 * 1024 * 1024, \
            f"CUDA cache size {cache_size} out of range"

    def test_npo2(self):
        """Next power of 2 should be correct."""
        from sem.spinor.complex_pscan import _npo2

        assert _npo2(1) == 1
        assert _npo2(2) == 2
        assert _npo2(3) == 4
        assert _npo2(7) == 8
        assert _npo2(8) == 8
        assert _npo2(100) == 128
        assert _npo2(256) == 256
        assert _npo2(257) == 512

    def test_largest_po2(self):
        """Largest power of 2 <= n should be correct."""
        from sem.spinor.complex_pscan import _largest_po2

        assert _largest_po2(0) == 0
        assert _largest_po2(1) == 1
        assert _largest_po2(2) == 2
        assert _largest_po2(3) == 2
        assert _largest_po2(7) == 4
        assert _largest_po2(8) == 8
        assert _largest_po2(100) == 64
        assert _largest_po2(256) == 256
        assert _largest_po2(257) == 256


class TestGradientFlow:
    """Test gradient flow through adaptive chunking."""

    @pytest.mark.parametrize("strategy", ["auto", "balanced"])
    def test_gradient_flow_all_strategies(self, strategy):
        """Gradients should flow through chunked scan with any strategy."""
        from sem.spinor.complex_pscan import complex_parallel_scan_chunked

        torch.manual_seed(42)
        B, S, D = 2, 256, 32

        A_r = torch.randn(B, S, D, requires_grad=True)
        A_i = torch.randn(B, S, D, requires_grad=True)
        X_r = torch.randn(B, S, D, requires_grad=True)
        X_i = torch.randn(B, S, D, requires_grad=True)

        H_r, H_i = complex_parallel_scan_chunked(
            A_r, A_i, X_r, X_i,
            chunk_size=None,
            chunk_strategy=strategy
        )

        loss = (H_r**2 + H_i**2).sum()
        loss.backward()

        assert A_r.grad is not None, f"No grad for A_r with strategy={strategy}"
        assert A_i.grad is not None, f"No grad for A_i with strategy={strategy}"
        assert X_r.grad is not None, f"No grad for X_r with strategy={strategy}"
        assert X_i.grad is not None, f"No grad for X_i with strategy={strategy}"

        # Gradients should be non-zero
        assert A_r.grad.abs().sum() > 0
        assert X_r.grad.abs().sum() > 0
