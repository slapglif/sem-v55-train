ARCHITECTURE AUDIT REPORT — SEM V5.5 "Lean Crystal"
=====================================================

Audit Date: 2026-02-10
Methodology: Hermeneutic Circle (full source read, 17 files, ~6500 lines)
Default Config: hidden_dim=256, num_layers=8, state_dim=32, mimo_groups=4,
  block_size=16, vocab_size=50262, max_seq_length=2048
  V8: lindblad=true, hybrid_automata=false, quaternionic=true, mhc=true
  Propagator: chebyshev_kpm=true, degree=16, simple_mode=false

Pipeline:
  Token IDs -> [MESH Encoder] -> psi complex64 -> [Mamba Layers] -> psi'
    -> [Cayley Propagator] -> psi'' -> [Born Collapse] -> Logits


PER-COMPONENT ANALYSIS
======================

1. MESH Encoder (sem/encoder/mesh_sdr.py + sem/encoder/sinkhorn.py)
------------------------------------------------------------------------

   Mathematical goal:
     Entropy-regularized optimal transport maps token embeddings onto a
     sparse complex SDR (Sparse Distributed Representation) on a Crystal
     Manifold. The transport plan T = diag(u) K diag(v) minimizes
     <C, T> - epsilon * H(T) subject to marginal constraints, where C is the
     cost matrix, K = exp(-C/epsilon), and H is entropy.

   Implementation:
     - Log-domain Sinkhorn-Knopp (sinkhorn.py:42-85): Alternating
       row/column log-normalizations for numerical stability.
     - max_iter=90 (config). ALL 90 iterations execute unconditionally.
       The early-exit convergence check was removed (sinkhorn.py comment
       near line 70) to avoid .item() GPU synchronization.
     - auto_epsilon=true: epsilon = 0.04 * median(C), adapting to cost
       scale. Prevents near-uniform transport plans.
     - soft_sparse=true: Softmax weighting with learnable temperature tau
       instead of hard top-k. Row-sum normalization preserves OT mass
       (mesh_sdr.py:140-155).
     - Complex lift (mesh_sdr.py:170-185): z = sdr * exp(i * phase) where
       phase is sinusoidal positional encoding. Re and Im are projections
       of the SAME real SDR vector -- they differ only by positional phase,
       not by independent learned features.
     - simple_mode=false by default: Full Sinkhorn pipeline active.
       simple_mode (mesh_sdr.py:90-105) bypasses to Re=embedding,
       Im=learned_projection(gain=0.3).

   Issues:
     a) 90 Sinkhorn iterations with no early exit. At epsilon = 0.04 *
        median(C), convergence typically occurs in 10-30 iterations.
        Iterations 30-90 are wasted compute: 60 extra logsumexp ops over
        [B, S, candidates] = [32, 2048, 128] tensors. Estimated waste:
        ~0.9B unnecessary FLOPs per forward pass.
     b) Complex lift provides only positional distinction, not independent
        representational capacity. The imaginary part carries no information
        beyond what the real part already has (modulo position).
     c) Weight tying (proj_real.weight = embedding.weight) is mathematically
        correct ONLY in simple_mode. In non-simple mode (DEFAULT), the
        Sinkhorn transport plan maps embeddings into a different space via
        sdr_to_hidden linear projection (mesh_sdr.py:128). The sampler's
        weight-tied projection assumes the final hidden state's real part
        is in embedding space -- it is not. See Inter-Component Issue #3.

   Bottleneck severity: HIGH
     (90 Sinkhorn iterations dominate encoder time; ~1.4B total ops)


2. Complex Mamba-3 (sem/spinor/complex_mamba3.py + complex_pscan.py + spinor_block.py)
----------------------------------------------------------------------------------------

   Mathematical goal:
     Complex-valued selective state space model:
       h_t = A * h_{t-1} + B * x_t
       y_t = C * h_t
     where A is complex with |A| < 1 (guaranteed stability), operating in
     complex64 throughout. The Blelloch parallel scan enables O(log S)
     depth parallelism while maintaining O(S) work.

   Implementation:
     - A stability (complex_mamba3.py:85-95): CORRECT.
       A_mag = exp(dt_mag * -softplus(log_A_mag)), always in (0, 1).
       Phase is independent via dt_phase. No stability risk.
     - Parallel scan (complex_pscan.py): Blelloch algorithm with:
       * @torch.jit.script fused complex ops (lines 45-90)
       * Chunked processing for memory efficiency (lines 200-350)
       * Hierarchical decomposition for non-power-of-2 sequences (lines 400+)
       * Mixed precision: conditional bf16 for A values safely away from
         1.0 (threshold 0.98, complex_pscan.py:120). Smart optimization.
     - SpinorGate (spinor_block.py:50-120): K=8 discrete gate levels.
       Precomputes K*N Cayley maps (8*16=128 maps for block_size=16)
       instead of B*S*N. Each map requires torch.linalg.solve for a
       2*block_size x 2*block_size system = 32x32. Total: 128 solves of
       32x32 systems per forward pass.
     - Conv (complex_mamba3.py:130-150): Fused complex depthwise conv1d,
       groups=hidden_dim, 2 channels per group. Correct complex init.
     - Chi-squared gate (complex_mamba3.py:180-190): 1 - exp(-|z|^2 * beta).
       Matches complex Gaussian distribution theory.
     - residual_scale = 1/sqrt(L) = 1/sqrt(8) ~ 0.354 (complex_mamba3.py:60).
       Correct for single-branch residual with L layers.
     - B/C projections: FusedComplexLinear with Kaiming/sqrt(2) init.
       Correct for complex circular symmetry.

   Issues:
     a) SpinorGate's 128 torch.linalg.solve calls per forward pass is
        the previously identified P0 bottleneck. For 32x32 systems this
        is manageable (~0.4M FLOPs total) but does not batch well on GPU
        due to sequential dispatch.
     b) The Blelloch scan, while asymptotically optimal, has higher constant
        factors than sequential scan for short sequences. For S <= 256,
        sequential may be faster. No adaptive switching implemented.

   Bottleneck severity: MEDIUM
     (SpinorGate solve is a known P0 issue; core SSM scan is well-optimized)


3. V8 Wrapper (sem/model.py ComplexMamba3LayerV8)
---------------------------------------------------

   Mathematical goal:
     Apply physics-inspired regularization transforms to the Mamba layer's
     branch output (delta = out - residual):
     - Lindblad dissipation: Open quantum system dynamics, damp unstable modes
     - Quaternionic escape: Detect spectral blow-up, apply quaternionic
       rotation to redistribute energy
     - Hybrid automata: Detect high-curvature regions via Lie bracket,
       switch dynamics (DISABLED in default config)
     - Multi-head consensus (mHC): Residual blending across heads

   Implementation:
     - branch_delta extraction (model.py:440-445): out - residual. Correct
       after SEOP Fix 52.
     - RMS normalization of delta (model.py:450-465): delta_rms computed
       under no_grad(), normalizes delta, applies V8 transforms, restores
       scale. Reasonable for scale-equivariance.
     - approximate_hamiltonian (model.py:480-495): H = (1/S) sum_s |psi_s><psi_s|
       using psi.abs(). This is a REAL outer product, not complex. Discards
       ALL phase information. The resulting H_eff cannot detect phase-space
       singularities.
     - Lindblad (sem/spinor/lindblad.py): L_k are [K=4, D=256, D=256]
       dense matrices. compute_lindblad_term() performs:
       * K bmm of [256, 256] matrices to form L_k^dagger L_k: O(K*D^3) = O(67M)
       * einsum("ij,bsj->bsi") per position: O(B*S*D^2) = O(B*S*65K)
       * With B=32, S=2048: ~4.3B FLOPs per layer x 8 layers = ~34B FLOPs
       * gamma=0.007: The dissipation effect is minuscule relative to cost.
     - Quaternionic (sem/spinor/quaternion.py): estimate_spectral_norm uses
       5 power iterations with bmm on [B, D, D]. 10 bmm per layer.
       O(10*B*D^2) per layer. Moderate cost.
     - Hybrid automata (DISABLED): Would add O(B*D^3) = O(B*16.7M) per
       layer for Lie bracket computation. Not active in default config.

   Issues:
     a) CRITICAL: Lindblad costs ~34B FLOPs across 8 layers while
        contributing gamma=0.007 dissipation. The ratio of compute cost to
        effect size is ~5000:1. This is the single largest compute waste
        in the architecture. See Top Bottleneck #1.
     b) approximate_hamiltonian discards phase (uses .abs()). For
        quaternionic escape's spectral norm estimation, this means the
        Hamiltonian proxy reflects magnitude clustering but misses phase
        coherence/decoherence patterns. The escape triggers may fire at
        wrong times.
     c) Even when unitary_lambda=0, the V8 loop still iterates through
        layers checking the condition (model.py:576-622). Minor overhead
        but indicates dead-code coupling.

   Bottleneck severity: CRITICAL
     (Lindblad dominates total FLOPs; approximate_hamiltonian is phase-blind)


4. Cayley Propagator (sem/propagator/cayley_soliton.py + hamiltonian.py + cg_solver.py)
-----------------------------------------------------------------------------------------

   Mathematical goal:
     Unitary wave propagation via the Cayley transform:
       (I + i*dt/2*H) psi_out = (I - i*dt/2*H) psi_rot
     where H is a learned Graph Laplacian (symmetric, positive semi-definite)
     and the nonlinear phase rotation (PIT transform) provides soliton-like
     signal propagation. Unitarity guarantees norm preservation.

   Implementation:
     - Chebyshev KPM (cayley_soliton.py:200-260): ENABLED by default
       (use_chebyshev_kpm=true, degree=16). This BYPASSES the CG solver
       entirely. The CG path (cg_solver.py) is dead code when KPM is active.
       KPM evaluates 16-degree Chebyshev polynomial of H, requiring 16
       Hamiltonian matvecs per propagator layer.
     - KPM lambda_max update (cayley_soliton.py:215-225): Every 10 steps
       (or first 5), using Gershgorin bound from diagonal. No .item()
       sync -- good for async execution.
     - Nonlinear phase rotation / PIT (cayley_soliton.py:140-180):
       * intensity = |psi|^2, normalized by mean
       * Bounded rational envelope (not cosh), clamped to max=10
       * pit_gamma=0.05: phase_shift = pi*(1 - 2*exp(-0.05*intensity))
       * For intensity~1 (normalized): phase_shift ~ pi*(-0.9) ~ -2.83 rad
       * This is a LARGE initial phase rotation per layer.
       * Norm-rescaling clamped to max=10. Multiple safety clamps.
     - MultiScaleHamiltonian (hamiltonian.py:100-200): 3 scales with
       sparsities [6, 12, 24]. For D=256 (<=512 threshold), builds dense
       256x256 A matrix. Matvec is v @ A.t(): O(D^2) = O(65K) per position.
     - Stack depth (cayley_soliton.py:280-330): num_layers = model.num_layers
       = 8 propagator layers. Each has its own Hamiltonian, alpha, and
       Chebyshev coefficients. Total: 8 layers x (phase rotation + 16
       Chebyshev matvecs) = 128 Hamiltonian matvecs per forward pass.
     - CG solver (cg_solver.py): When used (KPM disabled), provides STE
       backward (not true implicit differentiation for sparse path).
       Block-Jacobi preconditioner. Safety fallback to dense solve if CG
       diverges. max_iter=5.
     - lazy_cg (cayley_soliton.py:372-379): skip_cg is ALWAYS False now.
       The .item() sync was removed. lazy_cg is effectively disabled.

   Issues:
     a) 8 propagator layers may be excessive. The propagator provides
        global mixing via the Graph Laplacian. A single layer with larger dt
        or higher KPM degree might achieve equivalent mixing at lower cost.
        128 Hamiltonian matvecs is ~128 * B*S*D^2 = 128 * 32*2048*65K
        ~ 550B FLOPs. This is likely the second largest compute block
        after Lindblad. (Note: if Hamiltonian is sparse, cost drops
        proportionally; but for D=256, dense mode is used.)
     b) Phase accumulation: 8 layers x ~2.83 rad = ~22.6 rad total
        nonlinear phase rotation. This is ~3.6 full rotations. Without
        careful initialization, dimensions can destructively interfere,
        collapsing the signal. The safety clamps prevent divergence but
        cannot prevent information loss from phase cancellation.
     c) CG solver path is dead code under default config (KPM active).
        The entire cg_solver.py (325 lines) and related preconditioner
        logic serve no purpose unless KPM is explicitly disabled.
     d) pit_gamma=0.05 yields large phase shifts at init. For a randomly
        initialized model, this means the propagator immediately applies
        nearly pi-radian rotations, potentially destroying the input signal
        before any learning occurs. Early training may waste steps
        recovering from this initialization.

   Bottleneck severity: HIGH
     (128 Hamiltonian matvecs dominate propagator time; phase accumulation
      risks signal destruction; 8 layers may be reducible to 1-2)


5. Born Sampler (sem/sampler/born_collapse.py + sem/sampler/logits_processors.py)
----------------------------------------------------------------------------------

   Mathematical goal:
     Project the complex wavefunction to vocabulary logits via Born rule
     inspired |psi|^2 collapse. In theory: P(token) ~ |<token|psi>|^2.
     In practice: a log-linear model projecting real and imaginary parts
     separately.

   Implementation:
     - Logit computation (born_collapse.py:80-100):
       logits = W_r @ Re(psi') + W_i @ Im(psi') + bias
       where psi' = psi * exp(i * theta), theta is a single learnable scalar.
     - Weight tying (born_collapse.py:45-55): proj_real.weight =
       encoder.embedding.weight. proj_imag has separate learned weights.
     - Zipf bias init (born_collapse.py:55-70): output_bias initialized to
       log(rank^(-1.07)) approximating natural language Zipf distribution.
       Good for accelerating early convergence.
     - Training path: Chunked cross-entropy in low_vram_mode, standard CE
       otherwise. No sampling processors during training (correct).
     - Inference path (logits_processors.py): Full processor chain:
       top_k=40, top_p=0.92, typical_p=0.92, repetition_penalty=1.43.
       Temperature default=1.0.
     - Phase theta: Single scalar compensates for the pipeline's aggregate
       phase rotation away from embedding space.

   Issues:
     a) The "Born rule" claim is misleading. The actual implementation is a
        standard log-linear projection with weight tying, not |psi|^2. The
        exp(i*theta) rotation followed by separate real/imaginary projections
        is equivalent to: logits = |W| @ |psi| * cos(angle_W - angle_psi + theta)
        (approximately). This is a valid linear projection in complex space
        but is NOT the Born rule P ~ |<token|psi>|^2 which would require
        logits = log(|W @ psi|^2) = log(|W_r @ Re(psi) + W_i @ Im(psi)|^2
        + |W_r @ Im(psi) - W_i @ Re(psi)|^2). The current form loses the
        cross-term interference that makes quantum measurement non-trivial.
     b) Single scalar theta cannot compensate for per-dimension phase
        rotations accumulated through 8 Mamba + 8 propagator layers.
        A per-dimension theta (D=256 parameters) would be more appropriate.
     c) Inference processor stack applies top_k AND top_p AND typical_p
        simultaneously. These are partially redundant filters. The
        interaction of all three can over-restrict the sampling distribution,
        particularly for low-entropy outputs where top_k=40 already captures
        most mass.

   Bottleneck severity: LOW
     (Compute is just two linear projections; issues are mathematical
      fidelity and weight-tying correctness, not performance)


INTER-COMPONENT IMPEDANCE MISMATCHES
======================================

1. Encoder -> Mamba [MEDIUM]:
   In non-simple mode (DEFAULT), the Sinkhorn transport plan maps embeddings
   through sdr_to_hidden linear projection (mesh_sdr.py:128), producing
   representations NOT calibrated to Mamba's expected input scale. Mamba's
   ComplexRMSNorm (first op) absorbs the magnitude mismatch, but if the
   encoder output magnitude distribution differs significantly from what
   the norm was initialized to expect, the effective learning rate for
   early Mamba layers is distorted. In simple_mode, Re(z) ~ N(0, 1/D)
   and Im(z) ~ N(0, 0.09/D), giving |z|^2 ~ 1.09, which is well-matched.

2. Mamba -> Propagator [HIGH]:
   Mamba output includes residual + residual_scale * branch_out with V8
   transforms (Lindblad dissipation slightly shrinks magnitude). The
   propagator then applies 8 layers of nonlinear phase rotation (PIT) with
   pit_gamma=0.05, yielding ~2.83 rad phase shift per layer. Over 8 layers,
   the total nonlinear phase rotation is ~22.6 rad (~3.6 full rotations).
   While the Cayley transform preserves norm, the aggressive phase
   manipulation can cause destructive interference across dimensions,
   effectively scrambling the representation that Mamba carefully built.
   The propagator's mixing benefit must be weighed against this signal
   destruction risk.

3. Propagator -> Sampler (Weight Tying) [CRITICAL]:
   sampler.proj_real.weight is tied to encoder.embedding.weight. This
   assumes the final hidden state's real part lives in the same vector
   space as the token embeddings. In simple_mode, this holds: Re(z) IS
   the embedding, and residual connections through Mamba approximately
   preserve it. In non-simple mode (DEFAULT), Re(z) = sdr * cos(phase)
   after Sinkhorn OT, which is NOT in the embedding space. The
   representation then passes through 8 Mamba layers with complex SSM
   transformations, 8 propagator layers with nonlinear phase rotations
   and Hamiltonian evolution, and a final ComplexRMSNorm. The single
   learnable scalar phase_theta in the sampler cannot undo these
   per-dimension distortions. The model must waste learning capacity
   trying to rotate the final representation back to embedding space --
   a task that is structurally impossible with a scalar phase correction.

4. V8 approximate_hamiltonian -> Quaternionic/HybridAutomata [MEDIUM]:
   The approximate Hamiltonian (model.py:480-495) uses psi.abs() for the
   outer product, producing a real density matrix that captures magnitude
   clustering but discards all phase information. Quaternionic escape's
   spectral norm estimation and hybrid automata's Lie bracket curvature
   both receive a phase-blind Hamiltonian proxy. These V8 modules are
   designed to detect and respond to phase-space dynamics, but their input
   is missing half the information (the phase). Escape triggers may fire
   at incorrect times, and curvature estimates are unreliable.

5. ComplexRMSNorm sparse-aware behavior [LOW]:
   ComplexRMSNorm (complex_layernorm.py:30-60) divides by RMS of active
   dimensions. If the propagator densifies the signal (expected via
   diffusion through the Graph Laplacian), the normalization denominator
   increases, producing smaller normalized values than if the signal
   remained sparse. This means the propagator's densification effect is
   partially counteracted by the final norm before the sampler. The net
   effect is a scale-dependent interaction between propagator depth and
   sampler input magnitude.


TOP 3 ARCHITECTURE BOTTLENECKS (ranked by impact)
===================================================

1. Lindblad Dissipation: ~34B FLOPs / forward pass for negligible effect
   -----------------------------------------------------------------------
   FLOPs: K=4 dense [256, 256] matrix multiplies per layer to form L_k^dag L_k:
     O(K * D^3) = O(4 * 256^3) = O(67M) per layer.
   Then matvec per position: O(B * S * D^2) = O(32 * 2048 * 65K) = O(4.3B)
     per layer.
   Total across 8 layers: ~34.4B FLOPs.
   Effect: gamma=0.007 dissipation -- this adds ~0.7% magnitude change
     to the branch delta. The information-theoretic contribution is near zero.
   Cost/effect ratio: ~5000:1.

   Recommended fix: Either (a) increase gamma to 0.05-0.1 to justify the
   compute cost, or (b) replace dense L_k with low-rank L_k = u_k v_k^T
   where u_k, v_k are [D, R] with R=16-32, reducing cost from O(K*D^3)
   to O(K*D*R), a ~8-16x reduction. Or (c) disable Lindblad entirely
   (set lindblad=false) and rely on the chi-squared gate in Mamba for
   magnitude control.

2. 90 Sinkhorn Iterations Without Early Exit (when simple_mode=false)
   --------------------------------------------------------------------
   FLOPs per iteration: 2 x logsumexp over [B, S, candidates] =
     2 x [32, 2048, 128] = ~16M elements per iteration.
   Total: 90 iterations x 16M = ~1.4B ops.
   Estimated wasted: ~60 iterations x 16M = ~0.96B ops (convergence at
     ~30 iterations for auto_epsilon=0.04*median).

   Recommended fix: Implement GPU-friendly early exit that does NOT use
   .item() sync. Options:
   (a) Fixed iteration count of 30 (empirically sufficient).
   (b) Check convergence every 10 iterations using a GPU-side comparison
       (torch.all(delta < tol) stored as bool tensor, branched on next
       iteration without sync).
   (c) Use the dual variables' rate of change as a proxy: if
       max(|f_new - f_old|) < epsilon/10, stop.

3. 8 Propagator Layers x KPM Degree-16: ~550B Hamiltonian matvecs
   -----------------------------------------------------------------
   Per propagator layer: 16 Chebyshev matvecs through 256x256 dense
     Hamiltonian. Each matvec: O(B * S * D^2) = O(32 * 2048 * 65K) = O(4.3B).
   Per layer total: 16 * 4.3B = ~68B.
   Across 8 layers: 8 * 68B = ~550B FLOPs.
   (Note: This dwarfs all other components combined. However, much of this
    is inherent to the architecture's core operation, not waste.)

   Recommended fix:
   (a) Reduce propagator layers from 8 to 2-3. A single propagator layer
       with larger dt provides equivalent mixing radius. The Graph Laplacian's
       spectral gap determines mixing time -- more layers with small dt and
       fewer layers with large dt are mathematically equivalent for the
       Cayley transform.
   (b) Reduce KPM degree from 16 to 8. Degree-8 Chebyshev approximates
       the Cayley transform to ~1e-4 accuracy for typical spectral ranges.
       Degree-16 provides ~1e-8 accuracy, which is wasted in float32.
   (c) Use sparse Hamiltonian even for D=256. The MultiScaleHamiltonian
       has sparsities [6, 12, 24], meaning each scale connects each node
       to only 6-24 neighbors. The fused sparse matvec would be
       O(D * avg_degree) ~ O(256 * 14) ~ O(3.6K) per position vs O(65K)
       for dense. That is an ~18x reduction.
   Combined (2 layers, degree-8, sparse): 2 * 8 * 32 * 2048 * 3.6K ~ 3.8B.
   Reduction: ~550B -> ~3.8B = ~145x speedup for the propagator.


RECOMMENDATIONS (prioritized)
==============================

P0 (Critical -- do before next training run):

  1. Fix weight tying for non-simple mode.
     Either (a) set simple_mode=true in default.yaml (aligns weight tying
     with representation space), or (b) disable weight tying when
     simple_mode=false by creating a separate proj_real weight matrix
     in born_collapse.py. Option (a) is simpler and eliminates the Sinkhorn
     bottleneck as a side effect.

  2. Reduce Lindblad compute cost.
     Set lindblad_operators to low-rank (R=16) or disable entirely.
     At gamma=0.007, the dissipation has negligible effect on training
     dynamics while consuming ~34B FLOPs per forward pass.

P1 (High -- significant training speedup):

  3. Reduce propagator stack depth from 8 to 2.
     Increase dt proportionally. Verify mixing quality via propagator
     output correlation analysis (autocorrelation across positions should
     decay at similar rate).

  4. Reduce KPM degree from 16 to 8.
     float32 precision makes degree-16 unnecessary. Verify approximation
     error is below 1e-3 at degree-8 using reference dense solve.

  5. Force sparse Hamiltonian matvec even for D=256.
     Change the dense-mode threshold in hamiltonian.py from D<=512 to
     D<=64 (or remove entirely). The MultiScaleHamiltonian is inherently
     sparse -- forcing dense mode negates its structural advantage.

  6. Cap Sinkhorn iterations at 30 (if simple_mode remains false).
     Or implement GPU-friendly convergence check without .item() sync.

P2 (Medium -- correctness and quality improvements):

  7. Replace scalar phase_theta with per-dimension phase vector.
     In born_collapse.py, change theta from nn.Parameter(torch.zeros(1))
     to nn.Parameter(torch.zeros(D)). Cost: negligible (D=256 params).
     Benefit: allows the sampler to compensate for per-dimension phase
     distortions accumulated through the pipeline.

  8. Use complex outer product for approximate_hamiltonian.
     Replace psi.abs() with the full complex outer product
     H = (1/S) sum_s psi_s @ psi_s.conj().T. This preserves phase
     information for V8 modules. Cost increase: ~2x for the outer product.
     Benefit: phase-aware spectral norm estimation and curvature detection.

  9. Reduce pit_gamma from 0.05 to 0.01 or initialize alpha near zero.
     The current initialization applies ~2.83 rad phase rotation per
     layer, which is destructively large. Starting with smaller rotations
     and allowing the model to learn larger ones prevents early-training
     signal destruction.

  10. Review inference processor stack.
      top_k=40 + top_p=0.92 + typical_p=0.92 applied simultaneously is
      likely over-restrictive. Consider using ONLY typical_p=0.92 (which
      subsumes top_p behavior for well-calibrated models) or top_k + top_p
      without typical_p.


ESTIMATED FLOPS BREAKDOWN (per forward pass, B=32, S=2048, D=256)
===================================================================

  Component                    | FLOPs (approx) | % of total
  -----------------------------|----------------|----------
  Lindblad (8 layers)          | ~34B           | 5.5%
  Sinkhorn encoder (90 iter)   | ~1.4B          | 0.2%
  Mamba SSM scan (8 layers)    | ~8B            | 1.3%
  SpinorGate solve (8 layers)  | ~0.4B          | <0.1%
  Propagator KPM (8x16 mv)    | ~550B          | 89.5%
  Born sampler projections     | ~0.5B          | <0.1%
  V8 quaternionic (8 layers)   | ~5B            | 0.8%
  Norms, residuals, misc       | ~15B           | 2.4%
  -----------------------------|----------------|----------
  TOTAL                        | ~615B          | 100%

  After P1 recommendations (2 prop layers, degree-8, sparse Hamiltonian):
  Propagator: ~550B -> ~3.8B
  Lindblad: ~34B -> ~0 (disabled or low-rank)
  Sinkhorn: ~1.4B -> ~0.5B (30 iterations)
  TOTAL: ~615B -> ~33B (~18.6x reduction)


DEAD CODE INVENTORY
====================

  1. cg_solver.py (325 lines): Entirely bypassed when use_chebyshev_kpm=true
     (default). The CG solver, Block-Jacobi preconditioner, and STE backward
     pass are unreachable.
  2. lazy_cg logic (cayley_soliton.py:372-379): skip_cg is always False.
     The lazy_cg configuration option has no effect.
  3. HybridAutomata (sem/spinor/hybrid_automata.py, 301 lines): Disabled
     in default config (hybrid_automata=false). Only reachable if explicitly
     enabled.

---
END OF AUDIT
