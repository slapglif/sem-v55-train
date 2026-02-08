"""Tests for Born Collapse Sampler and LogitsProcessor chain."""

import torch
import pytest
from sem.sampler.born_collapse import BornCollapseSampler
from sem.sampler.logits_processors import (
    LogitsProcessorList,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
    MinPProcessor,
    TypicalProcessor,
    RepetitionPenaltyProcessor,
    FrequencyPenaltyProcessor,
    PresencePenaltyProcessor,
    NoRepeatNgramProcessor,
    TopAProcessor,
    EpsilonCutoffProcessor,
    EtaCutoffProcessor,
    build_processor_chain,
)
from sem.config import SamplerConfig


class TestBornCollapse:
    """Test the Born Collapse Sampler."""

    def test_output_shape(self):
        sampler = BornCollapseSampler(
            hidden_dim=16, vocab_size=100, temperature=1.0, top_k=50, top_p=0.95
        )
        psi = torch.randn(2, 8, 16, dtype=torch.complex64)
        result = sampler(psi, sample=True)

        assert "logits" in result
        assert "log_probs" in result
        assert "tokens" in result
        assert result["logits"].shape == (2, 8, 100)
        assert result["tokens"].shape == (2, 8)

    def test_probability_normalization(self):
        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=50, top_k=0, top_p=1.0)
        psi = torch.randn(1, 4, 16, dtype=torch.complex64)
        result = sampler(psi, sample=True)

        prob_sums = result["probs"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), (
            f"Probabilities don't sum to 1: {prob_sums}"
        )

    def test_temperature_effect(self):
        sampler_low = BornCollapseSampler(hidden_dim=16, vocab_size=50)
        sampler_high = BornCollapseSampler(hidden_dim=16, vocab_size=50)
        sampler_high.load_state_dict(sampler_low.state_dict())

        psi = torch.randn(1, 1, 16, dtype=torch.complex64)

        result_low = sampler_low(psi, temperature=0.1, top_k=0, top_p=1.0, sample=True)
        result_high = sampler_high(
            psi, temperature=2.0, top_k=0, top_p=1.0, sample=True
        )

        p_low = result_low["probs"]
        p_high = result_high["probs"]

        entropy_low = -(p_low * (p_low + 1e-12).log()).sum(dim=-1)
        entropy_high = -(p_high * (p_high + 1e-12).log()).sum(dim=-1)

        assert entropy_high.mean() > entropy_low.mean(), (
            f"Higher temp should have higher entropy: {entropy_high.mean():.4f} vs {entropy_low.mean():.4f}"
        )

    def test_training_forward(self):
        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=50)
        psi = torch.randn(2, 4, 16, dtype=torch.complex64, requires_grad=True)
        log_probs = sampler.training_forward(psi)

        assert log_probs.shape == (2, 4, 50)
        loss = log_probs.sum()
        loss.backward()
        assert psi.grad is not None

    def test_top_k_filtering(self):
        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=20)
        psi = torch.randn(1, 1, 16, dtype=torch.complex64)
        result = sampler(psi, top_k=5, top_p=1.0, sample=True)

        probs = result["probs"]
        nonzero = (probs > 1e-8).sum(dim=-1)
        assert nonzero.max() <= 5, (
            f"Top-k=5 but {nonzero.max()} tokens have nonzero prob"
        )

    def test_backward_compat_no_processors(self):
        sampler = BornCollapseSampler(
            hidden_dim=16,
            vocab_size=50,
            processors=None,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
        )
        psi = torch.randn(1, 4, 16, dtype=torch.complex64)
        result = sampler(psi, sample=True)
        assert result["tokens"].shape == (1, 4)
        assert result["probs"].sum(dim=-1).allclose(torch.ones(1, 4), atol=1e-5)

    def test_with_processor_chain(self):
        chain = build_processor_chain(
            SamplerConfig(temperature=0.8, top_k=10, top_p=0.9)
        )
        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=50, processors=chain)
        psi = torch.randn(2, 4, 16, dtype=torch.complex64)
        input_ids = torch.randint(0, 50, (2, 20))
        result = sampler(psi, input_ids=input_ids, sample=True)
        assert result["tokens"].shape == (2, 4)
        assert torch.isfinite(result["logits"]).all()

    def test_input_ids_forwarded_to_processors(self):
        chain = build_processor_chain(
            SamplerConfig(repetition_penalty=2.0, top_k=0, top_p=1.0)
        )
        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=50, processors=chain)
        psi = torch.randn(1, 1, 16, dtype=torch.complex64)
        repeated_id = 5
        input_ids = torch.tensor([[repeated_id] * 10])

        result_with_penalty = sampler(psi, input_ids=input_ids, sample=False)
        result_without = sampler(
            psi, input_ids=torch.empty(1, 0, dtype=torch.long), sample=False
        )

        logits_penalized = result_with_penalty["logits"][0, 0, repeated_id]
        logits_base = result_without["logits"][0, 0, repeated_id]
        assert logits_penalized == logits_base, "Penalty only applies when sampling"


class TestLogitsProcessors:
    """Test individual logits processors."""

    def _make_logits(self, V=20, peaked_at=0):
        logits = torch.randn(1, V)
        logits[0, peaked_at] = 10.0
        return logits

    def test_temperature_scales_logits(self):
        logits = torch.tensor([[2.0, 4.0, 6.0]])
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TemperatureProcessor(2.0)(input_ids, logits)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_temperature_identity_at_1(self):
        logits = torch.randn(2, 50)
        input_ids = torch.empty(2, 0, dtype=torch.long)
        result = TemperatureProcessor(1.0)(input_ids, logits)
        assert torch.equal(result, logits)

    def test_temperature_rejects_zero(self):
        with pytest.raises(ValueError):
            TemperatureProcessor(0.0)

    def test_top_k_keeps_k_tokens(self):
        logits = torch.randn(1, 100)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TopKProcessor(5)(input_ids, logits)
        finite_count = torch.isfinite(result).sum()
        assert finite_count <= 5

    def test_top_k_disabled_when_zero(self):
        logits = torch.randn(1, 100)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TopKProcessor(0)(input_ids, logits)
        assert torch.equal(result, logits)

    def test_top_p_concentrates_mass(self):
        logits = self._make_logits(V=50, peaked_at=0)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TopPProcessor(0.5)(input_ids, logits)
        probs = torch.softmax(result, dim=-1)
        nonzero = (probs > 1e-8).sum()
        assert nonzero < 50

    def test_top_p_disabled_at_1(self):
        logits = torch.randn(1, 50)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TopPProcessor(1.0)(input_ids, logits)
        assert torch.equal(result, logits)

    def test_min_p_filters_low_probability(self):
        logits = torch.zeros(1, 10)
        logits[0, 0] = 10.0
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = MinPProcessor(0.1)(input_ids, logits)
        probs = torch.softmax(result, dim=-1)
        nonzero = (probs > 1e-8).sum()
        assert nonzero < 10, f"Min-P should filter some tokens, got {nonzero}"

    def test_min_p_disabled_at_zero(self):
        logits = torch.randn(1, 50)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = MinPProcessor(0.0)(input_ids, logits)
        assert torch.equal(result, logits)

    def test_min_p_always_keeps_top1(self):
        logits = torch.zeros(1, 10)
        logits[0, 3] = 100.0
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = MinPProcessor(0.99)(input_ids, logits)
        assert torch.isfinite(result[0, 3])

    def test_typical_filters_atypical_tokens(self):
        logits = torch.randn(1, 100)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TypicalProcessor(0.5)(input_ids, logits)
        finite_count = torch.isfinite(result).sum()
        assert finite_count < 100

    def test_typical_disabled_at_1(self):
        logits = torch.randn(1, 50)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TypicalProcessor(1.0)(input_ids, logits)
        assert torch.equal(result, logits)

    def test_repetition_penalty_reduces_repeated(self):
        logits = torch.ones(1, 10) * 5.0
        input_ids = torch.tensor([[2, 5, 2, 2]])
        result = RepetitionPenaltyProcessor(2.0)(input_ids, logits.clone())
        assert result[0, 2] < logits[0, 2], "Repeated token should have lower logit"
        assert result[0, 5] < logits[0, 5], "Repeated token should have lower logit"
        assert result[0, 0] == logits[0, 0], "Non-repeated token should be unchanged"

    def test_repetition_penalty_negative_logits(self):
        logits = torch.ones(1, 10) * -3.0
        input_ids = torch.tensor([[2]])
        result = RepetitionPenaltyProcessor(2.0)(input_ids, logits.clone())
        assert result[0, 2] < logits[0, 2], (
            "Negative logit * penalty should be more negative"
        )

    def test_repetition_penalty_identity_at_1(self):
        logits = torch.randn(1, 50)
        input_ids = torch.tensor([[1, 2, 3]])
        result = RepetitionPenaltyProcessor(1.0)(input_ids, logits.clone())
        assert torch.allclose(result, logits)

    def test_frequency_penalty_proportional_to_count(self):
        logits = torch.zeros(1, 10)
        input_ids = torch.tensor([[3, 3, 3, 5]])
        result = FrequencyPenaltyProcessor(1.0)(input_ids, logits.clone())
        assert result[0, 3] < result[0, 5], (
            "Token appearing 3x should be penalized more than 1x"
        )
        assert result[0, 0] == 0.0, "Absent token should be unpenalized"

    def test_presence_penalty_binary(self):
        logits = torch.zeros(1, 10)
        input_ids = torch.tensor([[3, 3, 3, 5]])
        result = PresencePenaltyProcessor(1.0)(input_ids, logits.clone())
        assert result[0, 3] == result[0, 5], (
            "Presence penalty is binary, not count-based"
        )
        assert result[0, 3] < 0.0
        assert result[0, 0] == 0.0

    def test_no_repeat_ngram_bans_continuation(self):
        logits = torch.zeros(1, 10)
        input_ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3]])
        result = NoRepeatNgramProcessor(3)(input_ids, logits.clone())
        assert result[0, 4] == float("-inf"), (
            "Token 4 would complete repeated 3-gram [2,3,4]"
        )

    def test_no_repeat_ngram_allows_novel(self):
        logits = torch.zeros(1, 10)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        result = NoRepeatNgramProcessor(3)(input_ids, logits.clone())
        assert (result == 0.0).all(), "No repeated n-grams, nothing should be banned"

    def test_top_a_filters_low_probability(self):
        logits = torch.zeros(1, 10)
        logits[0, 0] = 10.0
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = TopAProcessor(0.1)(input_ids, logits)
        probs = torch.softmax(result, dim=-1)
        nonzero = (probs > 1e-8).sum()
        assert nonzero < 10

    def test_epsilon_cutoff_removes_low_prob(self):
        logits = torch.zeros(1, 10)
        logits[0, 0] = 10.0
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = EpsilonCutoffProcessor(0.01)(input_ids, logits)
        probs = torch.softmax(result, dim=-1)
        nonzero = (probs > 1e-8).sum()
        assert nonzero < 10

    def test_eta_cutoff_entropy_adaptive(self):
        logits = torch.randn(1, 100)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = EtaCutoffProcessor(0.01)(input_ids, logits)
        finite_count = torch.isfinite(result).sum()
        assert finite_count < 100


class TestProcessorChain:
    """Test processor chain composition and builder."""

    def test_empty_chain_is_identity(self):
        chain = LogitsProcessorList([])
        logits = torch.randn(2, 50)
        input_ids = torch.empty(2, 0, dtype=torch.long)
        result = chain(input_ids, logits)
        assert torch.equal(result, logits)

    def test_chain_applies_in_order(self):
        chain = LogitsProcessorList(
            [
                TopKProcessor(5),
                TopPProcessor(0.9),
            ]
        )
        logits = torch.randn(1, 100)
        input_ids = torch.empty(1, 0, dtype=torch.long)
        result = chain(input_ids, logits)
        finite_count = torch.isfinite(result).sum()
        assert finite_count <= 5

    def test_build_default_chain(self):
        config = SamplerConfig()
        chain = build_processor_chain(config)
        processor_names = [type(p).__name__ for p in chain]
        assert "TopKProcessor" in processor_names
        assert "TopPProcessor" in processor_names
        assert "RepetitionPenaltyProcessor" in processor_names
        assert "TypicalProcessor" in processor_names

    def test_build_chain_with_all_features(self):
        config = SamplerConfig(
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            min_p=0.05,
            typical_p=0.8,
            repetition_penalty=1.1,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            no_repeat_ngram_size=3,
            top_a=0.1,
            epsilon_cutoff=0.001,
            eta_cutoff=0.01,
        )
        chain = build_processor_chain(config)
        processor_names = [type(p).__name__ for p in chain]
        assert len(chain) == 12
        assert processor_names[0] == "RepetitionPenaltyProcessor"
        assert processor_names[1] == "FrequencyPenaltyProcessor"
        assert processor_names[2] == "PresencePenaltyProcessor"
        assert processor_names[3] == "NoRepeatNgramProcessor"
        assert processor_names[4] == "TemperatureProcessor"

    def test_temperature_last_moves_temperature(self):
        config = SamplerConfig(
            temperature=0.5,
            temperature_last=True,
            top_k=10,
            top_p=1.0,
            typical_p=1.0,
            repetition_penalty=1.0,
        )
        chain = build_processor_chain(config)
        processor_names = [type(p).__name__ for p in chain]
        assert processor_names[-1] == "TemperatureProcessor"
        assert processor_names[0] == "TopKProcessor"

    def test_all_disabled_produces_empty_chain(self):
        config = SamplerConfig(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            typical_p=1.0,
            repetition_penalty=1.0,
        )
        chain = build_processor_chain(config)
        assert len(chain) == 0

    def test_chain_with_repetition_needs_input_ids(self):
        config = SamplerConfig(repetition_penalty=1.5, top_k=0, top_p=1.0)
        chain = build_processor_chain(config)
        logits = torch.ones(1, 20) * 5.0
        input_ids = torch.tensor([[3, 3, 3]])
        result = chain(input_ids, logits.clone())
        assert result[0, 3] < 5.0

    def test_sampler_probs_sum_to_one_with_chain(self):
        config = SamplerConfig(
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            min_p=0.01,
            repetition_penalty=1.2,
        )
        chain = build_processor_chain(config)
        sampler = BornCollapseSampler(hidden_dim=16, vocab_size=100, processors=chain)
        psi = torch.randn(2, 4, 16, dtype=torch.complex64)
        input_ids = torch.randint(0, 100, (2, 30))
        result = sampler(psi, input_ids=input_ids, sample=True)
        prob_sums = result["probs"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
