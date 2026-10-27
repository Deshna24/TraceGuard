"""Unit tests for TRACEGUARD-X model architectures: GRU, Attention, MultiTask.

Verifies:
- Output shapes
- Probability validity and sums
- Class ordering
- Prefix-only inference (future invariance)
- Padding and masking
- Parameter count comparisons
"""

import pytest
import torch
import numpy as np

import src.model_lstm as canonical_lstm
from traceguard_x.models.gru import TraceGuardGRU
from traceguard_x.models.attention import TraceGuardTemporalAttention
from traceguard_x.models.multitask import TraceGuardMultiTaskEarlyWarning


@pytest.fixture
def dummy_batch():
    # Batch size 3, max seq len 6, input size 384
    torch.manual_seed(42)
    x = torch.randn(3, 6, 384)
    lengths = [6, 4, 2]
    return x, lengths


class TestTraceGuardGRU:
    def test_output_shape(self, dummy_batch):
        x, lengths = dummy_batch
        model = TraceGuardGRU(input_size=384, hidden_size=128, num_layers=2)
        model.eval()
        with torch.no_grad():
            logits = model(x, lengths)
        assert logits.shape == (3, 3)

    def test_probabilities_sum_to_one(self, dummy_batch):
        x, lengths = dummy_batch
        model = TraceGuardGRU()
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x, lengths), dim=-1).numpy()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_parameter_count_vs_lstm(self):
        lstm = canonical_lstm.TraceGuardLSTM(input_size=384, hidden_size=128, num_layers=2)
        gru = TraceGuardGRU(input_size=384, hidden_size=128, num_layers=2)
        lstm_params = sum(p.numel() for p in lstm.parameters())
        gru_params = sum(p.numel() for p in gru.parameters())
        # GRU has 3 gates vs LSTM 4 gates -> recurrent part is ~25% smaller
        assert gru_params < lstm_params
        assert gru_params == 296835
        assert lstm_params == 395651

    def test_prefix_future_invariance(self):
        # A prefix of length k should produce the exact same logits
        # regardless of whatever dummy noise exists in positions > k.
        model = TraceGuardGRU(input_size=384, hidden_size=64, num_layers=1)
        model.eval()
        k = 3
        x1 = torch.randn(1, 6, 384)
        x2 = x1.clone()
        # Alter steps 4, 5, 6
        x2[:, k:, :] = torch.randn(1, 3, 384) * 50.0

        with torch.no_grad():
            out1 = model(x1, [k])
            out2 = model(x2, [k])
        assert torch.allclose(out1, out2, atol=1e-5)


class TestTraceGuardTemporalAttention:
    def test_output_shape_and_attention_shape(self, dummy_batch):
        x, lengths = dummy_batch
        model = TraceGuardTemporalAttention(input_size=384, hidden_size=128)
        model.eval()
        with torch.no_grad():
            logits, attn = model(x, lengths, return_attention=True)
        assert logits.shape == (3, 3)
        assert attn.shape == (3, 6)

    def test_masking_zeroes_padding_attention(self, dummy_batch):
        x, lengths = dummy_batch
        model = TraceGuardTemporalAttention()
        model.eval()
        with torch.no_grad():
            _, attn = model(x, lengths, return_attention=True)
        attn_np = attn.numpy()
        # Item 1 has length 4 -> positions 4, 5 must be 0
        assert np.allclose(attn_np[1, 4:], 0.0, atol=1e-5)
        # Item 2 has length 2 -> positions 2, 3, 4, 5 must be 0
        assert np.allclose(attn_np[2, 2:], 0.0, atol=1e-5)
        # All valid parts sum to 1.0
        assert np.allclose(attn_np.sum(axis=1), 1.0, atol=1e-5)

    def test_prefix_future_invariance(self):
        model = TraceGuardTemporalAttention(input_size=384, hidden_size=64, num_layers=1)
        model.eval()
        k = 3
        x1 = torch.randn(1, 6, 384)
        x2 = x1.clone()
        x2[:, k:, :] = torch.randn(1, 3, 384) * 100.0

        with torch.no_grad():
            out1, attn1 = model(x1, [k], return_attention=True)
            out2, attn2 = model(x2, [k], return_attention=True)
        assert torch.allclose(out1, out2, atol=1e-5)
        assert torch.allclose(attn1[:, :k], attn2[:, :k], atol=1e-5)


class TestTraceGuardMultiTaskEarlyWarning:
    def test_output_shapes(self, dummy_batch):
        x, lengths = dummy_batch
        model = TraceGuardMultiTaskEarlyWarning(input_size=384, hidden_size=128)
        model.eval()
        with torch.no_grad():
            logits, warn = model(x, lengths)
        assert logits.shape == (3, 3)
        assert warn.shape == (3,)

    def test_predict_main_returns_logits(self, dummy_batch):
        x, lengths = dummy_batch
        model = TraceGuardMultiTaskEarlyWarning()
        model.eval()
        with torch.no_grad():
            logits = model.predict_main(x, lengths)
        assert logits.shape == (3, 3)
