"""Unit tests for Track D calibration and uncertainty."""

import numpy as np
import pytest
import torch

from traceguard_x.experiments.track_d_calibration import (
    TemperatureScaler,
    compute_uncertainty_metrics,
)


def test_temperature_scaling_fitting():
    torch.manual_seed(42)
    scaler = TemperatureScaler()

    # Generate synthetic validation logits with overconfidence
    val_logits = torch.randn(100, 3) * 5.0  # high temperature / overconfident
    val_labels = torch.randint(0, 3, (100,))

    fitted_temp = scaler.fit(val_logits, val_labels)
    assert fitted_temp > 0.0
    assert isinstance(fitted_temp, float)


def test_accuracy_and_ranking_invariance():
    """Temperature scaling must not change the predicted argmax class."""
    scaler = TemperatureScaler()
    scaler.temperature.data = torch.tensor([1.8])

    logits = torch.randn(50, 3)
    raw_preds = logits.argmax(dim=-1).numpy()

    with torch.no_grad():
        calib_logits = scaler(logits)
        calib_preds = calib_logits.argmax(dim=-1).numpy()

    np.testing.assert_array_equal(raw_preds, calib_preds)


def test_uncertainty_metrics_computation():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    # Perfectly confident
    y_prob = np.eye(3)[y_true]

    metrics = compute_uncertainty_metrics(y_true, y_prob)
    assert metrics["brier_score"] == pytest.approx(0.0)
    assert metrics["ece"] == pytest.approx(0.0)
    assert metrics["high_confidence_error_rates"]["error_rate_ge_090"] == pytest.approx(0.0)
