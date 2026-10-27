"""Unit tests for traceguard_x.common: data integrity and metric correctness."""

import numpy as np
import pytest

from traceguard_x.common import data as xdata
from traceguard_x.common import metrics as xmetrics


@pytest.fixture(scope="module")
def dataset():
    return xdata.load_dataset()


@pytest.fixture(scope="module")
def split_parts(dataset):
    split_info = xdata.load_split(42)
    return xdata.split_trajectories(dataset, split_info), split_info


class TestDataLoading:
    def test_dataset_size_and_labels(self, dataset):
        assert len(dataset) == 600
        labels = [t["label"] for t in dataset]
        for cls in ("BENIGN", "INJECTION_RESISTED", "HIJACKED"):
            assert labels.count(cls) == 200

    def test_split_partition_sizes(self, split_parts):
        parts, _ = split_parts
        assert len(parts["train"]) == 420
        assert len(parts["val"]) == 90
        assert len(parts["test"]) == 90

    def test_no_group_leakage(self, dataset, split_parts):
        _, split_info = split_parts
        report = xdata.verify_no_group_leakage(dataset, split_info)
        assert report["ok"], report

    def test_split_rejects_leaked_split(self, dataset):
        # A split reusing one group across partitions must be rejected.
        bad = {
            "train_group_ids": ["CF_0001", "CF_0002"],
            "val_group_ids": ["CF_0003"],
            "test_group_ids": ["CF_0001"],
        }
        with pytest.raises(ValueError):
            xdata.split_trajectories(dataset, bad)

    def test_embeddings_shape(self):
        emb = xdata.load_embeddings()
        sample = next(iter(emb.values()))
        assert sample.shape == (6, 384)
        assert len(emb) == 600


class TestMetrics:
    def test_full_classification_metrics_perfect(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_prob = np.eye(3)[y_true]
        y_pred = list(y_true)
        m = xmetrics.full_classification_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_class_ordering_in_per_class(self):
        # BENIGN=0, INJECTION_RESISTED=1, HIJACKED=2 must keep canonical order.
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        m = xmetrics.full_classification_metrics(y_true, y_pred, np.eye(3))
        assert list(m["per_class"].keys()) == xmetrics.CLASS_NAMES

    def test_false_positive_rates(self):
        # 2 benign (1 hijack-flagged), 1 resisted (clean), 2 hijacked.
        y_true = [0, 0, 1, 2, 2]
        y_pred = [2, 0, 1, 2, 2]
        fpr = xmetrics.false_positive_rates(y_true, y_pred)
        assert fpr["BENIGN"] == pytest.approx(0.5)
        assert fpr["INJECTION_RESISTED"] == pytest.approx(0.0)

    def test_detection_records_and_latency(self):
        rows = [
            # Hijacked traj detected at prefix 2, deviation at step 4.
            {"trajectory_id": "T_H", "true_label": "HIJACKED", "prefix_length": 1,
             "p_hijacked": 0.1, "deviation_step": 4, "hijack_subtype": "B"},
            {"trajectory_id": "T_H", "true_label": "HIJACKED", "prefix_length": 2,
             "p_hijacked": 0.7, "deviation_step": 4, "hijack_subtype": "B"},
            # Hijacked traj never detected.
            {"trajectory_id": "T_N", "true_label": "HIJACKED", "prefix_length": 1,
             "p_hijacked": 0.2, "deviation_step": 3, "hijack_subtype": "A"},
            # Benign traj with a false alarm at prefix 3.
            {"trajectory_id": "T_B", "true_label": "BENIGN", "prefix_length": 3,
             "p_hijacked": 0.9},
        ]
        records = xmetrics.detection_records_from_prefixes(rows)
        by_id = {r["trajectory_id"]: r for r in records}
        assert by_id["T_H"]["detection_step"] == 2
        assert by_id["T_H"]["latency"] == -2
        assert by_id["T_H"]["pre_action"] is True
        assert by_id["T_N"]["detected"] is False
        stats = xmetrics.latency_statistics(records)
        assert stats["n_hijacked"] == 2
        assert stats["detection_rate"] == pytest.approx(0.5)
        assert stats["pre_action_detection_rate"] == pytest.approx(0.5)
        assert stats["pct_never_detected"] == pytest.approx(0.5)

    def test_threshold_is_inclusive(self):
        rows = [{"trajectory_id": "T", "true_label": "HIJACKED",
                 "prefix_length": 1, "p_hijacked": 0.5, "deviation_step": 2}]
        records = xmetrics.detection_records_from_prefixes(rows, threshold=0.5)
        assert records[0]["detected"] is True

    def test_brier_score(self):
        y_true = [0, 1]
        y_prob = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert xmetrics.brier_score(y_true, y_prob) == pytest.approx(0.0)
        uniform = np.full((2, 3), 1 / 3)
        expected = 2 * ((1 - 1 / 3) ** 2 + 2 * (1 / 3) ** 2) / 2
        assert xmetrics.brier_score(y_true, uniform) == pytest.approx(expected)

    def test_ece_perfectly_calibrated_and_confident(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_prob = np.eye(3)[y_true]
        out = xmetrics.expected_calibration_error(y_true, y_prob)
        assert out["ece"] == pytest.approx(0.0, abs=1e-9)

    def test_classwise_ece_keys(self):
        y_true = [0, 1, 2]
        y_prob = np.eye(3)
        cw = xmetrics.classwise_ece(y_true, y_prob)
        assert set(cw.keys()) == set(xmetrics.CLASS_NAMES)
