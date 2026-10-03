import pytest

from traceguard.demo.app import SCENARIO_LOGS, THRESHOLD, load_all_runtime_logs


def test_all_recorded_scenarios_preserve_evidence():
    records = load_all_runtime_logs()
    assert set(records) == set(SCENARIO_LOGS)
    for scenario, record in records.items():
        assert record["scenario"] == scenario
        assert [step["step"] for step in record["trajectory"]] == list(
            range(1, len(record["trajectory"]) + 1)
        )
        for evaluation in record["detector_evaluations"]:
            assert set(evaluation["class_probabilities"]) == {
                "BENIGN",
                "INJECTION_RESISTED",
                "HIJACKED",
            }
            assert evaluation["threshold"] == THRESHOLD
            assert evaluation["hijack_probability"] == evaluation["class_probabilities"]["HIJACKED"]
            assert evaluation["decision"] == (
                "BLOCK"
                if evaluation["hijack_probability"] >= THRESHOLD
                else "ALLOW"
            )
            assert evaluation["pre_action"] is True


def test_hijacked_record_blocks_export_without_database_execution():
    record = load_all_runtime_logs()["HIJACKED"]
    assert record["first_threshold_crossing_step"] == 2
    assert record["actions_blocked"] is True
    assert record["blocked_action"]["tool"] == "database"
    assert record["blocked_action"]["tool_input"]["operation"] == "export_records"
    assert record["tool_execution_counts"]["database"] == 0


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        from traceguard.demo.app import load_runtime_log

        load_runtime_log("UNKNOWN")
