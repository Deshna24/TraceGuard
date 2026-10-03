"""Focused verification for the controlled TRACEGUARD tool layer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACEGUARD_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRACEGUARD_DIR))

from agent.tools import ToolSpec, build_default_registry  # noqa: E402
from runtime.action_gate import ProposedAction  # noqa: E402
from runtime.trajectory import TrajectoryState  # noqa: E402
from src.config import DATA_PATH, MODELS_DIR  # noqa: E402


RESULTS_PATH = SCRIPT_DIR / "p0_tools_results.json"
SPLIT_PATH = TRACEGUARD_DIR / "outputs" / "splits" / "split_seed42.json"
CHECKPOINT_PATH = MODELS_DIR / "lstm_seed42_best.pth"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    hashes_before = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }
    registry, tools = build_default_registry()
    if registry.names() != ("search", "calculator", "database"):
        raise AssertionError(f"Unexpected tool names: {registry.names()}")
    if len(registry.describe()) != 3:
        raise AssertionError("Tool metadata discovery failed")

    search_input = {"query": "capital of Japan"}
    first_search = registry.invoke("search", search_input)
    second_search = registry.invoke("search", search_input)
    if not first_search.success or first_search != second_search:
        raise AssertionError("Search output is not deterministic")
    injection = registry.invoke(
        "search",
        {"query": "research result", "include_injection_fixture": True},
    )
    if not injection.success or "Injected instruction (observation only)" not in injection.output:
        raise AssertionError("Controlled injection fixture was not returned")
    if tools["search"].execution_count != 3:
        raise AssertionError("Search execution count is incorrect")

    calculator = registry.invoke("calculator", {"expression": "2 + 3 * 4"})
    if not calculator.success or calculator.output != 14:
        raise AssertionError("Calculator returned the wrong result")
    invalid_calculator = registry.invoke("calculator", {"expression": "__import__('os').getcwd()"})
    if invalid_calculator.success:
        raise AssertionError("Calculator accepted arbitrary code")

    record = registry.invoke("database", {"operation": "read_record", "record_id": "customer_001"})
    query = registry.invoke("database", {"operation": "query_records", "status": "active"})
    if not record.success or record.output["name"] != "Fictional Ada":
        raise AssertionError("Database read failed")
    if not query.success or len(query.output) != 1:
        raise AssertionError("Database query failed")
    export = registry.invoke("database", {"operation": "export_records"})
    if not export.success or export.output["exported"] is not False:
        raise AssertionError("Sandbox export was not represented safely")
    if tools["database"].execution_count != 3:
        raise AssertionError("Database execution count is incorrect")
    proposed_export = ProposedAction(
        action="Export fictional records in the sandbox",
        tool="database",
        tool_input={"operation": "export_records"},
    )
    if registry.resolve(proposed_export.tool).name != proposed_export.tool:
        raise AssertionError("Tool action is not compatible with the gate action shape")

    observation = injection.as_observation()
    trajectory = TrajectoryState("Find a research result")
    trajectory.add_step(
        {
            "step": 1,
            "action": "Search for a research result",
            "tool": "search",
            "tool_input": {"query": "research result", "include_injection_fixture": True},
            "tool_observation": observation,
            "state": "Received a search observation",
        }
    )
    if trajectory.get_current_step()["tool_observation"] != observation:
        raise AssertionError("Injection observation was not preserved by trajectory state")

    calls_before_reset = tools["database"].execution_count
    tools["database"].reset()
    if tools["database"].execution_count != 0 or calls_before_reset != 3:
        raise AssertionError("Tool execution reset failed")
    try:
        registry.invoke("unknown", {})
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown tool did not fail clearly")
    try:
        registry.register(ToolSpec("search", "duplicate", {}, lambda _: first_search))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate tool name was accepted")

    hashes_after = {
        "dataset": _hash_file(DATA_PATH),
        "split": _hash_file(SPLIT_PATH),
        "checkpoint": _hash_file(CHECKPOINT_PATH),
    }
    if hashes_before != hashes_after:
        raise AssertionError("Frozen artifact hash changed during tool verification")

    results = {
        "all_passed": True,
        "module": "traceguard/agent/tools.py",
        "tools": list(registry.names()),
        "selected_high_impact_action": "database.export_records",
        "deterministic_search": True,
        "injection_observation_preserved": True,
        "calculator_rejects_code": True,
        "sandbox_export": True,
        "gate_action_compatible": True,
        "execution_counts": {
            "search": tools["search"].execution_count,
            "calculator": tools["calculator"].execution_count,
            "database_after_reset": tools["database"].execution_count,
        },
        "unknown_tool_rejected": True,
        "duplicate_tool_rejected": True,
        "frozen_artifacts_unchanged": True,
        "hashes": {"before": hashes_before, "after": hashes_after},
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Tool verification passed. Results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
