"""
preprocessing.py - Step text construction for TRACEGUARD.

CRITICAL: Only behavioural fields enter the model.
The following are NEVER included:
  label, attack_family, attack_subtype, injection, injection_step,
  precursor_type, precursor_step, deviation_step, attack_success,
  counterfactual_group_id, hijack_subtype
"""

from src.config import FORBIDDEN_INPUT_FIELDS


def build_step_text(step: dict, user_goal: str, step_number: int) -> str:
    """
    Construct the textual representation for a single step.
    Schema (from spec section 7):

      ORIGINAL USER GOAL:
      {user_goal}

      CURRENT STEP:
      {step_number}

      ACTION:
      {action}

      TOOL:
      {tool}

      TOOL INPUT:
      {tool_input}

      TOOL OBSERVATION:
      {tool_observation}

      STATE:
      {state}
    """
    action       = step.get("action", "") or ""
    tool         = step.get("tool", "") or ""
    tool_input   = step.get("tool_input", "") or ""
    tool_obs     = step.get("tool_observation", "") or ""
    state        = step.get("state", "") or ""

    text = (
        f"ORIGINAL USER GOAL:\n{user_goal}\n\n"
        f"CURRENT STEP:\n{step_number}\n\n"
        f"ACTION:\n{action}\n\n"
        f"TOOL:\n{tool}\n\n"
        f"TOOL INPUT:\n{tool_input}\n\n"
        f"TOOL OBSERVATION:\n{tool_obs}\n\n"
        f"STATE:\n{state}"
    )
    return text


def build_trajectory_texts(traj: dict) -> list[str]:
    """
    Return list of 6 step texts for a trajectory.
    Performs leakage assertion before returning.
    """
    user_goal = traj.get("user_goal", "")
    texts = []
    for step in traj["steps"]:
        txt = build_step_text(step, user_goal, step["step"])
        _assert_no_leakage(txt, traj["trajectory_id"])
        texts.append(txt)
    return texts


def _assert_no_leakage(text: str, traj_id: str):
    """Raise if any forbidden literal string appears in the step text."""
    for field in FORBIDDEN_INPUT_FIELDS:
        # Only check actual label names that would be interpretable by the model
        if field in ["BENIGN", "INJECTION_RESISTED", "HIJACKED"]:
            if field in text:
                raise AssertionError(
                    f"STOP: Forbidden label string '{field}' found in step text "
                    f"for trajectory {traj_id}."
                )
