"""
Tests for the SEPA-Eval critics package.

Three tests:
  1. test_semantic_critic_judge_200      — 200 response → CriticResult returned
  2. test_semantic_critic_judge_500      — 500 response → CriticError raised
  3. test_circular_bias_routing          — Qwen model_id → fallback path taken
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sepa_eval.critics.semantic_critic import CriticError, CriticResult, SemanticCritic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(status_code: int, json_body: dict | None = None) -> MagicMock:
    """Build a minimal requests.Response mock."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = str(json_body)
    if json_body is not None:
        mock_resp.json.return_value = json_body
    return mock_resp


_VALID_JUDGE_JSON = {
    "completion": 0.9,
    "object_correct": True,
    "collateral_damage": False,
    "explanation": "The robot successfully picked the red cup.",
}


# ---------------------------------------------------------------------------
# Test 1: 200 response → CriticResult
# ---------------------------------------------------------------------------

def test_semantic_critic_judge_200():
    """A 200 response with valid JSON is parsed into a CriticResult."""
    critic = SemanticCritic(default_model="local")
    mock_resp = _make_mock_response(200, _VALID_JUDGE_JSON)

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = critic.judge(frames=[b"\x89PNG\r\n"], instruction="Pick the red cup")

    mock_post.assert_called_once()
    assert isinstance(result, CriticResult)
    assert result.completion == pytest.approx(0.9)
    assert result.object_correct is True
    assert result.collateral_damage is False
    assert "successfully" in result.explanation


# ---------------------------------------------------------------------------
# Test 2: 500 response → CriticError
# ---------------------------------------------------------------------------

def test_semantic_critic_judge_500():
    """A 500 response raises CriticError."""
    critic = SemanticCritic(default_model="local")
    mock_resp = _make_mock_response(500)
    mock_resp.text = "Internal server error"

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(CriticError, match="500"):
            critic.judge(frames=[b"\x89PNG\r\n"], instruction="Pick the red cup")


# ---------------------------------------------------------------------------
# Test 3: circular bias routing
# ---------------------------------------------------------------------------

def test_circular_bias_routing():
    """
    When model_id matches a circular_bias_pattern, _call_gpt4o_mini is used
    and _call_local_judge is NOT called.
    """
    critic = SemanticCritic(
        default_model="local",
        circular_bias_patterns=["Qwen*"],
    )

    expected_result = CriticResult(
        completion=0.75,
        object_correct=True,
        collateral_damage=False,
        explanation="Fallback judge result.",
    )

    with patch.object(critic, "_call_gpt4o_mini", return_value=expected_result) as mock_fallback, \
         patch.object(critic, "_call_local_judge") as mock_local:

        result = critic.judge(
            frames=[b"\x89PNG\r\n"],
            instruction="Pick the blue mug",
            model_id="QwenOFT-v1",
        )

    mock_fallback.assert_called_once()
    mock_local.assert_not_called()
    assert result is expected_result
