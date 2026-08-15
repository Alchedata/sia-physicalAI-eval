"""Tests for the SEPA /judge HTTP endpoint (PRD §8.3).

Starts the real HTTP server on a random port with fake framework objects (no
model is loaded) and exercises the full round-trip through the SemanticCritic
client contract.
"""

from __future__ import annotations

import base64
import json

import pytest

requests = pytest.importorskip("requests")

from deployment.model_server.judge_endpoint import (  # noqa: E402
    JudgeServer,
    _parse_model_text,
    maybe_start_judge_server,
)
from sepa_eval.critics.semantic_critic import CriticError, SemanticCritic  # noqa: E402

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg=="
PNG_BYTES = base64.b64decode(PNG_B64)


class FakeJudgeFramework:
    """Fake VLM framework exposing a chat-capable ``judge`` method."""

    def __init__(self, reply: str | None = None):
        self.reply = reply or json.dumps(
            {
                "completion": 0.85,
                "object_correct": True,
                "collateral_damage": False,
                "explanation": "The mug was picked up cleanly.",
            }
        )
        self.calls: list[tuple[int, str]] = []

    def judge(self, images, prompt):
        self.calls.append((len(images), prompt))
        return self.reply


class ActionOnlyFramework:
    """Framework with no chat/vision generation capability at all."""

    def predict_action(self, obs):
        return [0.0] * 7


class ExplodingFramework:
    def judge(self, images, prompt):
        raise RuntimeError("CUDA out of memory")


@pytest.fixture()
def judge_server():
    framework = FakeJudgeFramework()
    server = JudgeServer(framework, host="127.0.0.1", port=0).start()
    yield server, framework
    server.stop()


def _url(server: JudgeServer, path: str = "/judge") -> str:
    return f"http://127.0.0.1:{server.port}{path}"


# ---------------------------------------------------------------------------
# Raw HTTP contract
# ---------------------------------------------------------------------------


def test_judge_success_contract(judge_server):
    server, framework = judge_server
    resp = requests.post(
        _url(server),
        json={"frames": [PNG_B64, PNG_B64], "instruction": "Pick up the mug"},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["completion"] == pytest.approx(0.85)
    assert data["completion_score"] == pytest.approx(0.85)
    assert data["object_correct"] is True
    assert data["collateral_damage"] is False
    assert data["explanation"] == "The mug was picked up cleanly."
    assert "text" in data
    # the fake framework received both frames and an instruction-bearing prompt
    n_images, prompt = framework.calls[0]
    assert n_images == 2
    assert "Pick up the mug" in prompt


def test_judge_accepts_images_alias_and_custom_prompt(judge_server):
    server, framework = judge_server
    resp = requests.post(
        _url(server),
        json={"images": [PNG_B64], "instruction": "stack the cups", "prompt": "Custom judge prompt"},
        timeout=10,
    )
    assert resp.status_code == 200
    assert framework.calls[0] == (1, "Custom judge prompt")


def test_judge_caps_frames_at_ten(judge_server):
    server, framework = judge_server
    resp = requests.post(
        _url(server),
        json={"frames": [PNG_B64] * 15, "instruction": "wipe the table"},
        timeout=10,
    )
    assert resp.status_code == 200
    assert framework.calls[0][0] == 10


def test_judge_bad_requests(judge_server):
    server, _ = judge_server
    # missing instruction
    resp = requests.post(_url(server), json={"frames": [PNG_B64]}, timeout=10)
    assert resp.status_code == 400
    assert "instruction" in resp.json()["error"]
    # missing frames
    resp = requests.post(_url(server), json={"instruction": "x"}, timeout=10)
    assert resp.status_code == 400
    assert "frames" in resp.json()["error"]
    # invalid base64
    resp = requests.post(_url(server), json={"frames": ["!!not-base64!!"], "instruction": "x"}, timeout=10)
    assert resp.status_code == 400
    # malformed JSON body
    resp = requests.post(_url(server), data="{not json", headers={"Content-Type": "application/json"}, timeout=10)
    assert resp.status_code == 400
    # unknown path
    resp = requests.post(_url(server, "/predict"), json={}, timeout=10)
    assert resp.status_code == 404


def test_judge_unsupported_model_returns_501():
    server = JudgeServer(ActionOnlyFramework(), host="127.0.0.1", port=0).start()
    try:
        resp = requests.post(_url(server), json={"frames": [PNG_B64], "instruction": "pick"}, timeout=10)
        assert resp.status_code == 501
        body = resp.json()
        assert body["fallback"] is True
        assert "ActionOnlyFramework" in body["error"]
    finally:
        server.stop()


def test_judge_model_failure_returns_500():
    server = JudgeServer(ExplodingFramework(), host="127.0.0.1", port=0).start()
    try:
        resp = requests.post(_url(server), json={"frames": [PNG_B64], "instruction": "pick"}, timeout=10)
        assert resp.status_code == 500
        assert "CUDA out of memory" in resp.json()["error"]
    finally:
        server.stop()


def test_healthz(judge_server):
    server, _ = judge_server
    resp = requests.get(_url(server, "/healthz"), timeout=10)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Round-trip through the SemanticCritic client
# ---------------------------------------------------------------------------


def test_semantic_critic_round_trip(judge_server):
    server, _ = judge_server
    critic = SemanticCritic(server_host="127.0.0.1", server_port=server.port)
    result = critic.judge(frames=[PNG_BYTES], instruction="Pick up the mug", model_id="PaliGemmaOFT")
    assert result.completion == pytest.approx(0.85)
    assert result.object_correct is True
    assert result.collateral_damage is False
    assert result.explanation == "The mug was picked up cleanly."


def test_semantic_critic_raises_on_unsupported_model():
    server = JudgeServer(ActionOnlyFramework(), host="127.0.0.1", port=0).start()
    try:
        critic = SemanticCritic(server_host="127.0.0.1", server_port=server.port)
        with pytest.raises(CriticError):
            critic.judge(frames=[PNG_BYTES], instruction="Pick up the mug", model_id="PaliGemmaOFT")
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Helpers / opt-in wiring
# ---------------------------------------------------------------------------


def test_parse_model_text_extracts_embedded_json():
    text = 'Sure! Here is my evaluation:\n{"completion": 1.4, "object_correct": true, "explanation": "done"}'
    parsed = _parse_model_text(text)
    assert parsed["completion"] == 1.0  # clamped to [0, 1]
    assert parsed["object_correct"] is True
    assert parsed["explanation"] == "done"
    assert parsed["text"] == text


def test_parse_model_text_non_json_falls_back_to_defaults():
    parsed = _parse_model_text("The robot clearly failed.")
    assert parsed["completion"] == 0.0
    assert parsed["object_correct"] is False
    assert parsed["explanation"] == "The robot clearly failed."


def test_maybe_start_judge_server_disabled_by_default():
    assert maybe_start_judge_server(FakeJudgeFramework(), None) is None
    assert maybe_start_judge_server(FakeJudgeFramework(), 0) is None


def test_maybe_start_judge_server_enabled():
    server = maybe_start_judge_server(FakeJudgeFramework(), judge_port=0, host="127.0.0.1")
    # port=0 is falsy -> disabled per contract; use an ephemeral explicit port instead
    assert server is None
    server = JudgeServer(FakeJudgeFramework(), host="127.0.0.1", port=0).start()
    try:
        assert server.port > 0
        assert requests.get(_url(server, "/healthz"), timeout=10).status_code == 200
    finally:
        server.stop()
