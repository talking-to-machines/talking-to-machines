"""
Tests for storage modules (Phase 13).

Covers:
  - EventLogger: append mode, JSONL format, thread-safe writes
  - Guardrails: fast-response and repeated-response anomaly detection
  - Checkpointer: save/load round-trip (session JSON)
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from talkingtomachines.storage.event_log import EventLogger
from talkingtomachines.orchestrator.guardrails import Guardrails, AnomalyFlag


# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------


def test_event_logger_creates_file(tmp_path):
    log_path = tmp_path / "traces.jsonl"
    logger = EventLogger(log_path)
    logger.log("session_start", run_id="r1", session_id="s1")
    assert log_path.exists()


def test_event_logger_appends_valid_jsonl(tmp_path):
    log_path = tmp_path / "traces.jsonl"
    logger = EventLogger(log_path)
    logger.log(
        "llm_call",
        run_id="r1",
        session_id="s1",
        group_id="g1",
        agent_instance_id="ai1",
        data={"prompt_type": "DISCUSSION"},
    )
    logger.log(
        "llm_call",
        run_id="r1",
        session_id="s1",
        group_id="g1",
        agent_instance_id="ai2",
        data={"prompt_type": "DISCUSSION"},
    )

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert record["event_type"] == "llm_call"
        assert "timestamp" in record


def test_event_logger_flat_data_merge(tmp_path):
    """data dict should be merged flat into the top-level JSONL record."""
    log_path = tmp_path / "traces.jsonl"
    logger = EventLogger(log_path)
    logger.log(
        "llm_call", run_id="r1", data={"prompt_type": "DISCUSSION", "latency_ms": 123.4}
    )

    record = json.loads(log_path.read_text().strip())
    assert record["prompt_type"] == "DISCUSSION"
    assert record["latency_ms"] == 123.4


def test_event_logger_creates_parent_dirs(tmp_path):
    log_path = tmp_path / "deep" / "nested" / "traces.jsonl"
    logger = EventLogger(log_path)
    logger.log("session_start")
    assert log_path.exists()


def test_event_logger_thread_safe(tmp_path):
    """100 concurrent writes should produce exactly 100 valid JSONL lines."""
    log_path = tmp_path / "traces.jsonl"
    logger = EventLogger(log_path)
    errors = []

    def writer(i):
        try:
            logger.log("llm_call", run_id=f"r{i}", data={"idx": i})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 100
    indices = {json.loads(ln)["idx"] for ln in lines}
    assert indices == set(range(100))


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_guardrails_no_anomaly_normal_response():
    g = Guardrails()
    flags = g.check_response(
        turn_id="t1",
        agent_instance_id="a1",
        content="Hello, I choose 5.",
        latency_ms=500.0,
    )
    assert flags == []
    assert g.anomaly_flags == []


def test_guardrails_fast_response_flag():
    g = Guardrails()
    flags = g.check_response(
        turn_id="t1",
        agent_instance_id="a1",
        content="Fast!",
        latency_ms=10.0,  # below MIN_LATENCY_MS=50
    )
    assert "suspiciously_fast" in flags
    assert any(f.flag_type == "suspiciously_fast" for f in g.anomaly_flags)


def test_guardrails_zero_latency_not_flagged():
    """Latency == 0 (e.g. mocked call) should not trigger fast-response flag."""
    g = Guardrails()
    flags = g.check_response(
        turn_id="t1",
        agent_instance_id="a1",
        content="Hi",
        latency_ms=0.0,
    )
    assert "suspiciously_fast" not in flags


def test_guardrails_repeated_response_flag():
    g = Guardrails()
    content = "I always say the same thing."
    for i in range(Guardrails.REPEAT_WINDOW):
        flags = g.check_response(
            turn_id=f"t{i}",
            agent_instance_id="a1",
            content=content,
            latency_ms=300.0,
        )
    assert "repeated_response" in flags


def test_guardrails_repeated_response_different_agents_independent():
    """Repeated-response window is per agent."""
    g = Guardrails()
    content = "Identical."
    # Agent a1 gets REPEAT_WINDOW identical responses
    for i in range(Guardrails.REPEAT_WINDOW):
        g.check_response(f"t{i}", "a1", content, 300.0)
    # Agent a2 gets only 1 — should not be flagged
    flags = g.check_response("t99", "a2", content, 300.0)
    assert "repeated_response" not in flags


def test_guardrails_anomaly_flag_recorded():
    g = Guardrails()
    g.check_response("t1", "a1", "Fast!", latency_ms=5.0)
    assert len(g.anomaly_flags) == 1
    flag = g.anomaly_flags[0]
    assert isinstance(flag, AnomalyFlag)
    assert flag.turn_id == "t1"
    assert flag.flag_type == "suspiciously_fast"


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------


def test_checkpointer_save_and_load(tmp_path):
    from talkingtomachines.orchestrator.checkpointing import Checkpointer
    from talkingtomachines.core.models import Session

    session = Session(
        session_id="s1",
        run_id="r1",
        experiment_id="exp1",
        cep_hash="abc",
    )

    cp = Checkpointer(tmp_path)
    saved_path = cp.save(session)

    # Checkpoint is saved under checkpoints/<session_id>.json
    checkpoint_file = tmp_path / "checkpoints" / "s1.json"
    assert checkpoint_file.exists(), f"Expected checkpoint at {checkpoint_file}"

    loaded, loaded_state = cp.load("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.run_id == "r1"
    assert loaded.experiment_id == "exp1"
    assert loaded_state is None  # No state was passed to save()
