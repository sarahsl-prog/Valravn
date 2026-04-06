from __future__ import annotations

import pytest

from valravn.training.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(n_pass=3, n_reject=2)
    buf.add_failure("case-1", {"input": "foo"})
    result = buf.sample(5)
    assert len(result) == 1
    assert result[0] == {"input": "foo"}


def test_replay_buffer_graduation():
    buf = ReplayBuffer(n_pass=3, n_reject=2)
    buf.add_failure("case-1", {"input": "foo"})
    # Two successes — not yet graduated
    buf.record_outcome("case-1", success=True)
    buf.record_outcome("case-1", success=True)
    assert len(buf.sample(10)) == 1
    # Third consecutive success — should graduate (remove from buffer)
    buf.record_outcome("case-1", success=True)
    assert len(buf.sample(10)) == 0


def test_replay_buffer_consecutive_reset():
    buf = ReplayBuffer(n_pass=3, n_reject=2)
    buf.add_failure("case-1", {"input": "bar"})
    # Two successes...
    buf.record_outcome("case-1", success=True)
    buf.record_outcome("case-1", success=True)
    # ...then a failure resets the pass count
    buf.record_outcome("case-1", success=False)
    # Now need another 3 consecutive successes to graduate
    buf.record_outcome("case-1", success=True)
    buf.record_outcome("case-1", success=True)
    assert len(buf.sample(10)) == 1  # Still present — only 2 passes since reset
    buf.record_outcome("case-1", success=True)
    assert len(buf.sample(10)) == 0  # Now graduated


def test_replay_buffer_nonexistent_outcome_is_noop():
    buf = ReplayBuffer(n_pass=3, n_reject=2)
    # Should not raise
    buf.record_outcome("no-such-case", success=True)
    buf.record_outcome("no-such-case", success=False)
    assert len(buf.sample(10)) == 0


def test_replay_buffer_sample_with_empty_buffer():
    buf = ReplayBuffer()
    assert buf.sample(5) == []


def test_replay_buffer_save_and_load(tmp_path):
    buf = ReplayBuffer(n_pass=4, n_reject=3)
    buf.add_failure("case-1", {"input": "alpha"})
    buf.add_failure("case-2", {"input": "beta"})
    buf.record_outcome("case-1", success=True)

    save_path = tmp_path / "replay_buffer.json"
    buf.save(save_path)

    loaded = ReplayBuffer.load(save_path)
    assert loaded.n_pass == 4
    assert loaded.n_reject == 3
    assert set(loaded.buffer.keys()) == {"case-1", "case-2"}
    assert loaded.buffer["case-1"]["passes"] == 1
    assert loaded.buffer["case-2"]["passes"] == 0
    assert loaded.buffer["case-2"]["fails"] == 1
