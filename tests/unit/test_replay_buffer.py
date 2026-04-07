from __future__ import annotations

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


def test_replay_buffer_rejection():
    """Case is ejected after n_reject consecutive failures."""
    buf = ReplayBuffer(n_pass=3, n_reject=2)
    buf.add_failure("case-1", {"input": "foo"})  # fails=1
    buf.record_outcome("case-1", success=False)   # fails=2 → ejected
    assert "case-1" not in buf.buffer
    assert buf.sample(10) == []


def test_save_and_load(tmp_path):
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


def test_replay_buffer_archives_rejected_cases(tmp_path):
    """Q1: Cases rejected after n_reject failures are archived, not just deleted."""
    archive_path = tmp_path / "abandoned_cases.jsonl"
    buf = ReplayBuffer(n_pass=3, n_reject=2, archive_path=archive_path)
    
    buf.add_failure("case-doomed", {"input": "problematic evidence", "case_id": "case-doomed"})
    assert "case-doomed" in buf.buffer
    
    # Second failure triggers rejection (fails=2 >= n_reject=2)
    buf.record_outcome("case-doomed", success=False)
    
    # Case should be removed from buffer
    assert "case-doomed" not in buf.buffer
    
    # Case should be archived
    assert archive_path.exists()
    with open(archive_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    
    import json
    record = json.loads(lines[0])
    assert record["case_id"] == "case-doomed"
    assert record["case_data"]["input"] == "problematic evidence"
    assert record["final_fails"] == 2
    assert record["final_passes"] == 0
    assert record["n_reject_threshold"] == 2
    assert "archived_at" in record


def test_replay_buffer_archive_roundtrip(tmp_path):
    """Archive path is persisted and restored via save/load."""
    archive_path = tmp_path / "archive.jsonl"
    buf = ReplayBuffer(n_pass=3, n_reject=2, archive_path=archive_path)
    buf.add_failure("case-1", {"input": "test"})
    
    state_path = tmp_path / "replay_state.json"
    buf.save(state_path)
    
    loaded = ReplayBuffer.load(state_path)
    assert loaded.archive_path == archive_path


def test_replay_buffer_no_archive_path_skips_archiving(tmp_path):
    """Without archive_path, rejected cases are silently deleted (legacy behavior)."""
    buf = ReplayBuffer(n_pass=3, n_reject=2, archive_path=None)
    
    buf.add_failure("case-1", {"input": "foo"})
    buf.record_outcome("case-1", success=False)  # Rejection triggered
    
    # Case removed, no crash
    assert "case-1" not in buf.buffer


def test_replay_buffer_appends_to_existing_archive(tmp_path):
    """Multiple rejected cases are appended to same archive file."""
    archive_path = tmp_path / "abandoned.jsonl"
    buf = ReplayBuffer(n_pass=3, n_reject=2, archive_path=archive_path)
    
    # Reject first case
    buf.add_failure("case-a", {"input": "a"})
    buf.record_outcome("case-a", success=False)
    
    # Reject second case
    buf.add_failure("case-b", {"input": "b"})
    buf.record_outcome("case-b", success=False)
    
    import json
    with open(archive_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    
    assert len(lines) == 2
    assert lines[0]["case_id"] == "case-a"
    assert lines[1]["case_id"] == "case-b"
    
    # archived_count incremented
    assert buf.archived_count == 2
