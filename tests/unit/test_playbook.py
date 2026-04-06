from __future__ import annotations

from valravn.training.playbook import SecurityPlaybook


def test_playbook_add_entry():
    pb = SecurityPlaybook()
    pb.add_entry("e1", rule="Block port 22", rationale="Reduce attack surface", iteration=1)
    assert "e1" in pb.entries
    entry = pb.entries["e1"]
    assert entry["rule"] == "Block port 22"
    assert entry["rationale"] == "Reduce attack surface"
    assert entry["added_iteration"] == 1


def test_playbook_update_entry():
    pb = SecurityPlaybook()
    pb.add_entry("e1", rule="Block port 22", rationale="Original reason", iteration=0)
    pb.update_entry("e1", rule="Block ports 22 and 23", rationale="Updated reason")
    entry = pb.entries["e1"]
    assert entry["rule"] == "Block ports 22 and 23"
    assert entry["rationale"] == "Updated reason"
    # added_iteration should remain unchanged
    assert entry["added_iteration"] == 0


def test_playbook_delete_entry():
    pb = SecurityPlaybook()
    pb.add_entry("e1", rule="Block port 22", rationale="Reason", iteration=0)
    pb.delete_entry("e1")
    assert "e1" not in pb.entries


def test_playbook_to_prompt_section():
    pb = SecurityPlaybook()
    pb.add_entry("e1", rule="Block port 22", rationale="Reason A", iteration=0)
    pb.add_entry("e2", rule="Enable MFA", rationale="Reason B", iteration=1)
    section = pb.to_prompt_section()
    assert "Block port 22" in section
    assert "Enable MFA" in section
    assert "Reason A" in section
    assert "Reason B" in section
    # Should be markdown list format
    assert "-" in section or "*" in section


def test_playbook_save_and_load(tmp_path):
    pb = SecurityPlaybook(version=3)
    pb.add_entry("e1", rule="Block port 22", rationale="Reason", iteration=2)
    save_path = tmp_path / "playbook.json"
    pb.save(save_path)

    loaded = SecurityPlaybook.load(save_path)
    assert loaded.version == 3
    assert "e1" in loaded.entries
    assert loaded.entries["e1"]["rule"] == "Block port 22"
    assert loaded.entries["e1"]["added_iteration"] == 2


def test_playbook_delete_nonexistent_is_noop():
    pb = SecurityPlaybook()
    # Should not raise
    pb.delete_entry("nonexistent_id")
    assert pb.entries == {}
