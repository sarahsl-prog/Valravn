from __future__ import annotations

from valravn.training.progress_memory import InvestigationBlueprint, ProgressAnchor, ProgressMemory


def test_progress_memory_has_default_blueprints():
    pm = ProgressMemory()
    assert len(pm.blueprints) > 0


def test_progress_memory_retrieves_matching_blueprint():
    pm = ProgressMemory()
    result = pm.retrieve_blueprint("memory dump analysis with injection detection")
    assert result.incident_type == "memory_analysis"


def test_progress_memory_blueprint_has_anchors():
    pm = ProgressMemory()
    bp = pm.blueprints[0]
    assert len(bp.anchors) > 0
    stage_1 = next((a for a in bp.anchors if a.stage == 1), None)
    assert stage_1 is not None


def test_progress_memory_add_blueprint():
    pm = ProgressMemory()
    initial_count = len(pm.blueprints)
    new_bp = InvestigationBlueprint(
        incident_type="network_forensics",
        anchors=[
            ProgressAnchor(
                stage=1,
                description="Capture and filter network traffic",
                typical_tools=["tcpdump", "wireshark"],
                completion_signal="PCAP file acquired",
            )
        ],
        success_rate=0.75,
    )
    pm.add_blueprint(new_bp)
    assert len(pm.blueprints) == initial_count + 1


def test_progress_memory_fallback_to_first_blueprint():
    pm = ProgressMemory()
    result = pm.retrieve_blueprint("xyzzy unrelated gibberish query foobar")
    assert result.incident_type == pm.blueprints[0].incident_type


def test_progress_memory_save_and_load(tmp_path):
    pm = ProgressMemory()
    pm.add_blueprint(
        InvestigationBlueprint(
            incident_type="custom_test",
            anchors=[
                ProgressAnchor(
                    stage=1,
                    description="Run custom scan",
                    typical_tools=["custom_tool"],
                    completion_signal="Scan complete",
                )
            ],
            success_rate=0.5,
        )
    )
    save_path = tmp_path / "progress_memory.json"
    pm.save(save_path)

    loaded = ProgressMemory.load(save_path)
    assert len(loaded.blueprints) == len(pm.blueprints)
    incident_types = [bp.incident_type for bp in loaded.blueprints]
    assert "custom_test" in incident_types
    # Verify anchor data survives roundtrip
    custom_bp = next(bp for bp in loaded.blueprints if bp.incident_type == "custom_test")
    assert custom_bp.anchors[0].stage == 1
    assert custom_bp.anchors[0].typical_tools == ["custom_tool"]
    assert custom_bp.success_rate == 0.5
