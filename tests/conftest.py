import os
from pathlib import Path

import pytest


@pytest.fixture
def read_only_evidence(tmp_path) -> Path:
    # Place evidence in its own subdirectory so the protected root (parent)
    # is evidence_dir, not tmp_path itself — keeps output_dir outside FR-007 scope.
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    p = evidence_dir / "evidence.raw"
    p.write_bytes(b"x")
    os.chmod(p, 0o444)
    return p


@pytest.fixture
def output_dir(tmp_path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d
