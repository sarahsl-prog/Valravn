import os
from pathlib import Path
import pytest


@pytest.fixture
def read_only_evidence(tmp_path) -> Path:
    p = tmp_path / "evidence.raw"
    p.write_bytes(b"x")
    os.chmod(p, 0o444)
    return p


@pytest.fixture
def output_dir(tmp_path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d
