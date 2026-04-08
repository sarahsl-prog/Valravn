from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env from repo root (if present) before any os.environ reads.
# Values already set in the shell environment take precedence (override=False).
load_dotenv(Path(__file__).parents[2] / ".env", override=False)


class RetryConfig(BaseModel):
    max_attempts: int = 3
    retry_delay_seconds: float = 0.0
    timeout_seconds: int = 3600


class OutputConfig(BaseModel):
    output_dir: Path

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / "analysis"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def exports_dir(self) -> Path:
        return self.output_dir / "exports"

    @property
    def traces_dir(self) -> Path:
        return self.output_dir / "analysis" / "traces"

    @property
    def checkpoints_db(self) -> Path:
        return self.output_dir / "analysis" / "checkpoints.db"

    def ensure_dirs(self) -> None:
        for d in (self.analysis_dir, self.reports_dir, self.exports_dir, self.traces_dir):
            d.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    retry: RetryConfig = RetryConfig()
    mlflow: dict = {
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": "valravn-evaluation",
    }


def load_config(path: Path | None) -> AppConfig:
    data: dict = {}
    if path and path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    cfg = AppConfig(**data)

    env_max = os.environ.get("VALRAVN_MAX_RETRIES")
    if env_max:
        cfg.retry.max_attempts = int(env_max)

    return cfg
