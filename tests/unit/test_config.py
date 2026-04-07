

from valravn.config import OutputConfig, RetryConfig, load_config


def test_retry_config_defaults():
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.retry_delay_seconds == 0.0


def test_load_config_from_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("retry:\n  max_attempts: 5\n")
    cfg = load_config(cfg_file)
    assert cfg.retry.max_attempts == 5


def test_load_config_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VALRAVN_MAX_RETRIES", "7")
    cfg = load_config(None)
    assert cfg.retry.max_attempts == 7


def test_retry_config_timeout_default():
    cfg = RetryConfig()
    assert cfg.timeout_seconds == 3600


def test_retry_config_timeout_override():
    cfg = RetryConfig(timeout_seconds=600)
    assert cfg.timeout_seconds == 600


def test_output_config_dirs(tmp_path):
    cfg = OutputConfig(output_dir=tmp_path)
    assert cfg.analysis_dir == tmp_path / "analysis"
    assert cfg.reports_dir == tmp_path / "reports"
    assert cfg.checkpoints_db == tmp_path / "analysis" / "checkpoints.db"
    assert cfg.traces_dir == tmp_path / "analysis" / "traces"
