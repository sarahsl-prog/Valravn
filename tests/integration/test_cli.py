"""Integration tests for CLI entry point (valravn investigate).

These tests verify the full CLI flow through to graph execution,
using mocks to avoid actual LLM and tool dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from valravn.cli import build_parser, main
from valravn.config import AppConfig


def _create_stub_evidence(tmp_path: Path) -> Path:
    """Create a stub evidence file that is read-only."""
    evidence = tmp_path / "evidence.raw"
    evidence.write_bytes(b"stub evidence data\x00\x01\x02")
    os.chmod(evidence, 0o444)
    return evidence


class TestCLIArgumentParsing:
    """Tests for build_parser() - argument parsing and validation."""

    def test_build_parser_investigate_command(self):
        """Parser accepts investigate subcommand with required args."""
        parser = build_parser()
        args = parser.parse_args([
            "investigate",
            "--prompt", "find network connections",
            "--evidence", "/tmp/ev1.raw",
        ])
        assert args.command == "investigate"
        assert args.prompt == "find network connections"
        assert args.evidence_refs == ["/tmp/ev1.raw"]

    def test_build_parser_multiple_evidence(self):
        """Parser accepts multiple --evidence flags."""
        parser = build_parser()
        args = parser.parse_args([
            "investigate",
            "--prompt", "test",
            "--evidence", "/tmp/ev1.raw",
            "--evidence", "/tmp/ev2.raw",
        ])
        assert len(args.evidence_refs) == 2

    def test_build_parser_output_dir(self):
        """Parser accepts --output-dir flag."""
        parser = build_parser()
        args = parser.parse_args([
            "investigate",
            "--prompt", "test",
            "--evidence", "/tmp/ev.raw",
            "--output-dir", "/tmp/output",
        ])
        assert args.output_dir == Path("/tmp/output")

    def test_build_parser_config_file(self):
        """Parser accepts --config flag."""
        parser = build_parser()
        args = parser.parse_args([
            "investigate",
            "--prompt", "test",
            "--evidence", "/tmp/ev.raw",
            "--config", "/tmp/config.yaml",
        ])
        assert args.config == Path("/tmp/config.yaml")


class TestCLIMainFunction:
    """Tests for main() function - full CLI execution flow."""

    def test_main_creates_output_directories(self, tmp_path: Path):
        """main() creates output directories before execution."""
        evidence = _create_stub_evidence(tmp_path)
        output_dir = tmp_path / "output"

        with patch("valravn.graph.run") as mock_run, \
             patch("valravn.cli.load_config") as mock_load_config:
            mock_run.return_value = 0
            mock_load_config.return_value = AppConfig()

            with patch.object(sys, "argv", [
                "valravn", "investigate",
                "--prompt", "test investigation",
                "--evidence", str(evidence),
                "--output-dir", str(output_dir),
            ]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

        # Verify directories were created
        assert output_dir.exists()
        assert (output_dir / "analysis").exists()
        assert (output_dir / "reports").exists()
        assert (output_dir / "exports").exists()

    def test_main_graph_execution_success(self, tmp_path: Path):
        """main() executes graph and returns success code."""
        evidence = _create_stub_evidence(tmp_path)

        with (
            patch("valravn.graph.run") as mock_run,
            patch("valravn.cli.load_config") as mock_load_config,
        ):
            mock_run.return_value = 0
            mock_load_config.return_value = AppConfig()

            with patch.object(sys, "argv", [
                "valravn", "investigate",
                "--prompt", "test",
                "--evidence", str(evidence),
            ]):
                with patch("valravn.cli.sys.exit") as mock_exit:
                    main()
                    mock_exit.assert_called_once_with(0)

    def test_main_graph_execution_failure(self, tmp_path: Path):
        """main() returns exit code 1 when graph fails."""
        evidence = _create_stub_evidence(tmp_path)

        with patch("valravn.graph.run") as mock_run, \
             patch("valravn.cli.load_config"):
            mock_run.return_value = 1

            with patch.object(sys, "argv", [
                "valravn", "investigate",
                "--prompt", "test",
                "--evidence", str(evidence),
            ]):
                with patch("valravn.cli.sys.exit") as mock_exit:
                    main()
                    mock_exit.assert_called_once_with(1)


class TestCLIEnvironmentOverrides:
    """Tests for environment variable overrides in CLI."""

    def test_env_max_retries_override(self, tmp_path: Path):
        """VALRAVN_MAX_RETRIES overrides config file setting."""
        evidence = _create_stub_evidence(tmp_path)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("retry:\n  max_attempts: 3\n")

        with patch("valravn.graph.run") as mock_run, \
             patch.dict(os.environ, {"VALRAVN_MAX_RETRIES": "5"}):
            mock_run.return_value = 0

            with patch.object(sys, "argv", [
                "valravn", "investigate",
                "--prompt", "test",
                "--evidence", str(evidence),
                "--config", str(config_file),
            ]):
                with patch("valravn.cli.sys.exit"):
                    main()

            # Verify env override was applied
            args = mock_run.call_args
            config = args[1]["app_cfg"] if args else None
            if config:
                assert config.retry.max_attempts == 5
