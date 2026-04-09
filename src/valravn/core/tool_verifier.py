"""Tool availability verification for SANS SIFT workstation.

Provides pre-flight checks to ensure all required forensic tools
are available before investigation begins.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger


@dataclass
class ToolCheckResult:
    """Result of checking a single tool."""

    name: str
    available: bool
    path: str | None
    error: str | None = None


class ToolVerifier:
    """Verifies availability of required forensic tools on SIFT workstation.

    Usage:
        verifier = ToolVerifier()
        results = verifier.verify_all()

        if not verifier.all_critical_available():
            missing = verifier.get_missing_critical()
            raise RuntimeError(f"Missing tools: {missing}")
    """

    # Critical tools that must be available
    CRITICAL_TOOLS: dict[str, list[str]] = {
        # Sleuth Kit
        "fls": ["fls"],
        "icat": ["icat"],
        "ils": ["ils"],
        "blkls": ["blkls"],
        "mactime": ["mactime"],
        "tsk_recover": ["tsk_recover"],
        # Plaso
        "log2timeline.py": ["log2timeline.py"],
        "psort.py": ["psort.py"],
        "pinfo.py": ["pinfo.py"],
        # Volatility 3
        "vol.py": ["python3", "/opt/volatility3-2.20.0/vol.py"],
        # YARA
        "yara": ["/usr/local/bin/yara", "yara"],
        # Basic utilities
        "strings": ["strings"],
    }

    # Optional tools - warnings but not blocking
    OPTIONAL_TOOLS: dict[str, list[str]] = {
        # EWF tools
        "ewfmount": ["ewfmount"],
        "ewfinfo": ["ewfinfo"],
        "ewfverify": ["ewfverify"],
        # Memory baseliner
        "memory-baseliner": ["python3", "/opt/memory-baseliner/baseline.py"],
        # Bulk extractor
        "bulk_extractor": ["bulk_extractor"],
        # Photorec
        "photorec": ["photorec"],
        # EZ Tools (Windows-only, may not be available on Linux)
        "EvtxECmd": None,  # Windows .NET tool
        "RegistryExplorer": None,
        "TimelineExplorer": None,
    }

    def __init__(self, custom_checks: dict[str, list[str]] | None = None) -> None:
        """Initialize verifier.

        Args:
            custom_checks: Optional dict of tool_name -> command to override defaults
        """
        self.results: list[ToolCheckResult] = []

        # Apply custom checks if provided
        if custom_checks:
            for name, cmd in custom_checks.items():
                if name in self.CRITICAL_TOOLS:
                    self.CRITICAL_TOOLS[name] = cmd
                else:
                    self.CRITICAL_TOOLS[name] = cmd

    def check_tool(self, name: str, cmd: list[str]) -> ToolCheckResult:
        """Check if a specific tool is available.

        Args:
            name: Human-readable tool name
            cmd: Command list (first element should be the executable)

        Returns:
            ToolCheckResult with availability status
        """
        if not cmd:
            return ToolCheckResult(name=name, available=False, path=None, error="Empty command")

        executable = cmd[0]

        # Handle absolute paths
        if executable.startswith("/"):
            exe_path = Path(executable)
            if exe_path.exists() and exe_path.is_file():
                if shutil.which("python3"):
                    return ToolCheckResult(name=name, available=True, path=str(exe_path))
            return ToolCheckResult(
                name=name, available=False, path=None, error=f"Path not found: {executable}"
            )

        # Handle python3 <script> style commands
        if executable == "python3" and len(cmd) >= 2:
            script_path = Path(cmd[1])
            if script_path.exists() and script_path.is_file():
                # Check python3 is available
                python_path = shutil.which("python3")
                if python_path:
                    return ToolCheckResult(
                        name=name, available=True, path=f"{python_path} {script_path}"
                    )
            return ToolCheckResult(
                name=name, available=False, path=None, error=f"Script not found: {script_path}"
            )

        # Standard PATH lookup
        found_path = shutil.which(executable)
        if found_path:
            return ToolCheckResult(name=name, available=True, path=found_path)

        return ToolCheckResult(
            name=name, available=False, path=None, error=f"Command not found in PATH: {executable}"
        )

    def verify_all(self) -> list[ToolCheckResult]:
        """Verify all critical and optional tools.

        Returns:
            List of all check results
        """
        self.results = []

        logger.info("Verifying critical forensic tools...")
        for name, cmd in self.CRITICAL_TOOLS.items():
            if cmd is None:
                continue
            result = self.check_tool(name, cmd)
            self.results.append(result)

            if result.available:
                logger.debug(f"✓ {name}: {result.path}")
            else:
                logger.warning(f"✗ {name}: {result.error}")

        logger.info("Verifying optional forensic tools...")
        for name, cmd in self.OPTIONAL_TOOLS.items():
            if cmd is None:
                continue
            result = self.check_tool(name, cmd)
            self.results.append(result)

            if result.available:
                logger.debug(f"✓ {name}: {result.path}")
            else:
                logger.info(f"○ {name} (optional): {result.error}")

        return self.results

    def all_critical_available(self) -> bool:
        """Check if all critical tools are available.

        Returns:
            True if all critical tools found
        """
        if not self.results:
            self.verify_all()

        for result in self.results:
            if result.name in self.CRITICAL_TOOLS and not result.available:
                return False
        return True

    def get_missing_critical(self) -> list[str]:
        """Get list of missing critical tool names.

        Returns:
            List of missing critical tool names
        """
        if not self.results:
            self.verify_all()

        return [r.name for r in self.results if r.name in self.CRITICAL_TOOLS and not r.available]

    def get_missing_optional(self) -> list[str]:
        """Get list of missing optional tool names.

        Returns:
            List of missing optional tool names
        """
        if not self.results:
            self.verify_all()

        return [r.name for r in self.results if r.name in self.OPTIONAL_TOOLS and not r.available]

    def get_available_tools(self) -> dict[str, str]:
        """Get dict of available tool names and their paths.

        Returns:
            Dict mapping tool name to path
        """
        if not self.results:
            self.verify_all()

        return {r.name: r.path for r in self.results if r.available}

    def generate_report(self) -> str:
        """Generate a formatted report of tool availability.

        Returns:
            Multi-line string report
        """
        if not self.results:
            self.verify_all()

        lines = ["=" * 60]
        lines.append("FORENSIC TOOL AVAILABILITY REPORT")
        lines.append("=" * 60)

        lines.append("\nCritical Tools:")
        lines.append("-" * 40)
        for name in self.CRITICAL_TOOLS:
            result = next((r for r in self.results if r.name == name), None)
            if result:
                status = "✓ AVAILABLE" if result.available else "✗ MISSING"
                lines.append(f"  {status:15} {name}")
                if result.available and result.path:
                    lines.append(f"    Path: {result.path}")
                elif result.error:
                    lines.append(f"    Error: {result.error}")

        lines.append("\nOptional Tools:")
        lines.append("-" * 40)
        for name in self.OPTIONAL_TOOLS:
            if name is None:
                continue
            result = next((r for r in self.results if r.name == name), None)
            if result:
                status = "✓ AVAILABLE" if result.available else "○ NOT FOUND"
                lines.append(f"  {status:15} {name}")

        lines.append("\n" + "=" * 60)

        missing = self.get_missing_critical()
        if missing:
            lines.append(f"WARNING: {len(missing)} critical tool(s) missing!")
            lines.append(f"Missing: {', '.join(missing)}")
        else:
            lines.append("All critical tools available.")

        lines.append("=" * 60)

        return "\n".join(lines)


def verify_tools_or_raise(raise_on_missing: bool = True) -> ToolVerifier:
    """Convenience function to verify tools and optionally raise error.

    Args:
        raise_on_missing: If True, raises RuntimeError if critical tools missing

    Returns:
        ToolVerifier instance with results

    Raises:
        RuntimeError: If raise_on_missing=True and critical tools missing
    """
    verifier = ToolVerifier()
    verifier.verify_all()

    if not verifier.all_critical_available():
        missing = verifier.get_missing_critical()
        msg = f"Missing {len(missing)} critical forensic tool(s): {', '.join(missing)}"
        logger.error(msg)
        logger.error("\n" + verifier.generate_report())
        if raise_on_missing:
            raise RuntimeError(msg)
    else:
        logger.info("All critical forensic tools verified.")

    return verifier
