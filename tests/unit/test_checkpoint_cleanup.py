"""Tests for SQLite checkpoint cleanup policy (Q6)."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from valravn.checkpoint_cleanup import (
    CheckpointCleanupPolicy,
    cleanup_checkpoints,
    vacuum_db,
)


def _create_test_db(path: Path, checkpoints: list[tuple]) -> None:
    """Create a test database with sample checkpoints.
    
    Args:
        path: Database file path
        checkpoints: List of (thread_id, checkpoint_id, created_at) tuples
    """
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                created_at TEXT,
                data TEXT
            )
        """)
        
        for thread_id, cp_id, created_at in checkpoints:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints (id, thread_id, created_at, data) VALUES (?, ?, ?, ?)",
                (cp_id, thread_id, created_at, "test_data")
            )
        conn.commit()


class TestCheckpointCleanup:
    """Test checkpoint cleanup functionality."""

    def test_cleanup_by_age(self):
        """Delete checkpoints older than retention period."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=10)).isoformat()
        recent = (now - timedelta(days=1)).isoformat()
        
        checkpoints = [
            ("thread_1", "old_1", old),
            ("thread_1", "old_2", old),
            ("thread_1", "recent_1", recent),
        ]
        
        _create_test_db(db_path, checkpoints)
        
        # Cleanup with 7-day retention
        policy = CheckpointCleanupPolicy(retention_days=7)
        stats = policy.cleanup(db_path)
        
        assert stats["deleted_by_age"] == 2
        assert stats["total_deleted"] == 2
        
        # Verify only recent remains
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT id FROM checkpoints")
            remaining = {row[0] for row in cursor.fetchall()}
        
        assert remaining == {"recent_1"}
        
        db_path.unlink(missing_ok=True)

    def test_cleanup_by_count(self):
        """Delete excess checkpoints per thread."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        
        now = datetime.now(timezone.utc)
        
        # 5 checkpoints for thread_1
        checkpoints = [
            ("thread_1", f"cp_{i}", (now - timedelta(hours=i)).isoformat())
            for i in range(5)
        ]
        
        _create_test_db(db_path, checkpoints)
        
        # Keep only 3 per thread
        policy = CheckpointCleanupPolicy(
            retention_days=30,  # Don't delete by age
            max_checkpoints_per_thread=3
        )
        stats = policy.cleanup(db_path)
        
        # Should delete 2 (5 - 3)
        assert stats["deleted_by_count"] == 2
        
        # Verify 3 most recent remain (cp_0, cp_1, cp_2)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT id FROM checkpoints ORDER BY created_at DESC")
            remaining = [row[0] for row in cursor.fetchall()]
        
        assert remaining == ["cp_0", "cp_1", "cp_2"]
        
        db_path.unlink(missing_ok=True)

    def test_cleanup_combined_policies(self):
        """Both age and count restrictions apply."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = None
            now = datetime.now(timezone.utc)
            old = (now - timedelta(days=10)).isoformat()
            
            # Mix of old and new
            checkpoints = [
                ("thread_1", "very_old", old),
                ("thread_1", "cp_1", now.isoformat()),
                ("thread_1", "cp_2", now.isoformat()),
            ]
            
            db_path = Path(tmp.name)
            _create_test_db(db_path, checkpoints)
            
            policy = CheckpointCleanupPolicy(
                retention_days=7,
                max_checkpoints_per_thread=1  # Keep only 1
            )
            stats = policy.cleanup(db_path)
            
            # Old one deleted by age, then count limit enforced
            assert stats["deleted_by_age"] >= 1
            assert stats["deleted_by_count"] >= 1
            
            db_path.unlink(missing_ok=True)

    def test_empty_db_cleanup(self):
        """Cleanup on non-existent database."""
        db_path = Path("/tmp/nonexistent_test_db.db")
        
        policy = CheckpointCleanupPolicy()
        stats = policy.cleanup(db_path)
        
        assert stats["total_deleted"] == 0

    def test_cleanup_stats(self):
        """Get statistics about checkpoint database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        
        now = datetime.now(timezone.utc)
        
        checkpoints = [
            ("thread_1", "cp_1", (now - timedelta(days=1)).isoformat()),
            ("thread_2", "cp_2", now.isoformat()),
        ]
        
        _create_test_db(db_path, checkpoints)
        
        policy = CheckpointCleanupPolicy()
        stats = policy.get_stats(db_path)
        
        assert stats["total_checkpoints"] == 2
        assert stats["unique_threads"] == 2
        assert stats["db_size_bytes"] > 0
        
        db_path.unlink(missing_ok=True)

    def test_simple_cleanup_function(self):
        """Test convenience cleanup function."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        
        now = datetime.now(timezone.utc)
        
        checkpoints = [
            ("thread_1", "old", (now - timedelta(days=10)).isoformat()),
            ("thread_1", "new", now.isoformat()),
        ]
        
        _create_test_db(db_path, checkpoints)
        
        stats = cleanup_checkpoints(db_path, retention_days=7, max_checkpoints=100)
        
        assert stats["deleted_by_age"] == 1
        
        db_path.unlink(missing_ok=True)

    def test_vacuum_db(self):
        """Test database vacuum operation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = Path(tmp.name)
        
        now = datetime.now(timezone.utc)
        checkpoints = [("thread_1", "cp_1", now.isoformat())]
        _create_test_db(db_path, checkpoints)
        
        # Vacuum should not raise
        vacuum_db(db_path)
        
        # Verify database still valid
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM checkpoints")
            assert cursor.fetchone()[0] == 1
        
        db_path.unlink(missing_ok=True)