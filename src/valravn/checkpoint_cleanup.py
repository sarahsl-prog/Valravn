"""SQLite checkpoint cleanup policy for LangGraph.

Prevents unbounded growth of checkpoint files by implementing
retention policies and manual cleanup utilities.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from loguru import logger
from langgraph.checkpoint.sqlite import SqliteSaver

# Default retention policy - keep last 7 days
DEFAULT_RETENTION_DAYS = 7
DEFAULT_MAX_CHECKPOINTS = 1000


class CheckpointCleanupPolicy:
    """Configuration for checkpoint cleanup behavior.
    
    Supports two retention strategies:
    1. Time-based: Delete checkpoints older than retention_days
    2. Count-based: Keep only the N most recent checkpoints per thread
    
    The more restrictive of the two policies is applied.
    """
    
    def __init__(
        self,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_checkpoints_per_thread: int = DEFAULT_MAX_CHECKPOINTS,
        min_checkpoints_per_thread: int = 2,
    ):
        """Initialize cleanup policy.
        
        Args:
            retention_days: Delete checkpoints older than this many days
            max_checkpoints_per_thread: Keep at most this many checkpoints per thread
            min_checkpoints_per_thread: Never delete below this many checkpoints
        """
        self.retention_days = retention_days
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self.min_checkpoints_per_thread = min_checkpoints_per_thread
    
    def cleanup(self, db_path: Path) -> dict[str, int]:
        """Apply cleanup policy to a checkpoint database.
        
        Args:
            db_path: Path to the SQLite checkpoint database
            
        Returns:
            Dict with cleanup statistics:
                - deleted_by_age: Number of old checkpoints removed
                - deleted_by_count: Number of excess checkpoints removed
                - total_deleted: Total checkpoints removed
        """
        if not db_path.exists():
            return {"deleted_by_age": 0, "deleted_by_count": 0, "total_deleted": 0}
        
        stats = {"deleted_by_age": 0, "deleted_by_count": 0}
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Clean up by age
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            cutoff_iso = cutoff_date.isoformat()
            
            cursor.execute(
                """
                DELETE FROM checkpoints
                WHERE created_at < ?
                """,
                (cutoff_iso,)
            )
            stats["deleted_by_age"] = cursor.rowcount
            
            # Clean up by count per thread (keep only recent MAX)
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            threads = cursor.fetchall()
            
            for (thread_id,) in threads:
                cursor.execute(
                    """
                    DELETE FROM checkpoints
                    WHERE thread_id = ? AND id NOT IN (
                        SELECT id FROM checkpoints
                        WHERE thread_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    )
                    """,
                    (thread_id, thread_id, self.max_checkpoints_per_thread)
                )
                stats["deleted_by_count"] += cursor.rowcount
            
            # Apply minimum retention (ensure at least min_checkpoints remain)
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            threads = cursor.fetchall()
            
            for (thread_id,) in threads:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM checkpoints
                    WHERE thread_id = ?
                    """,
                    (thread_id,)
                )
                count = cursor.fetchone()[0]
                
                if count < self.min_checkpoints_per_thread:
                    # Restore from deleted or log warning
                    logger.warning(
                        "Thread %r has only %d checkpoints (min: %d)",
                        thread_id, count, self.min_checkpoints_per_thread
                    )
            
            conn.commit()
        
        stats["total_deleted"] = stats["deleted_by_age"] + stats["deleted_by_count"]
        logger.info(
            "Checkpoint cleanup complete: %d deleted (%d by age, %d by count)",
            stats["total_deleted"],
            stats["deleted_by_age"],
            stats["deleted_by_count"]
        )
        
        return stats
    
    def get_stats(self, db_path: Path) -> dict:
        """Get checkpoint database statistics.
        
        Returns:
            Dict with:
                - total_checkpoints: Total checkpoint records
                - unique_threads: Number of unique thread IDs
                - oldest_checkpoint: ISO timestamp of oldest checkpoint
                - newest_checkpoint: ISO timestamp of newest checkpoint
                - db_size_bytes: Database file size
        """
        if not db_path.exists():
            return {
                "total_checkpoints": 0,
                "unique_threads": 0,
                "oldest_checkpoint": None,
                "newest_checkpoint": None,
                "db_size_bytes": 0,
            }
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get checkpoint counts
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
            threads = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM checkpoints"
            )
            oldest, newest = cursor.fetchone()
        
        return {
            "total_checkpoints": total,
            "unique_threads": threads,
            "oldest_checkpoint": oldest,
            "newest_checkpoint": newest,
            "db_size_bytes": db_path.stat().st_size,
        }


def cleanup_checkpoints(
    db_path: Path,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> dict[str, int]:
    """Simple cleanup: remove old checkpoints following retention policy.
    
    Args:
        db_path: Path to checkpoint database
        retention_days: Delete checkpoints older than this
        max_checkpoints: Keep at most this many per thread
        
    Returns:
        Cleanup statistics dict
    """
    policy = CheckpointCleanupPolicy(
        retention_days=retention_days,
        max_checkpoints_per_thread=max_checkpoints
    )
    return policy.cleanup(db_path)


def vacuum_db(db_path: Path) -> None:
    """Run VACUUM on the checkpoint database to reclaim space.
    
    Warning: This rewrite the entire database and requires temporary disk space.
    """
    if not db_path.exists():
        return
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("VACUUM")
    
    logger.info("Vacuumed checkpoint database: {}", db_path)
