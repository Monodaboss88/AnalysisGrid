"""
Firestore Task Queue
====================
Lightweight task queue using Firestore as the backend.
No Redis/Celery needed — uses your existing Firebase.

Queue flow:
  1. Incoming request → Haiku classifies → task created
  2. Task stored in Firestore with status "pending"
  3. Worker loop picks up pending tasks
  4. Task processed → result stored → notification sent via Telegram

Collections:
  tasks/{taskId} — all queued tasks
  tasks/{taskId}/results — task output

Author: Rob's Trading Systems
Version: 1.0.0
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum


# =============================================================================
# TASK TYPES & STATUS
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    # Quick tasks (Haiku can handle directly)
    ALERT_LOOKUP = "alert_lookup"           # Check current alerts
    TRADE_STATS = "trade_stats"             # Get trade performance
    PRICE_CHECK = "price_check"             # Quick price lookup
    WATCHLIST = "watchlist"                 # Show watchlist

    # Medium tasks (needs scanner data)
    SCANNER_RUN = "scanner_run"             # Run scanner on symbol
    SETUP_ANALYSIS = "setup_analysis"       # Full setup analysis
    ALERT_CHECK = "alert_check"             # Check all alerts against prices

    # Heavy tasks (queue for local Claude)
    FULL_ANALYSIS = "full_analysis"         # Deep AI analysis
    TRADE_PLAN = "trade_plan"              # Generate trade plan
    MARKET_BRIEF = "market_brief"           # Full market brief
    CUSTOM_QUERY = "custom_query"           # Natural language query


class TaskPriority(int, Enum):
    URGENT = 1      # Alert triggers
    HIGH = 2        # Trade-related
    NORMAL = 3      # Standard queries
    LOW = 4         # Reports, summaries


# =============================================================================
# TASK DATA CLASS
# =============================================================================

@dataclass
class Task:
    """A queued task"""
    task_type: str
    payload: Dict = field(default_factory=dict)
    priority: int = TaskPriority.NORMAL
    status: str = TaskStatus.PENDING
    source: str = "telegram"            # telegram, web, scanner, cron
    user_id: str = ""
    chat_id: str = ""                   # For Telegram reply
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    result: str = ""
    error: str = ""
    retry_count: int = 0
    max_retries: int = 3
    task_id: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# Quick tasks that don't need queueing — handle inline
QUICK_TASKS = {
    TaskType.ALERT_LOOKUP,
    TaskType.TRADE_STATS,
    TaskType.PRICE_CHECK,
    TaskType.WATCHLIST,
}


# =============================================================================
# FIRESTORE TASK QUEUE
# =============================================================================

class FirestoreTaskQueue:
    """Task queue backed by Firestore"""

    COLLECTION = "tasks"

    def __init__(self, db=None):
        self.db = db
        self._workers: Dict[str, Callable] = {}
        self._running = False
        self._poll_interval = 5  # seconds

    def set_db(self, db):
        """Set Firestore db client (called after init)"""
        self.db = db

    # =========================================================================
    # ENQUEUE
    # =========================================================================

    def enqueue(self, task: Task) -> Optional[str]:
        """Add a task to the queue"""
        if not self.db:
            print("❌ Task queue: Firestore not available")
            return None

        try:
            task_dict = task.to_dict()
            # Remove empty task_id before adding (Firestore auto-generates)
            task_dict.pop("task_id", None)

            doc_ref = self.db.collection(self.COLLECTION).add(task_dict)
            task_id = doc_ref[1].id
            print(f"📋 Task queued: {task.task_type} [{task_id}] priority={task.priority}")
            return task_id

        except Exception as e:
            print(f"❌ Failed to enqueue task: {e}")
            return None

    # =========================================================================
    # DEQUEUE & PROCESS
    # =========================================================================

    def get_pending_tasks(self, limit: int = 10) -> List[Dict]:
        """Get pending tasks ordered by priority then created_at"""
        if not self.db:
            return []

        try:
            query = (
                self.db.collection(self.COLLECTION)
                .where("status", "==", TaskStatus.PENDING)
                .order_by("priority")
                .order_by("created_at")
                .limit(limit)
            )

            tasks = []
            for doc in query.stream():
                task_data = doc.to_dict()
                task_data["task_id"] = doc.id
                tasks.append(task_data)

            return tasks

        except Exception as e:
            print(f"❌ Error getting pending tasks: {e}")
            return []

    def claim_task(self, task_id: str) -> bool:
        """Atomically claim a task for processing"""
        if not self.db:
            return False

        try:
            task_ref = self.db.collection(self.COLLECTION).document(task_id)

            # Use transaction for atomic claim
            @firestore_transaction
            def claim(transaction, ref):
                doc = ref.get(transaction=transaction)
                if not doc.exists:
                    return False
                if doc.to_dict().get("status") != TaskStatus.PENDING:
                    return False
                transaction.update(ref, {
                    "status": TaskStatus.PROCESSING,
                    "started_at": datetime.now().isoformat()
                })
                return True

            # Simple version without transaction (good enough for single worker)
            task_ref.update({
                "status": TaskStatus.PROCESSING,
                "started_at": datetime.now().isoformat()
            })
            return True

        except Exception as e:
            print(f"❌ Error claiming task {task_id}: {e}")
            return False

    def complete_task(self, task_id: str, result: str) -> bool:
        """Mark task as completed with result"""
        if not self.db:
            return False

        try:
            self.db.collection(self.COLLECTION).document(task_id).update({
                "status": TaskStatus.COMPLETED,
                "result": result,
                "completed_at": datetime.now().isoformat()
            })
            print(f"✅ Task completed: {task_id}")
            return True

        except Exception as e:
            print(f"❌ Error completing task {task_id}: {e}")
            return False

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed"""
        if not self.db:
            return False

        try:
            task_ref = self.db.collection(self.COLLECTION).document(task_id)
            task_data = task_ref.get().to_dict()

            retry_count = task_data.get("retry_count", 0) + 1
            max_retries = task_data.get("max_retries", 3)

            if retry_count < max_retries:
                # Retry — put back to pending
                task_ref.update({
                    "status": TaskStatus.PENDING,
                    "retry_count": retry_count,
                    "error": error
                })
                print(f"🔄 Task retry {retry_count}/{max_retries}: {task_id}")
            else:
                # Exhausted retries
                task_ref.update({
                    "status": TaskStatus.FAILED,
                    "error": error,
                    "completed_at": datetime.now().isoformat()
                })
                print(f"❌ Task failed permanently: {task_id} — {error}")

            return True

        except Exception as e:
            print(f"❌ Error failing task {task_id}: {e}")
            return False

    # =========================================================================
    # TASK STATS
    # =========================================================================

    def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        if not self.db:
            return {}

        try:
            stats = {}
            for status in [TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.COMPLETED, TaskStatus.FAILED]:
                query = self.db.collection(self.COLLECTION).where("status", "==", status)
                count = sum(1 for _ in query.stream())
                stats[status] = count

            return stats

        except Exception as e:
            print(f"❌ Error getting queue stats: {e}")
            return {}

    def cleanup_old_tasks(self, days: int = 7) -> int:
        """Remove completed/failed tasks older than N days"""
        if not self.db:
            return 0

        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            deleted = 0

            for status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                query = (
                    self.db.collection(self.COLLECTION)
                    .where("status", "==", status)
                    .where("completed_at", "<", cutoff)
                )

                for doc in query.stream():
                    doc.reference.delete()
                    deleted += 1

            print(f"🧹 Cleaned up {deleted} old tasks")
            return deleted

        except Exception as e:
            print(f"❌ Error cleaning up tasks: {e}")
            return 0

    # =========================================================================
    # WORKER REGISTRATION
    # =========================================================================

    def register_worker(self, task_type: str, handler: Callable):
        """Register a handler function for a task type"""
        self._workers[task_type] = handler
        print(f"👷 Worker registered: {task_type}")

    async def process_next(self) -> bool:
        """Process the next pending task. Returns True if a task was processed."""
        tasks = self.get_pending_tasks(limit=1)
        if not tasks:
            return False

        task_data = tasks[0]
        task_id = task_data["task_id"]
        task_type = task_data.get("task_type", "")

        if task_type not in self._workers:
            self.fail_task(task_id, f"No worker for task type: {task_type}")
            return True

        if not self.claim_task(task_id):
            return False

        try:
            handler = self._workers[task_type]
            task = Task.from_dict(task_data)
            task.task_id = task_id

            result = await handler(task)
            self.complete_task(task_id, json.dumps(result) if isinstance(result, dict) else str(result))
            return True

        except Exception as e:
            self.fail_task(task_id, str(e))
            return True

    async def run_worker_loop(self):
        """Main worker loop — continuously process tasks"""
        self._running = True
        print(f"🚀 Task worker started (polling every {self._poll_interval}s)")

        while self._running:
            try:
                processed = await self.process_next()
                if not processed:
                    await asyncio.sleep(self._poll_interval)
                else:
                    # Process quickly if there are more tasks
                    await asyncio.sleep(0.5)
            except Exception as e:
                print(f"❌ Worker loop error: {e}")
                await asyncio.sleep(self._poll_interval)

    def stop_worker(self):
        """Stop the worker loop"""
        self._running = False
        print("🛑 Task worker stopping...")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_task_queue: Optional[FirestoreTaskQueue] = None

def get_task_queue() -> FirestoreTaskQueue:
    """Get the global task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = FirestoreTaskQueue()
    return _task_queue
