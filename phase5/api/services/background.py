from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, func: Callable, args: tuple, kwargs: dict) -> None:
        self.job_id = job_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = JobStatus.PENDING
        self.result: Any = None
        self.error: str | None = None
        self.created_at = time.time()
        self.completed_at: float | None = None


class BackgroundJobManager:
    def __init__(self, max_workers: int = 4) -> None:
        self._jobs: dict[str, Job] = {}
        self._max_workers = max_workers
        self._lock = threading.Lock()

    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> str:
        job_id = str(uuid.uuid4())
        job = Job(job_id, func, args, kwargs)
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job_id

    def _run_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        try:
            job.result = job.func(*job.args, **job.kwargs)
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
        finally:
            job.completed_at = time.time()

    def get_status(self, job_id: str) -> str | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.status

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.job_id,
                "status": job.status,
                "result": job.result,
                "error": job.error,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
            }

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            all_jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "created_at": j.created_at,
                    "completed_at": j.completed_at,
                    "error": j.error,
                }
                for j in all_jobs[:limit]
            ]


_job_manager: BackgroundJobManager | None = None


def get_job_manager() -> BackgroundJobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = BackgroundJobManager()
    return _job_manager
