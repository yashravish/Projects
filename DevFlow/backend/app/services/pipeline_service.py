from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models.pipeline import (
    PipelineRun,
    PipelineStage,
    PipelineStatus,
    StageName,
    StageStatus,
    TestResult,
    TestStatus,
)
from app.models.project import Project
from app.services.metrics_state import global_metrics

STAGE_ORDER: list[StageName] = list(StageName)


@dataclass
class _SimState:
    fail_at_stage: Optional[StageName] = None


def _should_force_failure(project_id: int, run_id: int) -> _SimState:
    """
    Deterministic 'sample' failures: ~12% of runs fail at a random stage.
    Special: project_id ending in 7 always fails at unit_tests for demo.
    """
    seed = f"{project_id}:{run_id}".encode()
    h = int(hashlib.sha256(seed).hexdigest()[:8], 16)
    rng = random.Random(h)
    if project_id % 10 == 7:
        return _SimState(fail_at_stage=StageName.unit_tests)
    if rng.random() < 0.12:
        return _SimState(fail_at_stage=STAGE_ORDER[rng.randint(0, len(STAGE_ORDER) - 1)])
    return _SimState()


async def _sleep_ms(ms: int) -> None:
    await asyncio.sleep(max(0, ms) / 1000.0)


async def run_pipeline_simulation(
    session: AsyncSession, run: PipelineRun, project: Project
) -> PipelineRun:
    s = get_settings()
    st = _should_force_failure(project.id, run.id)
    t0 = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    run.started_at = t0
    run.status = PipelineStatus.running
    run.stages = [
        PipelineStage(
            run=run,
            name=name,
            sort_order=i,
            status=StageStatus.pending,
            duration_ms=0,
            logs="",
            passed=True,
        )
        for i, name in enumerate(STAGE_ORDER)
    ]
    await session.flush()
    for stage in sorted(run.stages, key=lambda x: x.sort_order):
        await session.refresh(stage)
        await _sleep_ms(s.pipeline_simulation_delay_ms + (run.id % 5) * 2)
        stage.status = StageStatus.running
        await session.flush()
        dms = 15 + (run.id * 3 + stage.id) % 200
        fail_here = st.fail_at_stage == stage.name
        log_lines = [
            f"[{stage.name.value}] start project={project.slug} run={run.id}",
            f"[{stage.name.value}] duration_target_ms~={dms}",
        ]
        if fail_here:
            stage.status = StageStatus.failed
            stage.duration_ms = dms
            stage.passed = False
            if stage.name == StageName.unit_tests:
                log_lines.append("FAILED tests/test_auth.py::test_token_refresh - AssertionError: expected 200, got 401")
            elif stage.name == StageName.lint:
                log_lines.append("ruff: F401 re-imported 'os' in app/main.py")
            else:
                log_lines.append("ERROR: simulated stage failure (deterministic sample)")
            stage.logs = "\n".join(log_lines) + "\n"
            run.status = PipelineStatus.failed
        else:
            stage.status = StageStatus.success
            stage.duration_ms = dms
            stage.passed = True
            log_lines.append(f"[{stage.name.value}] success")
            stage.logs = "\n".join(log_lines) + "\n"
        await session.flush()
        if fail_here:
            break

    if run.status == PipelineStatus.running or run.status == PipelineStatus.success:
        run.status = PipelineStatus.success
    t1 = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    run.finished_at = t1
    if run.started_at:
        delta = t1 - run.started_at
        run.total_duration_ms = int(delta.total_seconds() * 1000)
    if run.status == PipelineStatus.success:
        n = 2 + (run.id % 4)
        for i in range(n):
            tr = TestResult(
                run=run,
                name=f"test_case_{i}",
                suite="unit",
                status=TestStatus.passed,
                duration_ms=1 + (i * 2) % 40,
                message="ok",
            )
            session.add(tr)
    elif run.status == PipelineStatus.failed:
        session.add(
            TestResult(
                run=run,
                name="failing_regression",
                suite="unit",
                status=TestStatus.failed,
                duration_ms=12,
                message="See stage logs for assertion details",
            )
        )
    success = run.status == PipelineStatus.success
    global_metrics.record_pipeline(
        success, run.total_duration_ms / 1000.0 if run.total_duration_ms else 0.0
    )
    await session.flush()
    return run


async def create_pending_run(
    session: AsyncSession, project: Project, branch: str, commit_sha: str
) -> PipelineRun:
    r = PipelineRun(
        project_id=project.id,
        status=PipelineStatus.pending,
        branch=branch,
        commit_sha=commit_sha,
        total_duration_ms=0,
        external_ref=None,
    )
    session.add(r)
    await session.flush()
    return r


async def get_run_with_relations(session: AsyncSession, run_id: int) -> PipelineRun | None:
    q = await session.execute(
        select(PipelineRun)
        .options(
            selectinload(PipelineRun.stages),
            selectinload(PipelineRun.test_results),
            selectinload(PipelineRun.project),
        )
        .where(PipelineRun.id == run_id)
    )
    return q.scalar_one_or_none()
