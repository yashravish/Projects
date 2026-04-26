from __future__ import annotations

import datetime as dt
import hashlib
import random
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.deployment import Deployment, DeploymentStatus
from app.services.metrics_state import global_metrics

CANARY_STEPS = (10, 25, 50, 100)


def _sim_error_rate_for_step(deploy_id: int, percent: int) -> float:
    r = random.Random(
        int(hashlib.sha256(f"deploy:{deploy_id}:{percent}".encode()).hexdigest()[:8], 16)
    )
    return round(r.random() * 0.2, 4)


def _rollout_healthy(percent: int, err: float, threshold: float) -> bool:
    if err > threshold and percent < 100:
        return False
    if percent < 100 and err > 0.25:
        return False
    return err <= threshold * 1.1


async def run_canary_simulation(
    session: AsyncSession, deployment: Deployment, target_max_percent: int
) -> Deployment:
    """
    Advance canary in steps until target. Auto-rollback if simulated error high.
    """
    s = get_settings()
    threshold = s.canary_rollback_error_rate
    target = min(100, max(10, target_max_percent))
    for step in CANARY_STEPS:
        if step > target:
            break
        deployment.canary_percent = step
        err = _sim_error_rate_for_step(deployment.id, step)
        deployment.error_rate = err
        if not _rollout_healthy(step, err, threshold):
            deployment.status = DeploymentStatus.rolled_back
            deployment.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
            global_metrics.record_deployment(False)
            await session.flush()
            return deployment
    deployment.error_rate = _sim_error_rate_for_step(deployment.id, 100)
    deployment.status = (
        DeploymentStatus.healthy
        if deployment.error_rate <= threshold
        else DeploymentStatus.degraded
    )
    deployment.canary_percent = 100
    deployment.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    global_metrics.record_deployment(deployment.status == DeploymentStatus.healthy)
    await session.flush()
    return deployment


async def rollback_deployment(
    session: AsyncSession, deployment: Deployment, previous: Optional[Deployment] = None
) -> Deployment:
    deployment.status = DeploymentStatus.rolled_back
    if previous:
        previous.status = DeploymentStatus.healthy
    deployment.error_rate = 0.0
    deployment.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    global_metrics.record_deployment(False)
    await session.flush()
    return deployment
