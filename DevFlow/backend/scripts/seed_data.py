"""Load sample data for local demos. Run: python -m scripts.seed_data (from backend/)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.core.database import AsyncSessionFactory, engine
from app import models  # noqa: F401
from app.models.ab_experiment import ABExperiment, ABMetricRollup, ExperimentStatus
from app.models.deployment import Deployment, DeploymentStatus
from app.models.feature_flag import FeatureFlag
from app.models.knowledge_base import ArticleType, KnowledgeBaseArticle
from app.models.project import Project
from app.models.metrics_event import MetricsEvent


async def run() -> None:
    from app.core.database import Base
    from app.core.config import get_settings
    s = get_settings()
    if "sqlite" in s.database_url:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionFactory() as session:
        p = (await session.execute(select(Project).where(Project.slug == "devflow"))).scalar_one_or_none()
        if not p:
            p = Project(
                name="DevFlow",
                slug="devflow",
                description="Default demo project: CI, releases, and observability.",
            )
            session.add(p)
            await session.flush()
        f = (await session.execute(select(FeatureFlag).where(FeatureFlag.name == "dark_mode_ui"))).scalar_one_or_none()
        if not f:
            session.add(
                FeatureFlag(
                    name="dark_mode_ui",
                    description="Themed layout for the dashboard",
                    enabled=True,
                    rollout_percentage=35,
                    environment="default",
                )
            )
        if not (await session.execute(select(KnowledgeBaseArticle).where(KnowledgeBaseArticle.slug == "runbook-failed-deploy"))).scalar_one_or_none():
            session.add(
                KnowledgeBaseArticle(
                    title="Runbook: failed deploy rollback",
                    slug="runbook-failed-deploy",
                    type=ArticleType.runbook,
                    content="1) check error_rate on deployment\n2) trigger rollback if canary is unhealthy",
                    tags="deploy,canary,rollback",
                )
            )
        if not (await session.execute(select(Deployment).where(Deployment.version == "v0.0.0-seed", Deployment.project_id == p.id))).scalar_one_or_none():
            session.add(
                Deployment(
                    project_id=p.id,
                    version="v0.0.0-seed",
                    status=DeploymentStatus.healthy,
                    environment="production",
                    canary_percent=100,
                    error_rate=0.01,
                )
            )
        if not (await session.execute(select(ABExperiment).where(ABExperiment.key == "checkout_flow"))).scalar_one_or_none():
            e = ABExperiment(
                project_id=p.id,
                key="checkout_flow",
                name="Checkout CTA",
                status=ExperimentStatus.running,
                traffic_a_percent=50,
            )
            session.add(e)
            await session.flush()
            for v in ("A", "B"):
                r = (await session.execute(
                    select(ABMetricRollup).where(ABMetricRollup.experiment_id == e.id, ABMetricRollup.variant == v)
                )).scalar_one_or_none()
                if not r:
                    session.add(ABMetricRollup(experiment_id=e.id, variant=v, assignments=200, conversion_count=40 if v == "A" else 32))
        for i in range(5):
            session.add(MetricsEvent(name="http.request.duration", value=40.0 + i, namespace="devflow", labels={"route": "/api/projects"}))
        await session.commit()
    await engine.dispose()
    print("Seed complete.")


if __name__ == "__main__":
    asyncio.run(run())
