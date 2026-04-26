from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.feature_flag import FeatureFlag
from app.schemas.flags import FeatureFlagCreate, FeatureFlagRead, FeatureFlagUpdate, FlagEvaluateBody
from app.services.feature_flags import is_flag_granted
from app.services.hash_util import stable_user_bucket

router = APIRouter(prefix="/api/flags", tags=["feature-flags"])


@router.get("", response_model=list[FeatureFlagRead])
async def list_flags(
    session: AsyncSession = Depends(get_db), env: str | None = Query(None)
) -> list[FeatureFlagRead]:
    q = select(FeatureFlag)
    if env is not None:
        q = q.where(FeatureFlag.environment == env)
    r = await session.execute(q.order_by(FeatureFlag.id.desc()).limit(200))
    return [FeatureFlagRead.model_validate(x) for x in r.scalars().all()]


@router.post("", response_model=FeatureFlagRead, status_code=201)
async def create_flag(
    body: FeatureFlagCreate, session: AsyncSession = Depends(get_db)
) -> FeatureFlagRead:
    f = FeatureFlag(
        name=body.name,
        description=body.description,
        enabled=body.enabled,
        rollout_percentage=body.rollout_percentage,
        environment=body.environment,
    )
    session.add(f)
    await session.flush()
    return FeatureFlagRead.model_validate(f)


@router.patch("/{flag_id}", response_model=FeatureFlagRead)
async def update_flag(
    flag_id: int, body: FeatureFlagUpdate, session: AsyncSession = Depends(get_db)
) -> FeatureFlagRead:
    f = await session.get(FeatureFlag, flag_id)
    if not f:
        raise HTTPException(404, "Flag not found")
    u = body.model_dump(exclude_unset=True)
    for k, v in u.items():
        setattr(f, k, v)
    await session.flush()
    return FeatureFlagRead.model_validate(f)


@router.post("/evaluate", response_model=dict)
async def evaluate_flag(
    body: FlagEvaluateBody, session: AsyncSession = Depends(get_db)
) -> dict:
    f = await session.get(FeatureFlag, body.flag_id)
    if not f:
        raise HTTPException(404, "Flag not found")
    if not f.enabled or f.rollout_percentage <= 0:
        return {"granted": False, "rollout": f.rollout_percentage, "user_bucket": 0}
    bucket = stable_user_bucket(f"{body.user_id}:flag:{f.id}", 100)
    granted = is_flag_granted(
        f.rollout_percentage, body.user_id, f.id, f.enabled
    )
    return {
        "granted": bool(granted),
        "rollout": f.rollout_percentage,
        "user_bucket": bucket,
        "flag": f.name,
        "environment": f.environment,
    }
