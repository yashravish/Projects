"""Auth routes: register, login, refresh, me.

Audit instrumentation: every register/login/refresh emits an audit event
(success or failure). Failed login attempts are particularly important
to record — they're the canonical signal of a credential-stuffing run.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.db.models import User
from app.deps import CurrentUser, SessionDep
from app.observability import audit_emitter
from app.schemas.auth import (
    AccessToken,
    LoginRequest,
    OrganizationOut,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenPair,
    UserOut,
)
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(payload: RegisterRequest, session: SessionDep) -> RegisterResponse:
    try:
        user, organization, access, refresh = await auth_service.register_user(
            session,
            email=payload.email,
            password=payload.password,
            organization_name=payload.organization_name,
        )
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    await audit_emitter.emit(
        session=session,
        organization_id=organization.id,
        actor_id=user.id,
        action="auth.register",
        resource_type="user",
        resource_id=user.id,
        outcome="success",
        metadata={
            "organization_name": organization.name,
            "organization_slug": organization.slug,
            "role": user.role,
        },
    )
    return RegisterResponse(
        user=UserOut.model_validate(user),
        organization=OrganizationOut.model_validate(organization),
        access_token=access,
        refresh_token=refresh,
    )


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginRequest, session: SessionDep) -> TokenPair:
    try:
        user, access, refresh = await auth_service.authenticate(
            session, email=payload.email, password=payload.password
        )
    except auth_service.AuthError as exc:
        # Best-effort attribution of the failure: try to find the user by
        # e-mail so the failed login is recorded under the intended
        # tenant. If the e-mail doesn't exist we cannot record an audit
        # row (no organization to scope it under) — that's acceptable;
        # the HTTP access log still captures the attempt.
        attempted = await session.scalar(
            select(User).where(User.email == payload.email.strip().lower())
        )
        if attempted is not None:
            await audit_emitter.emit(
                session=session,
                organization_id=attempted.organization_id,
                actor_id=attempted.id,
                action="auth.login",
                resource_type="user",
                resource_id=attempted.id,
                outcome="denied",
                metadata={"reason": exc.message[:200]},
            )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    await audit_emitter.emit(
        session=session,
        organization_id=user.organization_id,
        actor_id=user.id,
        action="auth.login",
        resource_type="user",
        resource_id=user.id,
        outcome="success",
        metadata={"role": user.role},
    )
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=AccessToken)
async def refresh(payload: RefreshRequest, session: SessionDep) -> AccessToken:
    try:
        access = await auth_service.refresh_access_token(
            session, refresh_token=payload.refresh_token
        )
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return AccessToken(access_token=access)


@router.get("/me", response_model=UserOut)
async def me(current_user: CurrentUser) -> UserOut:
    return UserOut.model_validate(current_user)
