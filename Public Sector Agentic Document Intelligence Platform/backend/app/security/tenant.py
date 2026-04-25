"""Tenant isolation helpers.

Every service method touching tenant-scoped tables takes `organization_id` as
a required parameter. `apply_tenant_filter` is the canonical way to scope a
SQLAlchemy SELECT to a tenant.

`tests/unit/test_tenant_isolation.py` uses AST inspection to fail the build if
any service method touching tenant tables omits `organization_id` from its
signature.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Select


# `Select` is invariant in its row type; using `Any` for this helper is the
# usual pattern for a tenant scoping function shared across every model shape.
def apply_tenant_filter(
    stmt: Select[Any],
    model: type[object],
    organization_id: uuid.UUID,
) -> Select[Any]:
    """Append a `WHERE organization_id = :org_id` predicate.

    Use this in every retrieval and listing query that returns tenant data.
    Centralising it makes the rule auditable.
    """
    org_column = getattr(model, "organization_id", None)
    if org_column is None:
        raise TypeError(
            f"{model.__name__} has no organization_id column; "
            "apply_tenant_filter is for tenant-scoped models only"
        )
    return stmt.where(org_column == organization_id)
