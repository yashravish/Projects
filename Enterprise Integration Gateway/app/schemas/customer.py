from datetime import datetime
from typing import Any

from pydantic import EmailStr, Field

from app.schemas.common import OrmBase


class CustomerBase(OrmBase):
    external_id: str = Field(..., description="Identifier from the source system")
    source: str = Field(..., description="Origin system: 'crm' or 'vendor'")
    name: str
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    postal_code: str | None = None
    status: str = "active"
    raw_data: dict[str, Any] | None = None


class CustomerCreate(CustomerBase):
    pass


class CustomerUpdate(OrmBase):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    postal_code: str | None = None
    status: str | None = None


class CustomerResponse(CustomerBase):
    id: int
    created_at: datetime
    updated_at: datetime
