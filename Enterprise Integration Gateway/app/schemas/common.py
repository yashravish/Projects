from pydantic import BaseModel, ConfigDict


class OrmBase(BaseModel):
    """Base schema class with ORM mode enabled."""

    model_config = ConfigDict(from_attributes=True)


class PaginatedResponse(OrmBase):
    total: int
    page: int
    page_size: int
    items: list
