import datetime as dt
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeArticleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    slug: str
    type: str
    content: str
    tags: str
    created_at: dt.datetime
    updated_at: dt.datetime


class KnowledgeArticleCreate(BaseModel):
    title: str
    slug: str
    type: str = "troubleshooting"
    content: str = ""
    tags: str = ""


class KnowledgeArticleUpdate(BaseModel):
    title: str | None = None
    type: str | None = None
    content: str | None = None
    tags: str | None = None
