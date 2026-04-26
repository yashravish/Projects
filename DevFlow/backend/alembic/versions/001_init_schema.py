"""Initial schema (created from SQLAlchemy models).

Revision ID: 001_init
Revises:
Create Date: 2025-01-01

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_init"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from app.core.database import Base
    from app import models  # noqa: F401
    op.get_bind()
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    from app.core.database import Base
    from app import models  # noqa: F401
    Base.metadata.drop_all(bind=op.get_bind())
