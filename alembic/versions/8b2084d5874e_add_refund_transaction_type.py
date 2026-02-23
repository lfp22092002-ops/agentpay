"""add_refund_transaction_type

Revision ID: 8b2084d5874e
Revises: cb9da47faf74
Create Date: 2026-02-23 08:44:52.403659

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b2084d5874e'
down_revision: Union[str, Sequence[str], None] = 'cb9da47faf74'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'refund'")


def downgrade() -> None:
    """Downgrade schema."""
    pass
