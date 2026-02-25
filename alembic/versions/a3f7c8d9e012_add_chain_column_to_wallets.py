"""add_chain_column_to_wallets

Revision ID: a3f7c8d9e012
Revises: 9405ac675852
Create Date: 2026-02-24 10:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f7c8d9e012'
down_revision: Union[str, Sequence[str], None] = '9405ac675852'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add chain column to wallets table."""
    op.add_column('wallets', sa.Column('chain', sa.String(length=20), nullable=False, server_default='base'))


def downgrade() -> None:
    """Remove chain column from wallets table."""
    op.drop_column('wallets', 'chain')
