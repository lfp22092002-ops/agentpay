"""hash_api_keys_for_security

Revision ID: b51259e3140d
Revises: a3f7c8d9e012
Create Date: 2026-02-25 06:57:15.628075

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import hashlib


# revision identifiers, used by Alembic.
revision: str = 'b51259e3140d'
down_revision: Union[str, Sequence[str], None] = 'a3f7c8d9e012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Hash existing plaintext API keys and migrate to hash-based storage."""
    # Step 1: Add new columns as nullable first
    op.add_column('agents', sa.Column('api_key_hash', sa.String(length=64), nullable=True))
    op.add_column('agents', sa.Column('api_key_prefix', sa.String(length=12), nullable=True, server_default=''))

    # Step 2: Migrate existing plaintext keys to hashed values
    conn = op.get_bind()
    agents = conn.execute(sa.text("SELECT id, api_key FROM agents WHERE api_key IS NOT NULL"))
    for agent_id, api_key in agents:
        if api_key:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_prefix = api_key[:8] + "..."
            conn.execute(
                sa.text("UPDATE agents SET api_key_hash = :hash, api_key_prefix = :prefix WHERE id = :id"),
                {"hash": key_hash, "prefix": key_prefix, "id": agent_id}
            )

    # Step 3: Make api_key_hash non-nullable now that all rows have values
    op.alter_column('agents', 'api_key_hash', nullable=False)

    # Step 4: Create index on new column, drop old index and column
    op.create_index(op.f('ix_agents_api_key_hash'), 'agents', ['api_key_hash'], unique=True)
    op.drop_index(op.f('ix_agents_api_key'), table_name='agents')
    op.drop_column('agents', 'api_key')


def downgrade() -> None:
    """Can't reverse a hash — just add back the column (empty)."""
    op.add_column('agents', sa.Column('api_key', sa.VARCHAR(length=64), autoincrement=False, nullable=True))
    op.create_index(op.f('ix_agents_api_key'), 'agents', ['api_key'], unique=True)
    op.drop_index(op.f('ix_agents_api_key_hash'), table_name='agents')
    op.drop_column('agents', 'api_key_prefix')
    op.drop_column('agents', 'api_key_hash')
