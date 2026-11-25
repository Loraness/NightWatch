"""add_fts_search_vector_and_gin_index

Revision ID: 61549cb83b51
Revises: 5508c7dd8305
Create Date: 2026-11-21 15:12:21.496334

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '61549cb83b51'
down_revision: Union[str, Sequence[str], None] = '5508c7dd8305'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        ALTER TABLE document_chunks
        ADD COLUMN IF NOT EXISTS search_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_search_vector_gin
        ON document_chunks USING gin (search_vector)
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_search_vector_gin")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS search_vector")
