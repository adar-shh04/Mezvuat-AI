"""ModelMesh outbox: feedback batches, cache purges and feedback deletion (M8)

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "modelmesh_outbox",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=12), server_default="queued", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("kind IN ('cache_invalidate', 'feedback_delete', 'feedback_post')", name="ck_outbox_kind"),
        sa.CheckConstraint("status IN ('queued', 'done', 'failed')", name="ck_outbox_status"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_outbox_claim", "modelmesh_outbox", ["status", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_outbox_claim", table_name="modelmesh_outbox")
    op.drop_table("modelmesh_outbox")
