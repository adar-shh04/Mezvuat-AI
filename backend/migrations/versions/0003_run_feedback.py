"""user feedback on research runs (M7)

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_feedback",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("run_id", sa.String(length=40), nullable=False),
        sa.Column("workspace_id", sa.String(length=40), nullable=False),
        sa.Column("rating", sa.String(length=8), nullable=False),
        sa.Column("claim_id", sa.String(length=40), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("rating IN ('up', 'down')", name="ck_feedback_rating"),
        sa.ForeignKeyConstraint(["run_id"], ["research_run.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_run_feedback_run_id"), "run_feedback", ["run_id"], unique=False)
    op.create_index(op.f("ix_run_feedback_workspace_id"), "run_feedback", ["workspace_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_run_feedback_workspace_id"), table_name="run_feedback")
    op.drop_index(op.f("ix_run_feedback_run_id"), table_name="run_feedback")
    op.drop_table("run_feedback")
