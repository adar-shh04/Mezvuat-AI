"""research runs and llm_call detail (M4)

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_LLM_CALL_COLUMNS: tuple[tuple[str, sa.types.TypeEngine[object]], ...] = (
    ("provider", sa.String(length=32)),
    ("prompt_version", sa.String(length=64)),
    ("iteration", sa.Integer()),
    ("status", sa.String(length=16)),
    ("error_code", sa.String(length=64)),
    ("http_status", sa.Integer()),
    ("finish_reason", sa.String(length=16)),
    ("structured_mode", sa.String(length=24)),
)


def upgrade() -> None:
    op.create_table(
        "research_run",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("workspace_id", sa.String(length=40), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("data_policy", sa.String(length=16), nullable=False),
        sa.Column("retrieval_mode", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("answer", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'failed')", name="ck_run_status"),
        sa.CheckConstraint("data_policy IN ('any', 'local_only')", name="ck_run_data_policy"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_research_run_workspace_id"), "research_run", ["workspace_id"], unique=False)
    for name, type_ in _LLM_CALL_COLUMNS:
        op.add_column("llm_call", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _ in reversed(_LLM_CALL_COLUMNS):
        op.drop_column("llm_call", name)
    op.drop_index(op.f("ix_research_run_workspace_id"), table_name="research_run")
    op.drop_table("research_run")
