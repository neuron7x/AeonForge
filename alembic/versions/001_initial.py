from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tg_id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(200), nullable=True),
        sa.Column('display_name', sa.String(200), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('joined_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tg_id')
    )
    op.create_index('ix_users_tg_id', 'users', ['tg_id'])

    op.create_table('tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(300), nullable=False),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('requirement', sa.String(200), nullable=True),
        sa.Column('status', sa.String(50), default='available'),
        sa.Column('reward_cents', sa.Integer(), default=10000),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tasks_status', 'tasks', ['status'])

    op.create_table('batches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('due_at', sa.DateTime(), nullable=True),
        sa.Column('total_tasks', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(50), default='open'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_batches_user_status', 'batches', ['user_id', 'status'])

    op.create_table('assignments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('task_id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=True),
        sa.Column('assigned_at', sa.DateTime(), nullable=False),
        sa.Column('due_at', sa.DateTime(), nullable=True),
        sa.Column('submitted_at', sa.DateTime(), nullable=True),
        sa.Column('evidence_url', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), default='assigned'),
        sa.Column('verified_by', sa.String(100), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id']),
        sa.ForeignKeyConstraint(['batch_id'], ['batches.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_assignments_user_status', 'assignments', ['user_id', 'status'])
    op.create_index('ix_assignments_task_status', 'assignments', ['task_id', 'status'])

    op.create_table('payments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=True),
        sa.Column('amount_cents', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(10), default='USD'),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('payout_id', sa.String(200), nullable=True),
        sa.Column('error_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['batch_id'], ['batches.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_payments_user_status', 'payments', ['user_id', 'status'])

    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_created', 'audit_logs', ['created_at'])

def downgrade():
    op.drop_index('ix_audit_created', table_name='audit_logs')
    op.drop_table('audit_logs')
    op.drop_index('ix_payments_user_status', table_name='payments')
    op.drop_table('payments')
    op.drop_index('ix_assignments_task_status', table_name='assignments')
    op.drop_index('ix_assignments_user_status', table_name='assignments')
    op.drop_table('assignments')
    op.drop_index('ix_batches_user_status', table_name='batches')
    op.drop_table('batches')
    op.drop_index('ix_tasks_status', table_name='tasks')
    op.drop_table('tasks')
    op.drop_index('ix_users_tg_id', table_name='users')
    op.drop_table('users')
