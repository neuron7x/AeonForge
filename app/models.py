from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Index, JSON, Float
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone

Base = declarative_base()

def now_utc():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"
    __table_args__ = (Index("ix_users_tg_id", "tg_id"),)

    id = Column(Integer, primary_key=True)
    tg_id = Column(Integer, unique=True, nullable=False, index=True)
    username = Column(String(200))
    display_name = Column(String(200))
    is_active = Column(Boolean, default=True)
    joined_at = Column(DateTime, default=now_utc)
    reputation_score = Column(Float, default=0.0)
    level = Column(Integer, default=0)
    daily_limit = Column(Integer, default=10)

    assignments = relationship("Assignment", back_populates="user")
    payments = relationship("Payment", back_populates="user")
    batches = relationship("Batch", back_populates="user")

class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (Index("ix_tasks_status", "status"),)

    id = Column(Integer, primary_key=True)
    title = Column(String(300), nullable=False)
    text = Column(Text)
    requirement = Column(String(200))
    status = Column(String(50), default="available", index=True)
    reward_cents = Column(Integer, default=10000)
    created_at = Column(DateTime, default=now_utc)
    payload = Column(JSON, nullable=True)
    level_required = Column(Integer, default=0, index=True)
    deadline_minutes = Column(Integer, default=24 * 60)
    type = Column(String(100), nullable=True)
    bonus_quality_cents = Column(Integer, default=0)
    bonus_speed_cents = Column(Integer, default=0)

    assignments = relationship("Assignment", back_populates="task")

class Batch(Base):
    __tablename__ = "batches"
    __table_args__ = (Index("ix_batches_user_status", "user_id", "status"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=now_utc)
    due_at = Column(DateTime, nullable=True)
    total_tasks = Column(Integer, nullable=False)
    status = Column(String(50), default="open")  # open/completed/paid

    user = relationship("User", back_populates="batches")
    assignments = relationship("Assignment", back_populates="batch")

class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (
        Index("ix_assignments_user_status", "user_id", "status"),
        Index("ix_assignments_task_status", "task_id", "status"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=True)
    assigned_at = Column(DateTime, default=now_utc)
    due_at = Column(DateTime)
    submitted_at = Column(DateTime, nullable=True)
    evidence_url = Column(Text, nullable=True)
    submission_payload = Column(Text, nullable=True)
    status = Column(String(50), default="assigned", index=True)  # assigned/submitted/approved/rejected
    verified_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    qc_report = Column(JSON, nullable=True)
    auto_qc_passed = Column(Boolean, default=False)

    user = relationship("User", back_populates="assignments")
    task = relationship("Task", back_populates="assignments")
    batch = relationship("Batch", back_populates="assignments")

class Payment(Base):
    __tablename__ = "payments"
    __table_args__ = (Index("ix_payments_user_status", "user_id", "status"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    batch_id = Column(Integer, ForeignKey("batches.id"), nullable=True)
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(10), default="USD")
    status = Column(String(50), default="pending", index=True)  # pending/paid/failed
    payout_id = Column(String(200), nullable=True)
    error_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now_utc)
    processed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="payments")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_created", "created_at"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    action = Column(String(100), nullable=False)
    description = Column(Text)
    metadata = Column(Text)
    created_at = Column(DateTime, default=now_utc)
