from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, Index, Integer,
                        String, Text)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    __table_args__ = (Index("ix_users_tg_id", "tg_id"),)

    id = Column(Integer, primary_key=True)
    tg_id = Column(Integer, unique=True, nullable=False, index=True)
    username = Column(String(200))
    display_name = Column(String(200))
    is_active = Column(Boolean, default=True)
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    assignments = relationship("Assignment", back_populates="user")
    payments = relationship("Payment", back_populates="user")


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (Index("ix_tasks_status", "status"),)

    id = Column(Integer, primary_key=True)
    title = Column(String(300), nullable=False)
    text = Column(Text)
    requirement = Column(String(200))
    status = Column(String(50), default="available", index=True)
    reward_cents = Column(Integer, default=10000)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    assignments = relationship("Assignment", back_populates="task")


class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (
        Index("ix_assignments_user_status", "user_id", "status"),
        Index("ix_assignments_task_status", "task_id", "status"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    due_at = Column(DateTime)
    submitted_at = Column(DateTime, nullable=True)
    evidence_url = Column(Text, nullable=True)
    status = Column(String(50), default="assigned", index=True)
    verified_by = Column(String(100), nullable=True)

    user = relationship("User", back_populates="assignments")
    task = relationship("Task", back_populates="assignments")


class Payment(Base):
    __tablename__ = "payments"
    __table_args__ = (Index("ix_payments_user_status", "user_id", "status"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assignment_id = Column(Integer, nullable=True)
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(10), default="USD")
    status = Column(String(50), default="pending", index=True)
    payout_id = Column(String(200), nullable=True)
    error_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
