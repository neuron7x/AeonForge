from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserBase(BaseModel):
    username: Optional[str] = None
    display_name: Optional[str] = None


class UserResponse(UserBase):
    id: int
    tg_id: int
    joined_at: datetime

    class Config:
        from_attributes = True


class TaskBase(BaseModel):
    title: str
    text: Optional[str] = None
    requirement: str
    reward_cents: int = 10000


class TaskResponse(TaskBase):
    id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class AssignmentBase(BaseModel):
    evidence_url: Optional[str] = None


class AssignmentResponse(BaseModel):
    id: int
    user_id: int
    task_id: int
    assigned_at: datetime
    due_at: datetime
    status: str
    evidence_url: Optional[str] = None
    submitted_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PaymentResponse(BaseModel):
    id: int
    user_id: int
    amount_cents: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
