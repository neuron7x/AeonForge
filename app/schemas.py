from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserResponse(BaseModel):
    id: int
    tg_id: int
    username: Optional[str] = None
    display_name: Optional[str] = None
    joined_at: datetime
    class Config:
        from_attributes = True

class TaskResponse(BaseModel):
    id: int
    title: str
    requirement: str
    reward_cents: int
    status: str
    created_at: datetime
    class Config:
        from_attributes = True

class AssignmentResponse(BaseModel):
    id: int
    user_id: int
    task_id: int
    batch_id: Optional[int] = None
    assigned_at: datetime
    due_at: datetime
    status: str
    evidence_url: Optional[str] = None
    submitted_at: Optional[datetime] = None
    class Config:
        from_attributes = True

class BatchResponse(BaseModel):
    id: int
    user_id: int
    total_tasks: int
    status: str
    created_at: datetime
    due_at: Optional[datetime] = None
    class Config:
        from_attributes = True
