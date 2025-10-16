from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import generate_latest
from starlette.responses import Response

from core.biometric_engine import BiometricEngine, BiometricSample
from core.pomdp_planner import POMDPPlanner, TaskType, Observation
from infrastructure.metrics import (
    REQUESTS_TOTAL, REQUEST_DURATION, ACTIVE_USERS, EOI_SCORE, DELEGATION_ACTIONS
)
from infrastructure.security import verify_jwt

# ---------------------------- Pydantic models --------------------------------
class BiometricInput(BaseModel):
    user_id: str
    hrv_sdnn: float = Field(..., ge=0, le=200)
    hrv_rmssd: float = Field(..., ge=0, le=150)
    rhr: float = Field(..., ge=30, le=150)
    sleep_duration: float = Field(..., ge=0, le=16)
    sleep_efficiency: float = Field(..., ge=0, le=1)
    waso: float = Field(..., ge=0, le=240)
    context_switches: int = Field(0, ge=0, le=100)
    timestamp: Optional[datetime] = None


class TaskRequest(BaseModel):
    user_id: str
    task_type: str
    task_description: str
    observed_hrv: Optional[float] = None
    self_reported_load: Optional[str] = Field(None, pattern="^(low|medium|high)$")
    completion_time: Optional[float] = None


class DelegationResponse(BaseModel):
    task_type: str
    recommended_ai_autonomy: float = Field(..., ge=0, le=1)
    current_eoi: float
    eoi_category: str
    reasoning: str
    estimated_completion_time: float


class SystemStatus(BaseModel):
    status: str
    active_users: int
    total_delegations: int
    average_eoi: float
    uptime_seconds: float


# ---------------------------- global state -----------------------------------
class CBCSystem:
    def __init__(self) -> None:
        self.biometric_engine = BiometricEngine()
        num_particles = int(os.getenv("NUM_PARTICLES", "500"))
        num_simulations = int(os.getenv("NUM_SIMULATIONS", "300"))
        max_depth = int(os.getenv("MAX_DEPTH", "8"))
        self.pomdp_planner = POMDPPlanner(num_particles=num_particles, num_simulations=num_simulations, max_depth=max_depth)
        self.user_beliefs: Dict[str, Any] = {}
        self.total_delegations = 0
        self.start_time = datetime.now()


cbc_system: CBCSystem = CBCSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cbc_system
    cbc_system = CBCSystem()
    yield


app = FastAPI(title="CBC-Ω² API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    endpoint = request.url.path
    method = request.method
    REQUESTS_TOTAL.labels(endpoint=endpoint, method=method).inc()
    with REQUEST_DURATION.labels(endpoint=endpoint).time():
        response = await call_next(request)
    return response


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "cbc-omega-squared"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/biometric/submit")
async def submit_biometric_data(data: BiometricInput, _user=Depends(verify_jwt)) -> Dict[str, Any]:
    sample = BiometricSample(
        timestamp=data.timestamp or datetime.now(),
        hrv_sdnn=data.hrv_sdnn,
        hrv_rmssd=data.hrv_rmssd,
        rhr=data.rhr,
        sleep_duration=data.sleep_duration,
        sleep_efficiency=data.sleep_efficiency,
        waso=data.waso,
        context_switches=data.context_switches,
    )
    if not sample.validate():
        raise HTTPException(status_code=400, detail="Invalid biometric values")
    cbc_system.biometric_engine.add_sample(data.user_id, sample)
    components = cbc_system.biometric_engine.compute_eoi(data.user_id, sample)

    EOI_SCORE.labels(user_id=data.user_id).set(components.eoi)
    ACTIVE_USERS.set(len(cbc_system.biometric_engine.user_samples))

    if data.user_id not in cbc_system.user_beliefs:
        belief = cbc_system.pomdp_planner.initialize_belief(initial_eoi=components.eoi)
        cbc_system.user_beliefs[data.user_id] = belief

    return {"user_id": data.user_id, "eoi_components": components.as_dict(), "message": "OK"}


@app.post("/delegate/plan", response_model=DelegationResponse)
async def plan_delegation(request: TaskRequest, _user=Depends(verify_jwt)):
    if request.user_id not in cbc_system.user_beliefs:
        raise HTTPException(status_code=404, detail="User not found. Submit biometrics first.")
    try:
        task_type = TaskType(request.task_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task type. Allowed: {[t.value for t in TaskType]}")

    belief = cbc_system.user_beliefs[request.user_id]

    if request.observed_hrv is not None and request.self_reported_load and request.completion_time is not None:
        prev_action = next(a for a in cbc_system.pomdp_planner.actions if a.task_type == task_type and abs(a.ai_autonomy - 0.5) < 1e-6)
        obs = Observation(hrv_measurement=request.observed_hrv, self_reported_load=request.self_reported_load, task_completion_time=request.completion_time)
        belief = cbc_system.pomdp_planner.update_belief(belief, prev_action, obs)
        cbc_system.user_beliefs[request.user_id] = belief

    optimal = cbc_system.pomdp_planner.plan(belief, task_type)
    cur_eoi = float(belief.mean_eoi())
    cur_fatigue = float(belief.mean_fatigue())

    if cur_eoi < 0.5:
        category = "Green"; reasoning = "Low load. Balanced autonomy is safe."
    elif cur_eoi < 1.0:
        category = "Yellow"; reasoning = "Moderate load. Prefer medium-to-high autonomy."
    elif cur_eoi < 1.5:
        category = "Orange"; reasoning = "High load. Increase autonomy to reduce burden."
    else:
        category = "Red"; reasoning = "Critical load. Max autonomy with oversight."

    base_time = 1.0
    est = base_time * (1 + cur_eoi * 0.3 + cur_fatigue * 0.2) * (1 - optimal.ai_autonomy * 0.4)

    autonomy_level = "low" if optimal.ai_autonomy < 0.4 else "medium" if optimal.ai_autonomy < 0.7 else "high"
    DELEGATION_ACTIONS.labels(task_type=task_type.value, autonomy_level=autonomy_level).inc()
    cbc_system.total_delegations += 1

    return DelegationResponse(
        task_type=task_type.value,
        recommended_ai_autonomy=float(optimal.ai_autonomy),
        current_eoi=cur_eoi,
        eoi_category=category,
        reasoning=reasoning,
        estimated_completion_time=float(est),
    )


@app.get("/system/status")
async def status(_user=Depends(verify_jwt)) -> SystemStatus:
    if cbc_system.user_beliefs:
        avg_eoi = float(sum(b.mean_eoi() for b in cbc_system.user_beliefs.values()) / len(cbc_system.user_beliefs))
    else:
        avg_eoi = 0.0
    uptime = (datetime.now() - cbc_system.start_time).total_seconds()
    return SystemStatus(
        status="operational",
        active_users=len(cbc_system.biometric_engine.user_samples),
        total_delegations=cbc_system.total_delegations,
        average_eoi=avg_eoi,
        uptime_seconds=float(uptime),
    )
