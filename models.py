from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ActionModel(BaseModel):
    action: str


class ObservationModel(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    rows: int
    columns: List[str]
    missing_counts: Dict[str, int]
    duplicate_count: int
    categorical_inconsistencies: Dict[str, List[str]]
    numeric_outlier_counts: Dict[str, int]
    quality_score: float
    preview: List[Dict[str, Any]]


class StepResponseModel(BaseModel):
    observation: ObservationModel
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponseModel(BaseModel):
    observation: ObservationModel


class TaskModel(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    dataset_path: str
    max_steps: int


class GraderResponseModel(BaseModel):
    task_id: str
    score: float
    passed: bool
    summary: Dict[str, Any]


class BaselineResponseModel(BaseModel):
    scores: Dict[str, float]
    average_score: float
    details: Dict[str, Any]


class StateResponseModel(BaseModel):
    observation: ObservationModel
    done: bool
    info: Dict[str, Any]