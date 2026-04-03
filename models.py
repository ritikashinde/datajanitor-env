from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Action Types  (the complete, closed enum of what the agent can do)
# ---------------------------------------------------------------------------

ActionType = Literal[
    "remove_duplicates",
    "fill_missing_mean",
    "fill_missing_mode",
    "fill_missing_constant",
    "standardize_categories",
    "remove_outliers_iqr",
    "clip_outliers_iqr",
    "inspect_dataset",
    "finish",
]


# ---------------------------------------------------------------------------
# Per-action parameter schemas
# Each action carries ONLY the params it actually needs.
# ---------------------------------------------------------------------------

class RemoveDuplicatesParams(BaseModel):
    """
    subset : list of column names to consider when finding duplicates.
             Empty list → use ALL columns (default behaviour).
    keep   : which duplicate to keep — 'first' | 'last' | 'none'
    """
    subset: List[str] = []
    keep: Literal["first", "last", "none"] = "first"


class FillMissingMeanParams(BaseModel):
    """
    columns : numeric columns to fill. Empty → fill ALL numeric columns.
    """
    columns: List[str] = []


class FillMissingModeParams(BaseModel):
    """
    columns : columns to fill (any dtype). Empty → fill ALL columns.
    """
    columns: List[str] = []


class FillMissingConstantParams(BaseModel):
    """
    Fill missing values with a literal constant.

    columns : columns to target. Empty → fill ALL columns.
    value   : the fill value (str | int | float).
    """
    columns: List[str] = []
    value: Union[str, int, float]


class StandardizeCategoriesParams(BaseModel):
    """
    columns  : categorical columns to standardize.
               Empty → apply to ALL object/category columns.
    strategy : how to canonicalize
        'lower_strip'   → strip whitespace + lowercase          (e.g. ' Male ' → 'male')
        'title_strip'   → strip whitespace + Title Case          (e.g. ' male' → 'Male')
        'upper_strip'   → strip whitespace + UPPERCASE           (e.g. ' male' → 'MALE')
        'map'           → explicit value mapping via `mapping`
    mapping  : required when strategy='map'.
               Dict[column_name → Dict[raw_value → canonical_value]]
               e.g. {"gender": {"M": "Male", "F": "Female", "male": "Male"}}
    """
    columns: List[str] = []
    strategy: Literal["lower_strip", "title_strip", "upper_strip", "map"] = "title_strip"
    mapping: Optional[Dict[str, Dict[str, str]]] = None

    @model_validator(mode="after")
    def mapping_required_for_map_strategy(self) -> "StandardizeCategoriesParams":
        if self.strategy == "map" and not self.mapping:
            raise ValueError("'mapping' is required when strategy='map'")
        return self


class RemoveOutliersIQRParams(BaseModel):
    """
    columns       : numeric columns to check. Empty → ALL numeric columns.
    iqr_multiplier: fence = Q1/Q3 ± multiplier × IQR.  Default 1.5 (Tukey).
    """
    columns: List[str] = []
    iqr_multiplier: float = 1.5

    @field_validator("iqr_multiplier")
    @classmethod
    def multiplier_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("iqr_multiplier must be > 0")
        return v


class ClipOutliersIQRParams(BaseModel):
    """
    Like remove_outliers_iqr but CLIPS values to the fence instead of
    dropping the row.  Useful when row removal is too destructive.

    columns       : numeric columns to clip. Empty → ALL numeric columns.
    iqr_multiplier: fence multiplier. Default 1.5.
    """
    columns: List[str] = []
    iqr_multiplier: float = 1.5

    @field_validator("iqr_multiplier")
    @classmethod
    def multiplier_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("iqr_multiplier must be > 0")
        return v


class InspectDatasetParams(BaseModel):
    """No parameters — observation-only action."""
    pass


class FinishParams(BaseModel):
    """No parameters — terminates the episode."""
    pass


# ---------------------------------------------------------------------------
# Discriminated union — the canonical ActionModel
# ---------------------------------------------------------------------------

# Maps each action_type to its params model for clean dispatch
ACTION_PARAMS_MAP: Dict[str, type] = {
    "remove_duplicates":      RemoveDuplicatesParams,
    "fill_missing_mean":      FillMissingMeanParams,
    "fill_missing_mode":      FillMissingModeParams,
    "fill_missing_constant":  FillMissingConstantParams,
    "standardize_categories": StandardizeCategoriesParams,
    "remove_outliers_iqr":    RemoveOutliersIQRParams,
    "clip_outliers_iqr":      ClipOutliersIQRParams,
    "inspect_dataset":        InspectDatasetParams,
    "finish":                 FinishParams,
}


class ActionModel(BaseModel):
    """
    The single action object the agent sends to POST /step.

    Schema
    ------
    {
        "action_type": "<one of ActionType>",
        "params":      { ... action-specific fields ... }
    }

    Examples
    --------
    # Remove duplicates keeping the last occurrence, scoped to two columns
    {"action_type": "remove_duplicates", "params": {"subset": ["age", "gender"], "keep": "last"}}

    # Fill missing values in 'income' with the column mean
    {"action_type": "fill_missing_mean", "params": {"columns": ["income"]}}

    # Standardize gender column with an explicit value map
    {
        "action_type": "standardize_categories",
        "params": {
            "strategy": "map",
            "mapping": {"gender": {"M": "Male", "F": "Female", "male": "Male", "female": "Female"}}
        }
    }

    # Remove outliers with a tighter fence (1.0× IQR) on age only
    {"action_type": "remove_outliers_iqr", "params": {"columns": ["age"], "iqr_multiplier": 1.0}}

    # Finish the episode
    {"action_type": "finish", "params": {}}
    """
    action_type: ActionType
    params: Dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_and_parse_params(self) -> "ActionModel":
        """
        Validate that `params` conforms to the schema for the given action_type.
        Raises ValueError with a clear message if validation fails.
        Stores the parsed params model back into self.params for downstream use.
        """
        params_cls = ACTION_PARAMS_MAP[self.action_type]
        try:
            parsed = params_cls(**self.params)
            # Store as dict so the model stays JSON-serializable
            self.params = parsed.model_dump()
        except Exception as exc:
            raise ValueError(
                f"Invalid params for action '{self.action_type}': {exc}"
            ) from exc
        return self


# ---------------------------------------------------------------------------
# Observation / Response models (unchanged structure, kept here for cohesion)
# ---------------------------------------------------------------------------

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