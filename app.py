"""
app.py — DataJanitorEnv FastAPI application.

Action dispatch is now fully parameterized: the agent sends a structured
ActionModel with action_type + params, and the router calls the appropriate
cleaning primitive with those params unpacked.
"""

from copy import deepcopy
from fastapi import FastAPI, HTTPException

from models import (
    ActionModel,
    ObservationModel,
    StepResponseModel,
    ResetResponseModel,
    TaskModel,
    GraderResponseModel,
    StateResponseModel,
    BaselineResponseModel,
)
from baseline import run_baseline
from tasks import TASKS, load_task_dataset, get_task_config, list_tasks
from grader import grade_task_result
from utils import (
    summarize_dataset,
    remove_duplicates,
    fill_missing_mean,
    fill_missing_mode,
    fill_missing_constant,
    standardize_categories,
    remove_outliers_iqr,
    clip_outliers_iqr,
)


app = FastAPI(title="DataJanitorEnv")


# ---------------------------------------------------------------------------
# Action dispatch table
#
# Each entry maps action_type → a callable(df, **params) → df.
# Params come directly from ActionModel.params (already validated by Pydantic).
# ---------------------------------------------------------------------------

def _dispatch_remove_duplicates(df, **params):
    return remove_duplicates(df, subset=params.get("subset") or None, keep=params.get("keep", "first"))

def _dispatch_fill_missing_mean(df, **params):
    return fill_missing_mean(df, columns=params.get("columns") or None)

def _dispatch_fill_missing_mode(df, **params):
    return fill_missing_mode(df, columns=params.get("columns") or None)

def _dispatch_fill_missing_constant(df, **params):
    return fill_missing_constant(df, value=params["value"], columns=params.get("columns") or None)

def _dispatch_standardize_categories(df, **params):
    return standardize_categories(
        df,
        columns=params.get("columns") or None,
        strategy=params.get("strategy", "title_strip"),
        mapping=params.get("mapping"),
    )

def _dispatch_remove_outliers_iqr(df, **params):
    return remove_outliers_iqr(
        df,
        columns=params.get("columns") or None,
        iqr_multiplier=params.get("iqr_multiplier", 1.5),
    )

def _dispatch_clip_outliers_iqr(df, **params):
    return clip_outliers_iqr(
        df,
        columns=params.get("columns") or None,
        iqr_multiplier=params.get("iqr_multiplier", 1.5),
    )


# Maps action_type → (modifies_df: bool, dispatch_fn | None)
ACTIONS = {
    "inspect_dataset":        (False, None),
    "remove_duplicates":      (True,  _dispatch_remove_duplicates),
    "fill_missing_mean":      (True,  _dispatch_fill_missing_mean),
    "fill_missing_mode":      (True,  _dispatch_fill_missing_mode),
    "fill_missing_constant":  (True,  _dispatch_fill_missing_constant),
    "standardize_categories": (True,  _dispatch_standardize_categories),
    "remove_outliers_iqr":    (True,  _dispatch_remove_outliers_iqr),
    "clip_outliers_iqr":      (True,  _dispatch_clip_outliers_iqr),
    "finish":                 (False, None),   # handled separately
}


# ---------------------------------------------------------------------------
# Environment state (single-session, in-process)
# ---------------------------------------------------------------------------

ENV_STATE = {
    "task_id":    None,
    "df":         None,
    "original_df": None,
    "step_count": 0,
    "max_steps":  0,
    "done":       False,
    "info":       {},
}


def build_observation(task_id, step_count, max_steps, df) -> ObservationModel:
    summary = summarize_dataset(df)
    return ObservationModel(
        task_id=task_id,
        step_count=step_count,
        max_steps=max_steps,
        rows=summary["rows"],
        columns=summary["columns"],
        missing_counts=summary["missing_counts"],
        duplicate_count=summary["duplicate_count"],
        categorical_inconsistencies=summary["categorical_inconsistencies"],
        numeric_outlier_counts=summary["numeric_outlier_counts"],
        quality_score=summary["quality_score"],
        preview=summary["preview"],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "DataJanitorEnv is running"}


@app.get("/tasks", response_model=list[TaskModel])
def get_tasks():
    return list_tasks()


@app.post("/reset", response_model=ResetResponseModel)
def reset(task_id: str = "easy"):
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

    df = load_task_dataset(task_id)
    config = get_task_config(task_id)

    ENV_STATE.update({
        "task_id":     task_id,
        "df":          df.copy(),
        "original_df": deepcopy(df),
        "step_count":  0,
        "max_steps":   config["max_steps"],
        "done":        False,
        "info":        {"message": f"Environment reset for task '{task_id}'"},
    })

    observation = build_observation(task_id, 0, config["max_steps"], ENV_STATE["df"])
    return ResetResponseModel(observation=observation)


@app.get("/state", response_model=StateResponseModel)
def state():
    if ENV_STATE["df"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    observation = build_observation(
        ENV_STATE["task_id"], ENV_STATE["step_count"], ENV_STATE["max_steps"], ENV_STATE["df"]
    )
    return StateResponseModel(observation=observation, done=ENV_STATE["done"], info=ENV_STATE["info"])


@app.post("/step", response_model=StepResponseModel)
def step(action_model: ActionModel):
    if ENV_STATE["df"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    if ENV_STATE["done"]:
        raise HTTPException(status_code=400, detail="Episode already finished. Call /reset again.")

    action_type = action_model.action_type
    params      = action_model.params
    current_df  = ENV_STATE["df"]
    before_quality = summarize_dataset(current_df)["quality_score"]

    if action_type == "finish":
        ENV_STATE["done"] = True
        reward = 0.2 if before_quality >= 0.85 else -0.2
        ENV_STATE["info"] = {"action_taken": action_type, "message": "Episode finished by agent."}
        after_quality = before_quality

    elif action_type in ACTIONS:
        modifies_df, dispatch_fn = ACTIONS[action_type]

        if modifies_df and dispatch_fn is not None:
            try:
                new_df = dispatch_fn(current_df, **params)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))

            after_quality = summarize_dataset(new_df)["quality_score"]
            reward = round(after_quality - before_quality - 0.03, 4)

            # Penalise no-ops (wasted step)
            if after_quality == before_quality:
                reward = round(reward - 0.05, 4)

            ENV_STATE["df"] = new_df
            ENV_STATE["info"] = {
                "action_taken":  action_type,
                "params":        params,
                "quality_before": before_quality,
                "quality_after":  after_quality,
            }
        else:
            # inspect_dataset — read-only, small negative for wasting a step
            after_quality = before_quality
            reward = -0.01
            ENV_STATE["info"] = {"action_taken": action_type, "params": params}

    else:
        # Should be unreachable — Pydantic rejects unknown action_types at parse time
        raise HTTPException(status_code=400, detail=f"Invalid action_type: {action_type}")

    ENV_STATE["step_count"] += 1
    if ENV_STATE["step_count"] >= ENV_STATE["max_steps"]:
        ENV_STATE["done"] = True
        ENV_STATE["info"]["message"] = "Maximum steps reached."

    observation = build_observation(
        ENV_STATE["task_id"], ENV_STATE["step_count"], ENV_STATE["max_steps"], ENV_STATE["df"]
    )
    return StepResponseModel(
        observation=observation,
        reward=reward,
        done=ENV_STATE["done"],
        info=ENV_STATE["info"],
    )


@app.get("/grader", response_model=GraderResponseModel)
def grader():
    if ENV_STATE["df"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return grade_task_result(ENV_STATE["task_id"], ENV_STATE["df"])


@app.get("/baseline", response_model=BaselineResponseModel)
def baseline():
    return run_baseline()