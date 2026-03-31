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
    standardize_categories,
    remove_outliers_iqr,
)


app = FastAPI(title="DataJanitorEnv")


ACTIONS = {
    "inspect_dataset": lambda df: df.copy(),
    "remove_duplicates": remove_duplicates,
    "fill_missing_mean": fill_missing_mean,
    "fill_missing_mode": fill_missing_mode,
    "standardize_categories": standardize_categories,
    "remove_outliers_iqr": remove_outliers_iqr,
}


ENV_STATE = {
    "task_id": None,
    "df": None,
    "original_df": None,
    "step_count": 0,
    "max_steps": 0,
    "done": False,
    "info": {},
}


def build_observation(task_id: str, step_count: int, max_steps: int, df) -> ObservationModel:
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

    ENV_STATE["task_id"] = task_id
    ENV_STATE["df"] = df.copy()
    ENV_STATE["original_df"] = deepcopy(df)
    ENV_STATE["step_count"] = 0
    ENV_STATE["max_steps"] = config["max_steps"]
    ENV_STATE["done"] = False
    ENV_STATE["info"] = {"message": f"Environment reset for task '{task_id}'"}

    observation = build_observation(
        task_id=task_id,
        step_count=0,
        max_steps=config["max_steps"],
        df=ENV_STATE["df"],
    )

    return ResetResponseModel(observation=observation)


@app.get("/state", response_model=StateResponseModel)
def state():
    if ENV_STATE["df"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    observation = build_observation(
        task_id=ENV_STATE["task_id"],
        step_count=ENV_STATE["step_count"],
        max_steps=ENV_STATE["max_steps"],
        df=ENV_STATE["df"],
    )

    return StateResponseModel(
        observation=observation,
        done=ENV_STATE["done"],
        info=ENV_STATE["info"],
    )


@app.post("/step", response_model=StepResponseModel)
def step(action_model: ActionModel):
    if ENV_STATE["df"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    if ENV_STATE["done"]:
        raise HTTPException(status_code=400, detail="Episode already finished. Call /reset again.")

    action = action_model.action

    current_df = ENV_STATE["df"]
    before_quality = summarize_dataset(current_df)["quality_score"]

    if action == "finish":
        ENV_STATE["done"] = True
        after_quality = before_quality

        reward = 0.2 if after_quality >= 0.85 else -0.2
        ENV_STATE["info"] = {
            "action_taken": action,
            "message": "Episode finished by agent.",
        }

    elif action in ACTIONS:
        new_df = ACTIONS[action](current_df)
        after_quality = summarize_dataset(new_df)["quality_score"]

        reward = round(after_quality - before_quality - 0.03, 4)

        if after_quality == before_quality and action != "inspect_dataset":
            reward = round(reward - 0.05, 4)

        ENV_STATE["df"] = new_df
        ENV_STATE["info"] = {
            "action_taken": action,
            "quality_before": before_quality,
            "quality_after": after_quality,
        }

    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

    ENV_STATE["step_count"] += 1

    if ENV_STATE["step_count"] >= ENV_STATE["max_steps"]:
        ENV_STATE["done"] = True
        ENV_STATE["info"]["message"] = "Maximum steps reached."

    observation = build_observation(
        task_id=ENV_STATE["task_id"],
        step_count=ENV_STATE["step_count"],
        max_steps=ENV_STATE["max_steps"],
        df=ENV_STATE["df"],
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