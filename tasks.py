from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


TASKS = {
    "easy": {
        "task_id": "easy",
        "name": "Basic Data Cleaning",
        "description": "Clean duplicates, missing values, and simple categorical inconsistencies.",
        "difficulty": "easy",
        "dataset_path": DATA_DIR / "easy.csv",
        "max_steps": 6,
    },
    "medium": {
        "task_id": "medium",
        "name": "Intermediate Data Cleaning",
        "description": "Clean duplicates, missing values, category inconsistencies, and numeric outliers.",
        "difficulty": "medium",
        "dataset_path": DATA_DIR / "medium.csv",
        "max_steps": 7,
    },
    "hard": {
        "task_id": "hard",
        "name": "Advanced Data Cleaning",
        "description": "Clean a noisy dataset with multiple issues under a tighter action budget.",
        "difficulty": "hard",
        "dataset_path": DATA_DIR / "hard.csv",
        "max_steps": 8,
    },
}


def load_task_dataset(task_id: str) -> pd.DataFrame:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")

    dataset_path = TASKS[task_id]["dataset_path"]
    return pd.read_csv(dataset_path)


def get_task_config(task_id: str) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")

    config = TASKS[task_id].copy()
    config["dataset_path"] = str(config["dataset_path"])
    return config


def list_tasks() -> list:
    return [get_task_config(task_id) for task_id in TASKS]