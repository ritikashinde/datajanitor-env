import subprocess
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
TASKS = ["easy", "medium", "hard"]
BENCHMARK = "datajanitor_env"
MODEL_NAME = "rule_based_baseline"


def wait_for_server(timeout=15):
    for _ in range(timeout):
        try:
            r = requests.get(BASE_URL)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def reset_task(task_name: str):
    r = requests.post(f"{BASE_URL}/reset", params={"task_id": task_name})
    r.raise_for_status()
    return r.json()


def step_env(action: str):
    r = requests.post(f"{BASE_URL}/step", json={"action": action})
    r.raise_for_status()
    return r.json()


def get_grader():
    r = requests.get(f"{BASE_URL}/grader")
    r.raise_for_status()
    return r.json()


def choose_action(observation, actions_taken):
    missing = sum(observation["missing_counts"].values())
    duplicates = observation["duplicate_count"]
    outliers = sum(observation["numeric_outlier_counts"].values())
    inconsistencies = observation["categorical_inconsistencies"]

    if inconsistencies and "standardize_categories" not in actions_taken:
        return "standardize_categories"

    if duplicates > 0 and "remove_duplicates" not in actions_taken:
        return "remove_duplicates"

    if missing > 0 and "fill_missing_mean" not in actions_taken:
        return "fill_missing_mean"

    if missing > 0 and "fill_missing_mode" not in actions_taken:
        return "fill_missing_mode"

    if outliers > 0 and "remove_outliers_iqr" not in actions_taken:
        return "remove_outliers_iqr"

    if missing == 0 and duplicates == 0 and outliers == 0 and not inconsistencies:
        return "finish"

    if duplicates > 0:
        return "remove_duplicates"

    if missing > 0:
        if "fill_missing_mean" in actions_taken and "fill_missing_mode" not in actions_taken:
            return "fill_missing_mode"
        return "fill_missing_mean"

    if inconsistencies:
        return "standardize_categories"

    if outliers > 0:
        return "remove_outliers_iqr"

    return "finish"


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_name: str):
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_task(task_name)
        result = step_env("inspect_dataset")

        rewards.append(result["reward"])
        steps_taken = 1
        log_step(
            step=1,
            action="inspect_dataset",
            reward=result["reward"],
            done=result["done"],
            error=None,
        )

        actions_taken = {"inspect_dataset"}

        for step in range(2, 15):
            if result["done"]:
                break

            observation = result["observation"]
            action = choose_action(observation, actions_taken)

            result = step_env(action)
            rewards.append(result["reward"])
            steps_taken = step
            actions_taken.add(action)

            log_step(
                step=step,
                action=action,
                reward=result["reward"],
                done=result["done"],
                error=None,
            )

            if result["done"]:
                break

        grader = get_grader()
        score = float(grader["score"])
        success = bool(grader["passed"])

    except Exception as exc:
        log_step(
            step=steps_taken + 1,
            action="exception",
            reward=0.00,
            done=True,
            error=str(exc),
        )
        success = False
        score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    server = subprocess.Popen(
        ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server():
            raise RuntimeError("Server failed to start")

        for task in TASKS:
            run_task(task)

    finally:
        server.terminate()
        server.wait(timeout=5)


if __name__ == "__main__":
    main()