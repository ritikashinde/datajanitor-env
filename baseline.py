import requests

BASE_URL = "http://127.0.0.1:8000"


def reset_task(task_id: str):
    response = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    response.raise_for_status()
    return response.json()


def step(action: str):
    response = requests.post(
        f"{BASE_URL}/step",
        json={"action": action}
    )
    response.raise_for_status()
    return response.json()


def get_grader():
    response = requests.get(f"{BASE_URL}/grader")
    response.raise_for_status()
    return response.json()


def choose_action(observation, actions_taken):
    missing = sum(observation["missing_counts"].values())
    duplicates = observation["duplicate_count"]
    outliers = sum(observation["numeric_outlier_counts"].values())
    inconsistencies = observation["categorical_inconsistencies"]

    # 1. First fix category inconsistencies
    if inconsistencies and "standardize_categories" not in actions_taken:
        return "standardize_categories"

    # 2. Then remove duplicates
    if duplicates > 0 and "remove_duplicates" not in actions_taken:
        return "remove_duplicates"

    # 3. Then handle missing values
    # Prefer mean first for harder datasets, then mode if needed
    if missing > 0 and "fill_missing_mean" not in actions_taken:
        return "fill_missing_mean"

    if missing > 0 and "fill_missing_mode" not in actions_taken:
        return "fill_missing_mode"

    # 4. Then remove outliers
    if outliers > 0 and "remove_outliers_iqr" not in actions_taken:
        return "remove_outliers_iqr"

    # 5. If everything looks clean, finish
    if missing == 0 and duplicates == 0 and outliers == 0 and not inconsistencies:
        return "finish"

    # 6. Fallback tries in case some issues still remain
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

    # 7. Last resort
    return "finish"


def run_task(task_id: str):
    reset_task(task_id)

    history = []
    actions_taken = set()

    # Always inspect first
    result = step("inspect_dataset")
    history.append({
        "action": "inspect_dataset",
        "reward": result["reward"],
        "quality_score": result["observation"]["quality_score"],
        "done": result["done"],
    })
    actions_taken.add("inspect_dataset")

    if result["done"]:
        final_result = get_grader()
        return {
            "task_id": task_id,
            "score": final_result["score"],
            "passed": final_result["passed"],
            "history": history,
            "summary": final_result["summary"],
        }

    max_decision_steps = 10

    for _ in range(max_decision_steps):
        observation = result["observation"]
        action = choose_action(observation, actions_taken)

        result = step(action)

        history.append({
            "action": action,
            "reward": result["reward"],
            "quality_score": result["observation"]["quality_score"],
            "done": result["done"],
        })

        actions_taken.add(action)

        if result["done"]:
            break

    final_result = get_grader()

    return {
        "task_id": task_id,
        "score": final_result["score"],
        "passed": final_result["passed"],
        "history": history,
        "summary": final_result["summary"],
    }


def run_baseline():
    task_ids = ["easy", "medium", "hard"]
    details = {}
    total_score = 0.0

    for task_id in task_ids:
        result = run_task(task_id)
        details[task_id] = result
        total_score += result["score"]

    average_score = round(total_score / len(task_ids), 4)

    return {
        "scores": {task_id: details[task_id]["score"] for task_id in task_ids},
        "average_score": average_score,
        "details": details,
    }


if __name__ == "__main__":
    result = run_baseline()
    print(result)