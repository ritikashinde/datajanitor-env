import json
from baseline import run_baseline


def main():
    print("[START] Running baseline inference...")

    result = run_baseline()

    for task_id, score in result["scores"].items():
        print(f"[STEP] Task={task_id} Score={score}")

    print(f"[END] Average Score={result['average_score']}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()