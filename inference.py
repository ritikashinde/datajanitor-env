import json
import subprocess
import time
import requests
from baseline import run_baseline


BASE_URL = "http://127.0.0.1:8000"


def wait_for_server(timeout=10):
    for _ in range(timeout):
        try:
            res = requests.get(BASE_URL)
            if res.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def main():
    print("[START] Running baseline inference...")

    # Start FastAPI server
    server = subprocess.Popen(
        ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server():
            print("[ERROR] Server failed to start")
            return

        result = run_baseline()

        for task_id, score in result["scores"].items():
            print(f"[STEP] Task={task_id} Score={score}")

        print(f"[END] Average Score={result['average_score']}")

    finally:
        server.terminate()


if __name__ == "__main__":
    main()