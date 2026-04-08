import os
import sys
import uvicorn

# allow importing root-level app.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app  # FastAPI app


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)