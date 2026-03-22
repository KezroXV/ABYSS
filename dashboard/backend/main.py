import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.backend.predictor import predict_commit, load

app = FastAPI(title="ABYSS API", description="Bug risk prediction for any Git commit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


@app.on_event("startup")
def startup():
    load()


class PredictRequest(BaseModel):
    repo_url: str
    sha:      str


class HistoryRequest(BaseModel):
    repo_url: str
    limit:    int = 50


@app.get("/")
def root():
    return {"status": "ABYSS API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_commit(
            repo_url=req.repo_url,
            sha=req.sha,
            github_token=GITHUB_TOKEN
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": "BiLSTM-Attention", "version": "1.0"}