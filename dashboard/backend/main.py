import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.backend.predictor import predict_commit, load, predict_history

app = FastAPI(title="ABYSS API", description="Bug risk prediction for any Git commit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    load()


class PredictRequest(BaseModel):
    repo_url: str
    sha:      str
    # Token GitHub de l'utilisateur (optionnel).
    # Pour les repos publics, on peut fonctionner sans token.
    github_token: Optional[str] = None


class HistoryRequest(BaseModel):
    repo_url: str
    limit:    int = 50
    github_token: Optional[str] = None


@app.get("/")
def root():
    return {"status": "ABYSS API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_commit(
            repo_url=req.repo_url,
            sha=req.sha,
            github_token=req.github_token
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": "BiLSTM-Attention", "version": "1.0"}

@app.get("/history")
def history(repo_url: str, limit: int = 50, github_token: Optional[str] = None):
    try:
        result = predict_history(
            repo_url=repo_url,
            github_token=github_token,
            limit=min(limit, 100)
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

