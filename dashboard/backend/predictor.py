import torch
import numpy as np
import json
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.lstm_attention import BiLSTMAttention
from src.models.explain import explain_commit, load_model_and_metadata
from src.data.git_parser import get_commit_metrics, parse_github_url

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model    = None
metadata = None


def load():
    global model, metadata
    model, metadata = load_model_and_metadata()
    print("Modèle chargé")


def build_feature_vector(metrics: dict, commit_info: dict) -> np.ndarray:
    hour       = commit_info.get("hour", 0)
    dow        = commit_info.get("day_of_week", 0)
    is_weekend = int(dow >= 5)
    is_night   = int(hour >= 22 or hour < 6)
    is_friday  = int(dow == 4)
    is_friday_night = int(is_friday and hour >= 17)

    total = metrics["lines_added"] + metrics["lines_removed"]
    deletion_ratio  = metrics["lines_removed"] / total if total > 0 else 0
    log_lines       = np.log1p(total)
    log_files       = np.log1p(metrics["files_changed"])
    is_large        = int(total > 300)
    is_single       = int(metrics["files_changed"] == 1)

    vector = [
        metrics["files_changed"],
        metrics["lines_added"],
        metrics["lines_removed"],
        total,
        metrics["avg_cyclomatic_complexity"],
        hour,
        dow,
        is_weekend,
        is_night,
        is_friday,
        is_friday_night,
        commit_info.get("hours_since_last_commit", 0),
        commit_info.get("author_commit_count", 1),
        commit_info.get("author_bug_rate", 0.0),
        commit_info.get("is_new_author", 1),
        deletion_ratio,
        log_lines,
        log_files,
        is_large,
        is_single,
    ]

    return np.array(vector, dtype=np.float32)


def predict_commit(repo_url: str, sha: str, github_token: str, recent_commits: list = None):
    if model is None:
        load()

    # Récupérer les métriques du commit
    metrics     = get_commit_metrics(repo_url, sha, github_token)
    owner, repo = parse_github_url(repo_url)

    # Construire le vecteur de features pour ce commit
    commit_info = recent_commits[-1] if recent_commits else {}
    feature_vec = build_feature_vector(metrics, commit_info)

    # Construire la séquence de 20 commits
    seq_len = metadata["sequence_length"]

    if recent_commits and len(recent_commits) >= seq_len:
        sequence = np.array([
            build_feature_vector(
                c.get("metrics", metrics),
                c
            )
            for c in recent_commits[-seq_len:]
        ], dtype=np.float32)
    else:
        # Pas assez d'historique — répéter le commit actuel
        sequence = np.tile(feature_vec, (seq_len, 1))

    # Prédiction
    threshold    = metadata["best_threshold"]
    x            = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output, _ = model(x)

    score     = float(output.item())
    is_risky  = score >= threshold

    # Explication
    explanation = explain_commit(model, sequence, metadata["feature_cols"])

    return {
        "sha":              sha,
        "repo":             repo_url,
        "risk_score":       round(score, 4),
        "is_risky":         is_risky,
        "threshold":        threshold,
        "risk_level":       "high" if score >= 0.45 else "medium" if score >= threshold else "low",
        "top_features":     explanation["feature_importance"][:5],
        "attention_weights": explanation["attention_weights"],
        "metrics": {
            "files_changed":  metrics["files_changed"],
            "lines_added":    metrics["lines_added"],
            "lines_removed":  metrics["lines_removed"],
            "total_lines":    metrics["lines_added"] + metrics["lines_removed"],
        }
    }