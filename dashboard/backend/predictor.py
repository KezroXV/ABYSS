import torch
import numpy as np
import json
import sys
import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def get_recent_commits(owner: str, repo: str, sha: str, github_token: str, limit: int = 20):
    """Récupère les commits précédents pour construire la séquence."""
    url     = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Authorization": f"token {github_token}"}
    params  = {"sha": sha, "per_page": limit + 1}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return []

    commits = response.json()
    # On exclut le commit courant (index 0) et on garde les précédents
    return commits[1:limit + 1] if len(commits) > 1 else []


def commit_to_feature_vector(github_commit: dict, metrics: dict) -> np.ndarray:
    """Convertit un commit brut GitHub + ses métriques en vecteur numpy."""
    date_str = github_commit.get("commit", {}).get("author", {}).get("date", "")
    try:
        dt   = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        hour = dt.hour
        dow  = dt.weekday()
    except Exception:
        hour, dow = 12, 0

    is_weekend      = int(dow >= 5)
    is_night        = int(hour >= 22 or hour < 6)
    is_friday       = int(dow == 4)
    is_friday_night = int(is_friday and hour >= 17)

    total          = metrics["lines_added"] + metrics["lines_removed"]
    deletion_ratio = metrics["lines_removed"] / total if total > 0 else 0
    log_lines      = np.log1p(total)
    log_files      = np.log1p(metrics["files_changed"])
    is_large       = int(total > 300)
    is_single      = int(metrics["files_changed"] == 1)

    return np.array([
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
        0.0,   # hours_since_last_commit — pas disponible ici
        1,     # author_commit_count     — pas disponible ici
        0.0,   # author_bug_rate         — pas disponible ici
        1,     # is_new_author           — pas disponible ici
        deletion_ratio,
        log_lines,
        log_files,
        is_large,
        is_single,
    ], dtype=np.float32)

def fetch_commit_vector(args):
    """Fetch metrics + build vector pour un commit précédent."""
    repo_url, c, github_token = args
    try:
        c_metrics = get_commit_metrics(repo_url, c["sha"], github_token)
        vec       = commit_to_feature_vector(c, c_metrics)
        return c["sha"], vec
    except Exception:
        return c["sha"], np.zeros(20, dtype=np.float32)

def predict_commit(repo_url: str, sha: str, github_token: str):
    if model is None:
        load()

    owner, repo = parse_github_url(repo_url)
    seq_len     = metadata["sequence_length"]

    # 1. Les deux premiers appels en parallèle — métriques cible + liste précédents
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_metrics  = executor.submit(get_commit_metrics, repo_url, sha, github_token)
        f_previous = executor.submit(
            get_recent_commits, owner, repo, sha, github_token, seq_len - 1
        )
        target_metrics   = f_metrics.result()
        previous_commits = f_previous.result()

    print(f"Target metrics fetched, {len(previous_commits)} previous commits found")

    # 2. Fetch les métriques des commits précédents en parallèle
    tasks   = [(repo_url, c, github_token) for c in previous_commits]
    results = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_commit_vector, task): task for task in tasks}
        for future in as_completed(futures):
            c_sha, vec = future.result()
            results[c_sha] = vec

    # 3. Réaligner dans l'ordre chronologique (reversed = du plus ancien au plus récent)
    sequence_vectors = [
        results[c["sha"]]
        for c in reversed(previous_commits)
        if c["sha"] in results
    ]

    # 4. Ajouter le commit cible en dernier
    target_vec = commit_to_feature_vector(
        {"commit": {"author": {"date": ""}}},
        target_metrics
    )
    sequence_vectors.append(target_vec)

    # 5. Compléter si pas assez d'historique
    while len(sequence_vectors) < seq_len:
        sequence_vectors.insert(0, np.zeros(20, dtype=np.float32))

    sequence = np.array(sequence_vectors[-seq_len:], dtype=np.float32)

    # 6. Prédiction
    threshold = metadata["best_threshold"]
    x         = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output, _ = model(x)

    score    = float(output.item())
    is_risky = score >= threshold

    # 7. Explication
    explanation = explain_commit(model, sequence, metadata["feature_cols"])

    return {
        "sha":               sha,
        "repo":              repo_url,
        "risk_score":        round(score, 4),
        "is_risky":          is_risky,
        "threshold":         threshold,
        "risk_level":        "high" if score >= 0.45 else "medium" if score >= threshold else "low",
        "top_features":      explanation["feature_importance"][:5],
        "attention_weights": explanation["attention_weights"],
        "metrics": {
            "files_changed": target_metrics["files_changed"],
            "lines_added":   target_metrics["lines_added"],
            "lines_removed": target_metrics["lines_removed"],
            "total_lines":   target_metrics["lines_added"] + target_metrics["lines_removed"],
        }
    }

def predict_history(repo_url: str, github_token: str, limit: int = 50):
    owner, repo = parse_github_url(repo_url)
    seq_len     = metadata["sequence_length"]

    # 1. Récupérer plus de commits que demandé pour avoir l'historique
    url      = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers  = {"Authorization": f"token {github_token}"}
    params   = {"per_page": limit + seq_len}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code}")

    all_commits = response.json()

    # 2. Fetch les métriques de TOUS les commits en parallèle
    def fetch_metrics(c):
        try:
            return c["sha"], get_commit_metrics(repo_url, c["sha"], github_token)
        except:
            return c["sha"], {"files_changed": 0, "lines_added": 0, 
                              "lines_removed": 0, "avg_cyclomatic_complexity": 0.0}

    print(f"Fetching métriques pour {len(all_commits)} commits...")
    metrics_map = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_metrics, c): c for c in all_commits}
        done    = 0
        for future in as_completed(futures):
            done += 1
            sha, metrics = future.result()
            metrics_map[sha] = metrics
            print(f"  [{done}/{len(all_commits)}] métriques fetchées")

    # 3. Construire les vecteurs pour tous les commits
    vectors_map = {}
    for c in all_commits:
        sha = c["sha"]
        if sha in metrics_map:
            vectors_map[sha] = commit_to_feature_vector(c, metrics_map[sha])

    # 4. Construire les séquences — on utilise les vrais commits précédents
    # all_commits est trié du plus récent au plus ancien
    # Pour le commit i, ses précédents sont les commits i+1, i+2, ... (plus anciens)
    results = []
    for i, c in enumerate(all_commits[:limit]):
        sha = c["sha"]
        if sha not in vectors_map:
            continue

        # Prendre les seq_len-1 commits précédents (plus anciens = indices i+1 à i+seq_len)
        prev_commits = all_commits[i+1 : i+seq_len]
        
        sequence_vectors = [
            vectors_map[pc["sha"]]
            for pc in reversed(prev_commits)  # du plus ancien au plus récent
            if pc["sha"] in vectors_map
        ]

        # Ajouter le commit courant en dernier
        sequence_vectors.append(vectors_map[sha])

        # Compléter si pas assez
        while len(sequence_vectors) < seq_len:
            sequence_vectors.insert(0, np.zeros(20, dtype=np.float32))

        sequence = np.array(sequence_vectors[-seq_len:], dtype=np.float32)

        # Prédiction
        threshold = metadata["best_threshold"]
        x         = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            output, _ = model(x)

        score    = float(output.item())
        date_str = c.get("commit", {}).get("author", {}).get("date", "")
        author   = c.get("commit", {}).get("author", {}).get("name", "Unknown")
        m        = metrics_map[sha]

        results.append({
            "sha":        sha,
            "sha_short":  sha[:7],
            "author":     author,
            "date":       date_str,
            "message":    c.get("commit", {}).get("message", "")[:80],
            "risk_score": round(score, 4),
            "risk_level": "high" if score >= 0.45 else "medium" if score >= threshold else "low",
            "metrics": {
                "files_changed": m["files_changed"],
                "lines_added":   m["lines_added"],
                "lines_removed": m["lines_removed"],
                "total_lines":   m["lines_added"] + m["lines_removed"],
            }
        })

    results.sort(key=lambda x: x["date"], reverse=True)

    scores     = [r["risk_score"] for r in results]
    high_count = sum(1 for r in results if r["risk_level"] == "high")
    avg_score  = round(sum(scores) / len(scores), 4) if scores else 0

    return {
        "repo":      repo_url,
        "total":     len(results),
        "avg_score": avg_score,
        "high_risk": high_count,
        "commits":   results,
    }