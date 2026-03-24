import torch
import numpy as np
import requests
import os
import json
import time
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME      = "microsoft/codebert-base"
MAX_LENGTH      = 512
CHECKPOINT_PATH = "data/processed/embeddings_checkpoint.json"
BATCH_SIZE      = 16 if DEVICE.type == "cuda" else 8

print(f"CodeBERT running on : {DEVICE}")

tokenizer = None
codebert  = None


def load_codebert():
    global tokenizer, codebert
    if codebert is not None:
        return
    print("Loading CodeBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    codebert  = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    codebert.eval()
    print(f"CodeBERT loaded on {DEVICE}")


def get_commit_diff(owner: str, repo: str, sha: str, github_token: str) -> str:
    url     = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept":        "application/vnd.github.v3.diff"
    }
    for attempt in range(3):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text[:8000]
        elif response.status_code == 403:
            reset = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait  = max(reset - int(time.time()), 10)
            print(f"Rate limit — waiting {wait}s...")
            time.sleep(wait)
        elif response.status_code == 404:
            return ""
        else:
            time.sleep(3)
    return ""


def fetch_diff_worker(args):
    owner, repo, sha, github_token = args
    try:
        diff = get_commit_diff(owner, repo, sha, github_token)
        return sha, diff
    except Exception:
        return sha, ""


def embed_batch(shas: list, diffs: list) -> dict:
    load_codebert()

    diffs = [d if d.strip() else " " for d in diffs]

    inputs = tokenizer(
        diffs,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = codebert(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return {sha: emb.tolist() for sha, emb in zip(shas, embeddings)}


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        print(f"Checkpoint found — {len(data)} embeddings already computed")
        return data
    return {}


def save_checkpoint(embeddings_dict):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(embeddings_dict, f)


def generate_embeddings_for_dataset(
    commits_df,
    github_token: str,
    output_path: str = "data/processed/commit_embeddings.json",
    max_workers: int = 10
):
    load_codebert()

    embeddings = load_checkpoint()
    pending    = [
        row for _, row in commits_df.iterrows()
        if row["sha"] not in embeddings
    ]
    total = len(commits_df)
    done  = len(embeddings)

    print(f"{done} already done, {len(pending)} remaining out of {total}")

    if not pending:
        print("All embeddings already computed.")
        return embeddings

    print(f"Fetching {len(pending)} diffs in parallel ({max_workers} workers)...")
    diff_map = {}
    tasks = []
    for row in pending:
        try:
            owner, repo = row["repo"].split("/")
            tasks.append((owner, repo, row["sha"], github_token))
        except Exception:
            diff_map[row["sha"]] = ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_diff_worker, t): t for t in tasks}
        fetched = 0
        for future in as_completed(futures):
            sha, diff  = future.result()
            diff_map[sha] = diff
            fetched += 1
            if fetched % 200 == 0:
                print(f"  Diffs fetched: {fetched}/{len(tasks)}")

    print(f"All diffs fetched. Generating embeddings in batches of {BATCH_SIZE}...")

    sha_list  = list(diff_map.keys())
    diff_list = list(diff_map.values())

    for i in range(0, len(sha_list), BATCH_SIZE):
        batch_shas       = sha_list[i : i + BATCH_SIZE]
        batch_diffs      = diff_list[i : i + BATCH_SIZE]
        batch_embeddings = embed_batch(batch_shas, batch_diffs)

        embeddings.update(batch_embeddings)
        done += len(batch_shas)
        print(f"  [{done}/{total}] embeddings generated")

        if done % 500 == 0:
            save_checkpoint(embeddings)

    save_checkpoint(embeddings)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(embeddings, f)

    print(f"\nDone. {len(embeddings)} embeddings saved to {output_path}")
    return embeddings


if __name__ == "__main__":
    import pandas as pd

    token = os.getenv("GITHUB_TOKEN")
    df    = pd.read_csv("data/processed/commits_features.csv")

    print(f"Generating embeddings for {len(df)} commits...")
    generate_embeddings_for_dataset(df, token)