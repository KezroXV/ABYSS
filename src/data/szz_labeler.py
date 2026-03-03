import pandas as pd
import requests
import os
import json
import time
import logging
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────
CHECKPOINT_PATH = "data/processed/szz_checkpoint.json"
OUTPUT_PATH     = "data/processed/commits_szz_labeled.csv"
LOG_PATH        = "data/processed/szz_log.txt"

FIX_KEYWORDS = ["fix", "bug", "hotfix", "patch", "repair", "resolve", "issue"]

# Fichiers ignorés pour le blame (trop de faux positifs)
IGNORE_EXTENSIONS = [
    ".json", ".lock", ".md", ".yml", ".yaml",
    ".txt", ".svg", ".png", ".jpg", ".gif",
    ".csv", ".toml", ".ini", ".cfg", ".xml",
    ".html", ".css", ".scss", ".less"
]

MAX_LINES_PER_FILE = 400

# ─── LOGGING ──────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── CHECKPOINT ───────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        log.info(f"Checkpoint trouvé — {len(data['guilty_shas'])} SHA coupables, "
                 f"{len(data['processed_shas'])} fix commits déjà traités")
        return set(data["guilty_shas"]), set(data["processed_shas"])
    return set(), set()

def save_checkpoint(guilty_shas, processed_shas):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "guilty_shas":    list(guilty_shas),
            "processed_shas": list(processed_shas)
        }, f)

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        log.info("Checkpoint supprimé")

# ─── RATE LIMIT ───────────────────────────────────────────
def safe_get(url, headers, retries=3):
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        elif response.status_code == 403:
            wait = max(int(response.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()), 10)
            log.warning(f"Rate limit REST — attente {wait}s...")
            time.sleep(wait)
        elif response.status_code == 404:
            return None
        else:
            log.warning(f"Erreur {response.status_code} — tentative {attempt+1}/{retries}")
            time.sleep(3)
    return None

def safe_post(url, headers, payload, retries=3):
    for attempt in range(retries):
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response
        elif response.status_code == 403:
            wait = max(int(response.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()), 10)
            log.warning(f"Rate limit GraphQL — attente {wait}s...")
            time.sleep(wait)
        else:
            log.warning(f"Erreur GraphQL {response.status_code} — tentative {attempt+1}/{retries}")
            time.sleep(3)
    return None

# ─── CORE FUNCTIONS ───────────────────────────────────────
def is_fix_commit(message):
    return any(k in str(message).lower() for k in FIX_KEYWORDS)

def parse_removed_lines(patch):
    removed = []
    current_line = 0
    for line in patch.split("\n"):
        if line.startswith("@@"):
            try:
                old_part = line.split("@@")[1].strip().split(" ")[0]
                current_line = int(old_part.split(",")[0].replace("-", ""))
            except:
                continue
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(current_line)
            current_line += 1
        elif not line.startswith("+"):
            current_line += 1
    return removed

def get_fix_details(owner, repo, sha, github_token):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    headers = {"Authorization": f"token {github_token}"}

    response = safe_get(url, headers)
    if not response:
        return None, []

    data = response.json()
    parents = data.get("parents", [])
    if not parents:
        return None, []

    parent_sha = parents[0]["sha"]
    files_lines = []

    for file in data.get("files", []):
        path = file.get("filename", "")

        # Ignorer les fichiers non-code
        if any(path.endswith(ext) for ext in IGNORE_EXTENSIONS):
            continue

        patch = file.get("patch", "")
        if not patch:
            continue

        removed_lines = parse_removed_lines(patch)
        if removed_lines:
            # Limiter à MAX_LINES_PER_FILE pour accélérer
            files_lines.append((path, removed_lines[:MAX_LINES_PER_FILE]))

    return parent_sha, files_lines

def get_blame_sha(owner, repo, parent_sha, file_path, line_numbers, github_token):
    query = """
    query($owner: String!, $repo: String!, $sha: String!, $path: String!) {
      repository(owner: $owner, name: $repo) {
        object(expression: $sha) {
          ... on Commit {
            blame(path: $path) {
              ranges {
                startingLine
                endingLine
                commit { oid }
              }
            }
          }
        }
      }
    }
    """
    headers = {
        "Authorization": f"bearer {github_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "variables": {
            "owner": owner,
            "repo":  repo,
            "sha":   parent_sha,
            "path":  file_path
        }
    }

    response = safe_post("https://api.github.com/graphql", headers, payload)
    if not response:
        return set()

    try:
        ranges = response.json()["data"]["repository"]["object"]["blame"]["ranges"]
    except (TypeError, KeyError):
        return set()

    guilty_shas = set()
    for line_num in line_numbers:
        for r in ranges:
            if r["startingLine"] <= line_num <= r["endingLine"]:
                guilty_shas.add(r["commit"]["oid"])
                break

    return guilty_shas

# ─── MAIN PIPELINE ────────────────────────────────────────
def get_szz_guilty_shas(commits_df, github_token):
    guilty_shas, processed_shas = load_checkpoint()

    for repo_name, group in commits_df.groupby("repo"):
        owner, repo = repo_name.split("/")
        fix_commits = group[group["message"].apply(is_fix_commit)]
        total_fix   = len(fix_commits)
        already     = fix_commits["sha"].isin(processed_shas).sum()

        log.info(f"\n[{repo_name}] {total_fix} fix commits — "
                 f"{already} déjà traités, {total_fix - already} restants")

        for i, (_, row) in enumerate(fix_commits.iterrows()):
            sha = row["sha"]

            if sha in processed_shas:
                continue

            log.info(f"  [{i+1}/{total_fix}] {sha[:7]} ({repo_name})")

            parent_sha, files_lines = get_fix_details(owner, repo, sha, github_token)

            if not parent_sha:
                processed_shas.add(sha)
                save_checkpoint(guilty_shas, processed_shas)
                continue

            for file_path, line_numbers in files_lines:
                shas = get_blame_sha(
                    owner, repo, parent_sha,
                    file_path, line_numbers, github_token
                )
                guilty_shas.update(shas)

            processed_shas.add(sha)

            # Checkpoint toutes les 10 fix commits
            if len(processed_shas) % 10 == 0:
                save_checkpoint(guilty_shas, processed_shas)
                log.info(f"  💾 Checkpoint — {len(guilty_shas)} coupables / "
                         f"{len(processed_shas)} traités")

    save_checkpoint(guilty_shas, processed_shas)
    log.info(f"\nTotal commits coupables : {len(guilty_shas)}")
    return guilty_shas

def label_with_szz(commits_df, guilty_shas):
    commits_df["introduced_bug_szz"] = commits_df["sha"].apply(
        lambda sha: 1 if sha in guilty_shas else 0
    )
    return commits_df

# ─── ENTRY POINT ──────────────────────────────────────────
if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")

    df = pd.read_csv("data/processed/commits_labeled.csv")
    log.info(f"Commits chargés : {len(df)}")

    if "repo" not in df.columns:
        log.error("Colonne 'repo' manquante dans commits_labeled.csv — "
                  "relance Labeler.py sur commits_all_repos.csv")
        exit(1)

    guilty_shas = get_szz_guilty_shas(df, token)
    df          = label_with_szz(df, guilty_shas)

    log.info(f"\nDistribution SZZ :\n{df['introduced_bug_szz'].value_counts()}")

    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Sauvegardé dans {OUTPUT_PATH}")

    clear_checkpoint()