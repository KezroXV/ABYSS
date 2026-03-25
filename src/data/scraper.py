import pandas as pd
import os
import json
import time
import threading
from github import Github
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

CHECKPOINT_PATH = "data/raw/scraper_checkpoint.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        print(f"Checkpoint found — {len(data['completed_repos'])} repos already done")
        return data
    return {"completed_repos": [], "all_commits": []}


def save_checkpoint(completed_repos, all_commits):
    os.makedirs("data/raw", exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed_repos": completed_repos, "all_commits": all_commits}, f)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


class GithubScraper:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN is not set")
        # PyGithub expects a string for `login_or_token` (some versions assert on type).
        self.github = Github(token, per_page=100)
        self.lock   = threading.Lock()
        print("GitHub token loaded")

    def get_commits(self, owner, repo, limit=2000):
        repository = self.github.get_repo(f"{owner}/{repo}")
        commits    = repository.get_commits()
        commits_data = []

        for i, commit in enumerate(commits[:limit]):
            if (i + 1) % 200 == 0:
                print(f"  [{owner}/{repo}] {i+1}/{limit} commits fetched")

            try:
                commits_data.append({
                    "repo":    f"{owner}/{repo}",
                    "sha":     commit.sha or "Unknown",
                    "message": (commit.commit.message or "Unknown").replace("\n", " ")[:500],
                    "author":  commit.commit.author.name if commit.commit.author else "Unknown",
                    "date":    commit.commit.author.date.isoformat()
                               if (commit.commit.author and getattr(commit.commit.author, "date", None))
                               else "Unknown"
                })
            except Exception as e:
                print(f"  Skipping commit {i}: {e}")
                continue

        return commits_data

    def scrape_repo(self, owner, repo, limit):
        for attempt in range(3):
            try:
                print(f"\nScraping {owner}/{repo} (limit={limit})...")
                commits = self.get_commits(owner, repo, limit)
                print(f"  {owner}/{repo} — {len(commits)} commits fetched")
                return commits
            except Exception as e:
                error = str(e)
                if "rate limit" in error.lower() or "403" in error:
                    wait = 3600
                    print(f"  Rate limit on {owner}/{repo} — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Error on {owner}/{repo} (attempt {attempt+1}/3): {e}")
                    time.sleep(10)
        print(f"  Failed to scrape {owner}/{repo} after 3 attempts")
        return []

    def scrape_multiple_repos(self, repos, limit_per_repo=2000, parallel_repos=3):
        checkpoint      = load_checkpoint()
        completed_repos = checkpoint["completed_repos"]
        all_commits     = checkpoint["all_commits"]

        pending = [(o, r, l) for o, r, l in repos if f"{o}/{r}" not in completed_repos]
        print(f"{len(completed_repos)} repos already done, {len(pending)} remaining\n")

        if not pending:
            print("All repos already scraped.")
            return all_commits

        with ThreadPoolExecutor(max_workers=parallel_repos) as executor:
            futures = {
                executor.submit(self.scrape_repo, owner, repo, limit): (owner, repo)
                for owner, repo, limit in pending
            }

            for future in as_completed(futures):
                owner, repo = futures[future]
                try:
                    commits = future.result()
                    with self.lock:
                        all_commits.extend(commits)
                        completed_repos.append(f"{owner}/{repo}")
                        save_checkpoint(completed_repos, all_commits)

                        df_temp = pd.DataFrame(all_commits)
                        df_temp.to_csv("data/raw/commits_all_repos.csv", index=False)
                        print(f"  Checkpoint saved — {len(all_commits)} total commits")
                except Exception as e:
                    print(f"  Unexpected error for {owner}/{repo}: {e}")

        print(f"\nDone. {len(all_commits)} commits across {len(completed_repos)} repos")
        return all_commits


if __name__ == "__main__":
    REPOS = [
        ("torvalds",   "linux",   2000),
        ("keras-team", "keras",   2000),
        ("rails",      "rails",   2000),
        ("golang",     "go",      2000),
        ("rust-lang",  "rust",    2000),
    ]

    scraper     = GithubScraper()
    all_commits = scraper.scrape_multiple_repos(REPOS, parallel_repos=3)

    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(all_commits)
    df.to_csv("data/raw/commits_new_repos.csv", index=False)

    print(f"\nSaved to data/raw/commits_new_repos.csv")
    print(f"Shape : {df.shape}")
    print(df["repo"].value_counts())

    clear_checkpoint()