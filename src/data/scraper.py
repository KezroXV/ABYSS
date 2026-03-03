import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import time
from github import Github, Auth
import json
import csv
load_dotenv()
class GithubScraper:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        
        if not token:
            raise ValueError("GITHUB_TOKEN is not set")
        else:
            print("Github token is set")
        auth = Auth.Token(token)
        self.github = Github(auth=auth)
        
    def get_repository(self, owner, repo):
      return self.github.get_repo(f"{owner}/{repo}")

    def get_commits(self, owner, repo, limit=2000):
        repository = self.get_repository(owner, repo)
        commits = repository.get_commits()
        commits_data = []
        for i, commit in enumerate(commits[:limit]):
            if (i + 1) % 100 == 0:
                print(f"  → {i + 1}/{limit} commits récupérés...")
            commits_data.append({
                "repo": f"{owner}/{repo}",
                "sha": (commit.sha if commit.sha else "Unknown"),
                "message": (commit.commit.message if commit.commit.message else "Unknown"),
                "author": (commit.commit.author.name if commit.commit.author else "Unknown"),
                "date": (commit.commit.author.date.isoformat() if (commit.commit.author and getattr(commit.commit.author, 'date',   None)) else "Unknown")
            })
        return commits_data

    def scrape_multiple_repos(self, repos, limit_per_repo=2000):
        all_commits = []
        os.makedirs("data/raw", exist_ok=True)

        for i, (owner, repo) in enumerate(repos):
            print(f"\n[{i+1}/{len(repos)}] Scraping {owner}/{repo}...")
            try:
                commits = self.get_commits(owner, repo, limit_per_repo)
                all_commits.extend(commits)

                # Sauvegarde intermédiaire après chaque repo
                df_temp = pd.DataFrame(all_commits)
                df_temp.to_csv("data/raw/commits_all_repos.csv", index=False)
                print(f" {len(commits)} commits récupérés — sauvegarde intermédiaire OK")

            except Exception as e:
                print(f" Erreur sur {owner}/{repo} : {e}")
                continue
            
        print(f"\nTotal : {len(all_commits)} commits sur {len(repos)} repos")
        return all_commits
            
        print(f"\nTotal : {len(all_commits)} commits sur {len(repos)} repos")
        return all_commits
    def save_commits_to_json(self, owner, repo, file_name, limit=2000):
        commits_data = self.get_commits(owner, repo, limit)
        os.makedirs("data/raw", exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(commits_data, f, indent=2, ensure_ascii=False)
        print(f"Commits data saved to {file_name}")
        return commits_data
    def save_commits_to_csv(self, owner, repo, file_name, limit=2000):
        commits_data = self.get_commits(owner, repo, limit)
        os.makedirs("data/raw", exist_ok=True)
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=commits_data[0].keys())
            writer.writeheader() 
            writer.writerows(commits_data)  
        print(f"Commits data saved to {file_name}")
        return commits_data

if __name__ == "__main__":
    REPOS = [
        ("facebook", "react"),
        ("microsoft", "vscode"),
        ("vuejs", "vue"),
        ("django", "django"),
        ("expressjs", "express"),
    ]
    
    scraper = GithubScraper()
    all_commits = scraper.scrape_multiple_repos(REPOS, limit_per_repo=2000)
    
    # Sauvegarder tout en un seul CSV
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame(all_commits)
    df.to_csv("data/raw/commits_all_repos.csv", index=False)
    print(f"\nSauvegardé : data/raw/commits_all_repos.csv")
    print(f"Shape : {df.shape}")
    print(df["repo"].value_counts())