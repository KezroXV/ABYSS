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

    def get_commits(self,owner,repo,limit=10):
        repo = self.get_repository(owner,repo)
        commits = repo.get_commits()
        commits_data = []
        for commit in commits[:limit]:
            commits_data.append({
                "sha": (commit.sha if commit.sha else "Unknown"),
                "message": (commit.commit.message if commit.commit.message else "Unknown"),
                "author": (commit.commit.author.name if commit.commit.author else "Unknown"),
                "date": (commit.commit.author.date.isoformat() if (commit.commit.author and getattr(commit.commit.author, 'date', None)) else "Unknown")
            })
        return commits_data
    def save_commits_to_json(self, owner, repo, file_name, limit=10):
        commits_data = self.get_commits(owner, repo, limit)
        os.makedirs("data/raw", exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(commits_data, f, indent=2, ensure_ascii=False)
        print(f"Commits data saved to {file_name}")
        return commits_data
    def save_commits_to_csv(self, owner, repo, file_name, limit=10):
        commits_data = self.get_commits(owner, repo, limit)
        os.makedirs("data/raw", exist_ok=True)
        with open(file_name, "w") as f:
            csv.writer(f).writerow(commits_data[0].keys())
            for commit in commits_data:
                csv.writer(f).writerow(commit.values())
        print(f"Commits data saved to {file_name}")
        return commits_data

if __name__ == "__main__":
  G1 = GithubScraper() 
  print(G1.get_repository("facebook", "react"))
  print(G1.get_commits("facebook", "react"))
  G1.save_commits_to_json("facebook", "react", "data/raw/commits.json", 10)
  G1.save_commits_to_csv("facebook", "react", "data/raw/commits.csv", 10)