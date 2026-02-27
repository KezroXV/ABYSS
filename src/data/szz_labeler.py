import pandas as pd
import requests
import os 
from dotenv import load_dotenv
from git_parser import parse_github_url

load_dotenv()

FIX_KEYWORDS = ["fix", "bug", "hotfix", "patch", "repair", "resolve", "issue"]

def is_fix_commit(message):
  message_lower = str(message).lower()
  return any(keyword in message_lower for keyword in FIX_KEYWORDS)

def get_fix_details(owner, repo, sha, github_token):
  url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
  headers = {"Authorization": f"token {github_token}"}
    
  response = requests.get(url, headers=headers)
  if response.status_code != 200:
      print(f"Erreur pour {sha[:7]} : {response.status_code}")
      return None, []

  data = response.json()

  parents = data.get("parents", [])
  if not parents:
    return None, []
  parent_sha = parents[0]["sha"]

  files_lines = []
  for file in data.get("files", []):
    path = file.get("filename", "")
    patch = file.get("patch", "")
    if not patch:
      continue

    removed_lines = parse_removed_lines(patch)
    if removed_lines:
      files_lines.append((path, removed_lines))

  return parent_sha, files_lines

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
            commit {
              oid
            }
          }
        }
      }
    }
  }
}
    """
    
    expression = f"{parent_sha}:{file_path}"
    variables = {
    "owner": owner,
    "repo": repo,
    "sha": parent_sha,
    "path": file_path
}
    
    headers = {
        "Authorization": f"bearer {github_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        "https://api.github.com/graphql",
        json={"query": query, "variables": variables},
        headers=headers
    )
    
    if response.status_code != 200:
        return set()
    
    data = response.json()
    

    

    try:
        ranges = data["data"]["repository"]["object"]["blame"]["ranges"]
    except (TypeError, KeyError):
        return set()
    
    # Pour chaque ligne supprimée, trouver le commit qui l'a introduite
    guilty_shas = set()
    for line_num in line_numbers:
        for r in ranges:
            if r["startingLine"] <= line_num <= r["endingLine"]:
                guilty_shas.add(r["commit"]["oid"])
                break
    
    return guilty_shas

def get_szz_guilty_shas(commits_df, owner, repo, github_token):
    guilty_shas = set()
    
    fix_commits = commits_df[commits_df["message"].apply(is_fix_commit)]
    print(f"Fix commits trouvés : {len(fix_commits)}")
    
    for i, (_, row) in enumerate(fix_commits.iterrows()):
        sha = row["sha"]
        print(f"[{i+1}/{len(fix_commits)}] Analyse fix {sha[:7]}...")
        
        parent_sha, files_lines = get_fix_details(owner, repo, sha, github_token)
        if not parent_sha:
            continue
        
        for file_path, line_numbers in files_lines:
            shas = get_blame_sha(owner, repo, parent_sha, file_path, line_numbers, github_token)
            guilty_shas.update(shas)
    
    print(f"Commits coupables trouvés : {len(guilty_shas)}")
    return guilty_shas


def label_with_szz(commits_df, guilty_shas):
    commits_df["introduced_bug_szz"] = commits_df["sha"].apply(
        lambda sha: 1 if sha in guilty_shas else 0
    )
    return commits_df

if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    owner, repo = "facebook", "react"

    df = pd.read_csv("data/processed/commits_labeled.csv")
    df = df.head(100)

    guilty_shas = get_szz_guilty_shas(df, owner, repo, token)
    df = label_with_szz(df, guilty_shas)

    print(f"\nDistribution SZZ :")
    print(df["introduced_bug_szz"].value_counts())

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/commits_szz_labeled.csv", index=False)
    print("Sauvegardé dans commits_szz_labeled.csv")