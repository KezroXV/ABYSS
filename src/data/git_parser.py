import requests
import os
from dotenv import load_dotenv
import base64
from radon.complexity import cc_visit

load_dotenv()

EXTENTIONS_CODE = [".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss", ".sass", ".less", ".styl", ".php", ".java", ".cpp", ".c", ".h", ".hpp", ".hxx", ".hh", ".cxx", ".cc", ".c++", ".c++0x", ".c++11", ".c++14", ".c++17", ".c++20"]

DEFAULT_METRICS = {
    "files_changed": 0,
    "lines_added": 0,
    "lines_removed": 0,
    "avg_cyclomatic_complexity": 0.0
}

def parse_github_url(repo_url):

  url = repo_url.rstrip("/").replace(".git","")

  parts = url.split("/")

  owner = parts[-2]
  repo = parts[-1]

  return owner, repo

def get_commit_data(owner, repo, sha, github_token):
  url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
  headers = {}
  if github_token:
    headers["Authorization"] = f"Bearer {github_token}"

  response = requests.get(url, headers=headers or None)
  
  if response.status_code == 200:
    return response.json()
  elif response.status_code == 404:
        print(f"Commit {sha} introuvable.")
  elif response.status_code == 403:
        print("Rate limit atteint ou accès refusé.")
  elif response.status_code == 401:
        print("Token invalide.")
  elif response.status_code == 422:
    print(f"SHA {sha} invalide ou mal formaté.")  
  else:
        print(f"Erreur inattendue : {response.status_code}")
    
  return None

def extract_file_metrics(commit_data):
  files = commit_data.get("files", [])

  files_changed = len(files)
  lines_added = sum(f.get("additions", 0) for f in files)
  lines_removed = sum(f.get("deletions", 0) for f in files)

  return files_changed, lines_added, lines_removed

def get_avg_complexity(owner, repo, sha, files, github_token):
  headers = {}
  if github_token:
    headers = {"Authorization": f"Bearer {github_token}"}
  complexities = []

  for file in files: 
      path = file.get("filename","")

      if not any(path.endswith(ext) for ext in EXTENTIONS_CODE):
        continue

      url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={sha}"
      response = requests.get(url, headers=headers or None)

      if response.status_code != 200:
        continue

      try: 
        content_b64 = response.json().get("content","")
        code = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        results = cc_visit(code)

        if results:
          avg = sum(r.complexity for r in results) / len(results)
          complexities.append(avg)
      except Exception:
        continue

  if complexities:
        return sum(complexities) / len(complexities)
  return 0.0

def get_commit_metrics(repo_url, sha, github_token):
  owner, repo = parse_github_url(repo_url)

  commit_data = get_commit_data(owner, repo, sha, github_token)
  if commit_data is None:
    return DEFAULT_METRICS

  files_changed, lines_added, lines_removed = extract_file_metrics(commit_data)
  avg_complexity = get_avg_complexity(owner, repo, sha, commit_data.get("files", []), github_token)
  return {
    "files_changed": files_changed,
    "lines_added": lines_added,
    "lines_removed": lines_removed,
    "avg_cyclomatic_complexity": avg_complexity
  }

if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    print("Test SHA valide:")
    result = get_commit_metrics("https://github.com/facebook/react", "da64117876002bd60baa9e93cad77421bbf0293f", token)
    print(result)

    print("Test SHA invalide:")
    result = get_commit_metrics("https://github.com/facebook/react", "0000000", token)
    print(result)
