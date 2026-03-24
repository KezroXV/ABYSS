import os
import sys
import time
from pathlib import Path

# Racine du projet (ABYSS) — nécessaire si le script est lancé avec python path/to/test.py
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from src.data.diff_embedder import get_commit_diff, embed_diff, load_codebert

load_codebert()
token = os.getenv("GITHUB_TOKEN")

df = pd.read_csv("data/processed/commits_features.csv")
sample = df.head(5)

start = time.time()
for _, row in sample.iterrows():
    owner, repo = row["repo"].split("/")
    diff = get_commit_diff(owner, repo, row["sha"], token)
    embed_diff(diff)
    
elapsed = (time.time() - start) / 5
print(f"Temps moyen par commit : {elapsed:.2f}s")
print(f"Estimation 10k commits : {elapsed * 10000 / 3600:.1f}h")