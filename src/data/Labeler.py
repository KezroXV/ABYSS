import re
import pandas as pd
import os
import json

# Liste des keywords qui indiquent un "fix commit"
FIX_KEYWORDS = [
    'fix', 'bug', 'hotfix', 'patch',
    'regression', 'crash', 'error',
    'defect', 'broken'
]

def is_fix_commit(message):
    if not isinstance(message, str):
        return False
    
    message = message.lower()
    message = re.sub(r'[^a-z0-9]+', ' ', message)
    words = message.split()
    
    for keyword in FIX_KEYWORDS:
        if keyword in words:
            return True
    return False

def label_commits(commits_df):
    commits_df['introduced_bug'] = 0
    for i in range(1,len(commits_df)):
        if is_fix_commit(commits_df.loc[i,'message']) == True:
            commits_df.loc[i - 1, 'introduced_bug'] = 1
    return commits_df

def main():
  df = pd.read_csv("data/raw/commits.csv")
  fix_flags = df["message"].astype(str).str.lower().apply(is_fix_commit)
  print(f"Fix commits détectés: {fix_flags.sum()} / {len(df)} "
        f"({100 * fix_flags.sum() / len(df):.1f}%)")
  df = label_commits(df)
  os.makedirs("data/processed", exist_ok=True)
  df.to_csv("data/processed/commits_labeled.csv", index=False)
  print( (df['introduced_bug'] == 1).sum())
  total = len(df)
  print(f"Total: {total}, Buggy: {(df['introduced_bug'] == 1).sum()}, Bug rate: {100 * (df['introduced_bug'] == 1).sum() / total:.1f}%")

if __name__ == "__main__":
    main()