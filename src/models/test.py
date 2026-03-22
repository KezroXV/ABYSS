import pandas as pd

train = pd.read_csv("data/processed/splits/train.csv")
tfidf_cols = [c for c in train.columns if c.startswith("tfidf_")]
EXCLUDE = ["repo", "sha", "message", "author", "date",
           "introduced_bug", "introduced_bug_szz", "introduced_bug_hybrid",
           "message_length", "message_word_count",
           "is_short_message", "has_urgent_keyword"] + tfidf_cols

features = [c for c in train.columns if c not in EXCLUDE]
print(f"Nombre de features : {len(features)}")
print(features)