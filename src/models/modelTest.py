import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

train = pd.read_csv("data/processed/splits/train.csv")
val   = pd.read_csv("data/processed/splits/val.csv")

tfidf_cols = [c for c in train.columns if c.startswith("tfidf_")]
EXCLUDE = ["repo", "sha", "message", "author", "date",
           "introduced_bug", "introduced_bug_szz", "introduced_bug_hybrid",
           "message_length", "message_word_count",
           "is_short_message", "has_urgent_keyword"] + tfidf_cols

feature_cols = [c for c in train.columns if c not in EXCLUDE]

X_train = train[feature_cols].fillna(0)
y_train = train["introduced_bug_szz"]
X_val   = val[feature_cols].fillna(0)
y_val   = val["introduced_bug_szz"]

# Calculer le ratio pour scale_pos_weight
ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,  # gère le déséquilibre
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

preds = model.predict(X_val)
print(f"F1 : {f1_score(y_val, preds):.4f}")
print(classification_report(y_val, preds))
