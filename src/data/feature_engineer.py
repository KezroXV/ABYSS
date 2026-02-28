import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

def add_author_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    
    author_commit_count = {}
    author_bug_count = {}
    
    commit_counts = []
    bug_rates = []
    
    for _, row in df.iterrows():
        author = row["author"]
        
        # Nombre de commits passés de cet auteur
        count = author_commit_count.get(author, 0)
        commit_counts.append(count)
        
        # Taux de bug historique
        bugs = author_bug_count.get(author, 0)
        rate = bugs / count if count > 0 else 0.0
        bug_rates.append(rate)
        
        # Mettre à jour les compteurs
        author_commit_count[author] = count + 1
        if row["introduced_bug_hybrid"] == 1:
            author_bug_count[author] = bugs + 1
    
    df["author_commit_count"] = commit_counts
    df["author_bug_rate"] = bug_rates
    df["is_new_author"] = (df["author_commit_count"] < 5).astype(int)
    
    print("Author features ajoutées")
    return df

def add_temporal_features(df):
    df["date"] = pd.to_datetime(df["date"], utc=True)
    
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_friday_night"] = ((df["is_friday"] == 1) & (df["hour"] >= 17)).astype(int)
    
    # Délai depuis le commit précédent du même auteur
    df = df.sort_values("date")
    df["prev_commit_date"] = df.groupby("author")["date"].shift(1)
    df["hours_since_last_commit"] = (
        df["date"] - df["prev_commit_date"]
    ).dt.total_seconds() / 3600
    df["hours_since_last_commit"] = df["hours_since_last_commit"].fillna(0)
    df.drop(columns=["prev_commit_date"], inplace=True)
    
    print("Temporal features ajoutées")
    return df

def add_message_features(df):
    URGENT_KEYWORDS = ["hotfix", "quick", "wip", "temp", "hack", 
                       "dirty", "emergency", "critical", "urgent"]
    
    df["message"] = df["message"].fillna("").astype(str)
    
    # Longueur du message
    df["message_length"] = df["message"].apply(len)
    df["message_word_count"] = df["message"].apply(lambda x: len(x.split()))
    
    # Message trop court = mauvais signe
    df["is_short_message"] = (df["message_word_count"] < 5).astype(int)
    
    # Mots urgents
    df["has_urgent_keyword"] = df["message"].apply(
        lambda x: int(any(k in x.lower() for k in URGENT_KEYWORDS))
    )
    
    print("Message features ajoutées")
    return df

def add_tfidf_features(df, n_features=50):
    vectorizer = TfidfVectorizer(
        max_features=n_features,
        stop_words="english",
        min_df=2
    )
    
    tfidf_matrix = vectorizer.fit_transform(df["message"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()]
    )
    
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    print(f"TF-IDF features ajoutées ({n_features} features)")
    return df

def add_code_features(df):
    # Type de fichiers à risque
    HIGH_RISK_EXTENSIONS = [".js", ".jsx", ".ts", ".tsx", ".py", ".java"]
    TEST_EXTENSIONS = [".test.js", ".spec.js", ".test.ts", ".test.py"]
    
    # Ratio lignes supprimées / total (supprimer beaucoup = refactoring risqué)
    df["deletion_ratio"] = df.apply(
        lambda r: r["lines_removed"] / r["total_lines_changed"] 
        if r["total_lines_changed"] > 0 else 0, axis=1
    )
    
    # Taille du commit normalisée
    df["log_lines_changed"] = np.log1p(df["total_lines_changed"])
    df["log_files_changed"] = np.log1p(df["files_changed"])
    
    # Commit très large = risqué
    df["is_large_commit"] = (df["total_lines_changed"] > 300).astype(int)
    
    # Commit qui ne touche qu'un seul fichier
    df["is_single_file"] = (df["files_changed"] == 1).astype(int)
    
    print("Code features ajoutées")
    return df

def run_feature_engineering(input_path, output_path):
    print("Chargement des données...")
    df = pd.read_csv(input_path)
    print(f"Commits chargés : {len(df)}")
    
    # Fusionner avec les métriques de code depuis commits_clean.csv
    df_metrics = pd.read_csv("data/processed/commits_clean.csv")
    df_metrics = df_metrics[["sha", "files_changed", "lines_added", 
                              "lines_removed", "total_lines_changed", 
                              "avg_cyclomatic_complexity"]]
    df = pd.merge(df, df_metrics, on="sha", how="left")
    print(f"Métriques de code fusionnées")
    
    df = add_temporal_features(df)
    df = add_author_features(df)
    df = add_message_features(df)
    df = add_tfidf_features(df, n_features=50)
    df = add_code_features(df)
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset final : {len(df)} commits, {len(df.columns)} features")
    print(f"Sauvegardé dans {output_path}")
    
    return df

if __name__ == "__main__":
    df = run_feature_engineering(
        input_path="data/processed/commits_hybrid_labeled.csv",
        output_path="data/processed/commits_features.csv"
    )
    
    print("\nAperçu des colonnes :")
    print(df.columns.tolist())
    
    print("\nDistribution du label final :")
    print(df["introduced_bug_hybrid"].value_counts())