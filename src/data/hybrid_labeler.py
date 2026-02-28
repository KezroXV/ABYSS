import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def load_and_merge(keywords_path, szz_path):
    df_keywords = pd.read_csv(keywords_path)
    df_szz = pd.read_csv(szz_path)

    # Garder seulement sha + introduced_bug_szz du fichier SZZ
    df_szz = df_szz[["sha", "introduced_bug_szz"]]

    # Fusionner sur sha
    df = pd.merge(df_keywords, df_szz, on="sha", how="left")

    # Si un commit n'est pas dans le SZZ, on considère 0
    df["introduced_bug_szz"] = df["introduced_bug_szz"].fillna(0).astype(int)

    return df

def apply_hybrid_label(df, rule="or"):
    if rule == "or":
        df["introduced_bug_hybrid"] = (
            (df["introduced_bug"] == 1) | (df["introduced_bug_szz"] == 1)
        ).astype(int)
    elif rule == "and":
        df["introduced_bug_hybrid"] = (
            (df["introduced_bug"] == 1) & (df["introduced_bug_szz"] == 1)
        ).astype(int)
    return df

def save_hybrid(df, output_path):
    os.makedirs("data/processed", exist_ok=True)
    cols = ["sha", "message", "author", "date", 
            "introduced_bug", "introduced_bug_szz", "introduced_bug_hybrid"]
    df[cols].to_csv(output_path, index=False)
    print(f"Sauvegardé dans {output_path}")

if __name__ == "__main__":
    df = load_and_merge(
        "data/processed/commits_labeled.csv",
        "data/processed/commits_szz_labeled.csv"
    )

    df = apply_hybrid_label(df, rule="or")

    print("Distribution hybrid :")
    print(df["introduced_bug_hybrid"].value_counts())
    print(f"\nKeywords seuls : {df['introduced_bug'].sum()}")
    print(f"SZZ seuls      : {df['introduced_bug_szz'].sum()}")
    print(f"Hybrid (OR)    : {df['introduced_bug_hybrid'].sum()}")

    save_hybrid(df, "data/processed/commits_hybrid_labeled.csv")