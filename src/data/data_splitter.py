import pandas as pd
import os

INPUT_PATH = "data/processed/commits_features.csv"
LABEL_COL = "introduced_bug_hybrid"
DATE_COL = "date"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def load_and_sort(input_path):
    df = pd.read_csv(input_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True)
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"Commits chargés et triés : {len(df)}")
    print(f"Du {df[DATE_COL].min()} au {df[DATE_COL].max()}")
    return df

def split_temporal(df):
    n = len(df)
    train_end = int(TRAIN_RATIO * n)
    val_end = train_end + int(VAL_RATIO * n)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    return df_train, df_val, df_test

def save_splits(df_train, df_val, df_test):
    os.makedirs("data/processed/splits", exist_ok=True)

    df_train.to_csv("data/processed/splits/train.csv", index=False)
    df_val.to_csv("data/processed/splits/val.csv", index=False)
    df_test.to_csv("data/processed/splits/test.csv", index=False)

    print("Splits sauvegardés dans data/processed/splits/")

def verify_splits(df_train, df_val, df_test):
    print("\n=== Tailles ===")
    print(f"Train : {len(df_train)} commits")
    print(f"Val   : {len(df_val)} commits")
    print(f"Test  : {len(df_test)} commits")

    print("\n=== Plages de dates ===")
    print(f"Train : {df_train[DATE_COL].min()} -> {df_train[DATE_COL].max()}")
    print(f"Val   : {df_val[DATE_COL].min()} -> {df_val[DATE_COL].max()}")
    print(f"Test  : {df_test[DATE_COL].min()} -> {df_test[DATE_COL].max()}")

    print("\n=== Distribution du label ===")
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = df[LABEL_COL].value_counts()
        total = len(df)
        pct = round(counts.get(1, 0) / total * 100, 1)
        print(f"{name} -> {counts.get(1, 0)} bugs / {total} commits ({pct}%)")

    print("\n=== Vérification chronologique ===")
    assert df_train[DATE_COL].max() <= df_val[DATE_COL].min(), " Fuite train → val"
    assert df_val[DATE_COL].max() <= df_test[DATE_COL].min(), " Fuite val → test"
    print(" Ordre chronologique respecté")

if __name__ == "__main__":
    df = load_and_sort(INPUT_PATH)
    df_train, df_val, df_test = split_temporal(df)
    verify_splits(df_train, df_val, df_test)
    save_splits(df_train, df_val, df_test)