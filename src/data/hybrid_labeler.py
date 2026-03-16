import pandas as pd
import numpy as np
import os
import json
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from git_parser import get_commit_metrics
from dotenv import load_dotenv
load_dotenv()

CHECKPOINT_PATH = "data/processed/preprocessor_checkpoint.json"
checkpoint_lock = threading.Lock()

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        print(f"Checkpoint trouvé — {len(data['results'])} commits déjà traités")
        return data['results']
    return {}

def save_checkpoint(results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"results": results}, f)

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint supprimé")

def fetch_single_commit(args):
    i, total, sha, repo_url, github_token, checkpoint = args
    if sha in checkpoint:
        return sha, checkpoint[sha], True
    metrics = get_commit_metrics(repo_url, sha, github_token)
    return sha, metrics, False


class DataPreprocessor:

    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        print(f"Commit count loaded : {len(df)}")
        return df

    def clean_data(self, df):
        df.drop_duplicates(subset=['sha'], keep='first', inplace=True)
        df['author'] = df['author'].fillna('Unknown')
        df['message'] = df['message'].fillna('No Message')
        print(f"Commit count after cleaning : {len(df)}")
        return df

    def extract_temporal_features(self, df):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()]
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        print(f"Temporal features extracted")
        return df

    def add_commit_metrics(self, df, github_token):
        checkpoint = load_checkpoint()

        tasks = [
            (i, len(df), row['sha'], f"https://github.com/{row['repo']}", github_token, checkpoint)
            for i, row in df.iterrows()
        ]

        results_dict = {}
        done_count = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_single_commit, task): task for task in tasks}

            for future in as_completed(futures):
                sha, metrics, from_cache = future.result()
                results_dict[sha] = metrics
                done_count += 1

                status = "cache" if from_cache else "API"
                print(f"[{done_count}/{len(df)}] {sha[:7]} — {status}")

                if not from_cache:
                    with checkpoint_lock:
                        checkpoint[sha] = metrics
                        if len(checkpoint) % 100 == 0:
                            save_checkpoint(checkpoint)
                            print(f"  Checkpoint — {len(checkpoint)} traités")

        save_checkpoint(checkpoint)

        # Réaligner dans l'ordre du DataFrame
        results = [results_dict[sha] for sha in df['sha']]

        metrics_df = pd.DataFrame(results)
        df['files_changed'] = metrics_df['files_changed'].values
        df['lines_added'] = metrics_df['lines_added'].values
        df['lines_removed'] = metrics_df['lines_removed'].values
        df['total_lines_changed'] = df['lines_added'] + df['lines_removed']
        df['avg_cyclomatic_complexity'] = metrics_df['avg_cyclomatic_complexity'].values
        df['is_large_commit'] = (df['total_lines_changed'] > 200).astype(int)

        print(f"Commit metrics added (real data)")
        return df

    def save_clean_data(self, df, filepath):
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Clean data saved to {filepath}")
        return df

    def main(self):
        token = os.getenv("GITHUB_TOKEN")

        df = self.load_data("data/processed/commits_hybrid_labeled.csv")
        df = self.clean_data(df)
        df = self.extract_temporal_features(df)
        df = self.add_commit_metrics(df, token)
        self.save_clean_data(df, "data/processed/commits_clean.csv")

        clear_checkpoint()

        print(df.head())
        print(df.info())
        return df


if __name__ == "__main__":
    dp = DataPreprocessor()
    dp.main()
