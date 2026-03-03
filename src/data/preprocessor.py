import pandas as pd
import numpy as np
import os
import datetime
from git_parser import get_commit_metrics
from dotenv import load_dotenv
load_dotenv()
class DataPreprocessor:

    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        num_rows = len(df)
        print(f"Commit count loaded : {num_rows}")
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
        print(f"temp features extracted")
        return df
    
    def add_commit_metrics(self, df, github_token):
        results = []
        for i, row in df.iterrows():
            sha = row['sha']
            repo_url = f"https://github.com/{row['repo']}"
            print(f"[{i+1}/{len(df)}] Fetching {sha[:7]} ({row['repo']})...")
            metrics = get_commit_metrics(repo_url, sha, github_token)
            results.append(metrics)
        metrics_df = pd.DataFrame(results)
        df['files_changed'] = metrics_df['files_changed'].values
        df['lines_added'] = metrics_df['lines_added'].values
        df['lines_removed'] = metrics_df['lines_removed'].values
        df['total_lines_changed'] = df['lines_added'] + df['lines_removed']
        df['avg_cyclomatic_complexity'] = metrics_df['avg_cyclomatic_complexity'].values
        df['is_large_commit'] = (df['total_lines_changed'] > 200).astype(int)
        print(f"Commit metrics added (real data)")
        return df

    
    def save_clean_data(self,df, filepath):
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
        print(df.head())
        print(df.info())
        return df

if __name__ == "__main__":
    dp = DataPreprocessor()
    dp.main()