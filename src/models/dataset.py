import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 20
LABEL_COL = "introduced_bug_hybrid"
EXCLUDE_COLS = ["repo", "sha", "message", "author", "date",
                "introduced_bug", "introduced_bug_szz",
                "introduced_bug_hybrid"]

def get_feature_columns(df):
    return [col for col in df.columns if col not in EXCLUDE_COLS]


class CommitSequenceDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.feature_cols = get_feature_columns(df)
        self.seq_len = SEQUENCE_LENGTH
        self.sequences = []
        self.labels = []

        # Construire les séquences par repo
        for repo_name, group in df.groupby("repo"):
            group = group.sort_values("date").reset_index(drop=True)
            features = group[self.feature_cols].fillna(0).values.astype(np.float32)
            labels = group[LABEL_COL].values.astype(np.float32)

            for i in range(len(features) - self.seq_len):
                self.sequences.append(features[i : i + self.seq_len])
                self.labels.append(labels[i + self.seq_len - 1])

        print(f"Séquences construites : {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


if __name__ == "__main__":
    dataset = CommitSequenceDataset("data/processed/splits/train.csv")

    print(f"Nombre de séquences : {len(dataset)}")

    x, y = dataset[0]
    print(f"Shape d'une séquence : {x.shape}")
    print(f"Label correspondant : {y}")
    print(f"Nombre de features : {x.shape[1]}")