import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 20
LABEL_COL = "introduced_bug_hybrid"
EXCLUDE_COLS = ["sha", "message", "author", "date", 
                "introduced_bug", "introduced_bug_szz", 
                "introduced_bug_hybrid"]

def get_feature_columns(df):
    return [col for col in df.columns if col not in EXCLUDE_COLS]

class CommitSequenceDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        self.feature_cols = get_feature_columns(df)
        
        # Extraire features et labels en numpy
        self.features = df[self.feature_cols].fillna(0).values.astype(np.float32)
        self.labels = df[LABEL_COL].values.astype(np.float32)
        
        self.seq_len = SEQUENCE_LENGTH
    
    def __len__(self):
        # Nombre de séquences possibles
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        # Séquence de 20 commits
        x = self.features[idx : idx + self.seq_len]
        # Label du dernier commit de la séquence
        y = self.labels[idx + self.seq_len - 1]
        
        return torch.tensor(x), torch.tensor(y)

if __name__ == "__main__":
    dataset = CommitSequenceDataset("data/processed/splits/train.csv")
    
    print(f"Nombre de séquences : {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Shape d'une séquence : {x.shape}")  # doit être (20, 81)
    print(f"Label correspondant : {y}")
    
    print(f"Nombre de features : {x.shape[1]}")