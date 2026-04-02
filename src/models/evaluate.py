import torch
import numpy as np
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch.utils.data import DataLoader
from src.models.dataset import CommitSequenceDataset
from src.models.lstm_attention import BiLSTMAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_test():
    # Charger les métadonnées
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)

    print("=== Métadonnées du meilleur modèle ===")
    print(f"Epoch         : {metadata['epoch']}")
    print(f"Val F1        : {metadata['best_f1']:.4f}")
    print(f"Val Precision : {metadata['precision']:.4f}")
    print(f"Val Recall    : {metadata['recall']:.4f}")
    print(f"Threshold     : {metadata['best_threshold']:.2f}")

    # Charger le dataset test
    test_dataset = CommitSequenceDataset("data/processed/splits/test.csv")
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"\nTest : {len(test_dataset)} séquences")

    # Charger le modèle
    model = BiLSTMAttention(
        input_size=metadata["input_size"],
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
        dropout=metadata["dropout"]
    ).to(DEVICE)

    model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
    model.eval()

    # Prédictions
    all_preds_proba = []
    all_labels      = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            output, _ = model(x)
            all_preds_proba.extend(output.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds_proba = np.array(all_preds_proba)
    all_labels      = np.array(all_labels)
    threshold       = metadata["best_threshold"]

    preds     = (all_preds_proba >= threshold).astype(float)
    tp        = ((preds == 1) & (all_labels == 1)).sum()
    fp        = ((preds == 1) & (all_labels == 0)).sum()
    fn        = ((preds == 0) & (all_labels == 1)).sum()
    tn        = ((preds == 0) & (all_labels == 0)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / len(all_labels)

    print("\n=== Résultats sur le Test Set ===")
    print(f"F1        : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"\nTP : {int(tp)}  FP : {int(fp)}")
    print(f"FN : {int(fn)}  TN : {int(tn)}")
    print(f"\nSur {int(tp+fn)} vrais bugs -> modele en trouve {int(tp)} ({recall*100:.1f}%)")
    print(f"Sur {int(tp+fp)} alertes   -> {int(tp)} sont reels ({precision*100:.1f}%)")

    print("\n=== Tradeoff Precision/Recall selon le threshold ===")
    for t in np.arange(0.30, 0.80, 0.05):
        p = (all_preds_proba >= t).astype(float)
        tp = ((p == 1) & (all_labels == 1)).sum()
        fp = ((p == 1) & (all_labels == 0)).sum()
        fn = ((p == 0) & (all_labels == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"Threshold {t:.2f} | Precision {prec:.2f} | Recall {rec:.2f} | F1 {f1:.2f} | Alertes {int(tp+fp)}")
    print("\n=== Distribution des probabilités ===")
    print(f"Min  : {all_preds_proba.min():.4f}")
    print(f"Max  : {all_preds_proba.max():.4f}")
    print(f"Mean : {all_preds_proba.mean():.4f}")
    print(f"Std  : {all_preds_proba.std():.4f}")
    
    # Distribution par buckets
    for low, high in [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,1.0)]:
        count = ((all_preds_proba >= low) & (all_preds_proba < high)).sum()
        print(f"[{low:.1f}-{high:.1f}] : {count} commits")
if __name__ == "__main__":
    evaluate_test()