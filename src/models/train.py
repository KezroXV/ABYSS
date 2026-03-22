import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json

# ← sys.path AVANT les imports src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.dataset import CommitSequenceDataset, SEQUENCE_LENGTH, LABEL_COL
from src.models.lstm_attention import BiLSTMAttention

INPUT_SIZE    = 20
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 5e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce = nn.functional.binary_cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds_proba = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output, _ = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            all_preds_proba.extend(output.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds_proba = np.array(all_preds_proba)
    all_labels      = np.array(all_labels)

    best_f1, best_threshold = 0, 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds     = (all_preds_proba >= threshold).astype(float)
        tp        = ((preds == 1) & (all_labels == 1)).sum()
        fp        = ((preds == 1) & (all_labels == 0)).sum()
        fn        = ((preds == 0) & (all_labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    preds     = (all_preds_proba >= best_threshold).astype(float)
    tp        = ((preds == 1) & (all_labels == 1)).sum()
    fp        = ((preds == 1) & (all_labels == 0)).sum()
    fn        = ((preds == 0) & (all_labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return total_loss / len(loader), f1, precision, recall, best_threshold


def main():
    print(f"Device : {DEVICE}")

    train_dataset = CommitSequenceDataset("data/processed/splits/train.csv")
    val_dataset   = CommitSequenceDataset("data/processed/splits/val.csv")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train : {len(train_dataset)} séquences")
    print(f"Val   : {len(val_dataset)} séquences")

    model = BiLSTMAttention(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5
    )

    best_f1        = 0
    best_threshold = 0.5
    patience       = 7      # arrêter si pas d'amélioration pendant 7 epochs
    no_improve     = 0       # compteur d'epochs sans amélioration
    os.makedirs("models", exist_ok=True)

    print("\nEntraînement...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, f1, precision, recall, threshold = evaluate(model, val_loader, criterion)

        scheduler.step(f1)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"F1: {f1:.4f} | "
              f"Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | "
              f"Threshold: {threshold:.2f}")

        if f1 > best_f1:
            best_f1        = f1
            best_threshold = threshold
            no_improve     = 0  # reset le compteur

            torch.save(model.state_dict(), "models/best_model.pt")

            metadata = {
                "best_f1":         float(f1),
                "best_threshold":  float(threshold),
                "precision":       float(precision),
                "recall":          float(recall),
                "epoch":           epoch,
                "input_size":      INPUT_SIZE,
                "hidden_size":     HIDDEN_SIZE,
                "num_layers":      NUM_LAYERS,
                "dropout":         DROPOUT,
                "sequence_length": SEQUENCE_LENGTH,
                "feature_cols":    train_dataset.feature_cols,
                "label_col":       LABEL_COL
            }
            with open("models/model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"   Meilleur modèle sauvegardé (F1={f1:.4f}, Threshold={threshold:.2f})")

        else:
            no_improve += 1
            print(f"   Pas d'amélioration ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n Early stopping à l'epoch {epoch}")
                break


if __name__ == "__main__":
    main()