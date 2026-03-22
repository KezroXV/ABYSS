import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.dataset import CommitSequenceDataset, SEQUENCE_LENGTH
from src.models.lstm_attention import BiLSTMAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_model.pt")


def load_model_and_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    m = BiLSTMAttention(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    m.eval()
    return m, meta


def _tensor_sequence(seq):
    return torch.tensor(seq, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def grad_times_input_per_timestep(model, seq):
    x = _tensor_sequence(seq)
    x.requires_grad_(True)
    out, _ = model(x)
    out.backward()
    g = (x.grad * x).squeeze(0).detach().cpu().numpy()
    return g


def feature_ranking_from_grad(model, seq, feature_cols):
    g = grad_times_input_per_timestep(model, seq)
    imp = np.mean(np.abs(g), axis=0)
    signed = np.mean(g, axis=0)
    rows = [
        {"feature": feature_cols[i], "importance": float(imp[i]), "contribution": float(signed[i])}
        for i in range(len(feature_cols))
    ]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def forward_score_and_attention(model, seq):
    x = _tensor_sequence(seq)
    with torch.no_grad():
        out, attn = model(x)
    return float(out.item()), attn.squeeze().cpu().numpy()


def explain_commit(model, seq, feature_cols, top_n=10):
    score, attn = forward_score_and_attention(model, seq)
    feats = feature_ranking_from_grad(model, seq, feature_cols)
    s = float(attn.sum())
    pct = (attn / s * 100.0).tolist() if s > 0 else attn.tolist()
    return {
        "risk_score": round(score, 4),
        "feature_importance": feats[:top_n],
        "attention_weights": pct,
        "most_attended_commit": int(np.argmax(attn)),
    }


def global_importance(model, dataset, feature_cols, n_samples=200):
    n = min(n_samples, len(dataset))
    if n == 0:
        return []
    idxs = np.random.choice(len(dataset), n, replace=False)
    acc = np.zeros(len(feature_cols))
    for i in idxs:
        seq = dataset[i][0].numpy()
        g = grad_times_input_per_timestep(model, seq)
        acc += np.mean(np.abs(g), axis=0)
    acc /= len(idxs)
    out = [{"feature": feature_cols[j], "importance": float(acc[j])} for j in range(len(feature_cols))]
    return sorted(out, key=lambda r: r["importance"], reverse=True)


if __name__ == "__main__":
    model, meta = load_model_and_metadata()
    cols = meta["feature_cols"]

    val = CommitSequenceDataset("data/processed/splits/val.csv")
    seq0 = val[0][0].numpy()
    ex = explain_commit(model, seq0, cols)

    print(f"risk_score={ex['risk_score']}")
    print(f"most_attended_commit index={ex['most_attended_commit']} (seq len {SEQUENCE_LENGTH})")

    print("\nTop features (signed mean grad*input over time)")
    for r in ex["feature_importance"]:
        print(f"  {r['contribution']:+.6f}  {r['feature']}")

    w = ex["attention_weights"]
    ranked = sorted(enumerate(w), key=lambda t: t[1], reverse=True)[:5]
    print("\nAttention %, top timesteps")
    for idx, pct in ranked:
        # idx 0 = oldest in window if data is chronological left-to-right
        print(f"  step {idx}  {pct:.1f}%")

    print("\nGlobal importance (sampled)")
    for r in global_importance(model, val, cols, n_samples=200)[:10]:
        print(f"  {r['feature']:<30} {r['importance']:.6f}")
