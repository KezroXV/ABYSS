# ABYSS — Claude Code Context

## What this project is

ABYSS is a machine learning system that predicts bug risk for any GitHub commit before it's merged. It accepts any GitHub repository URL as input and operates entirely via the GitHub API — no cloning required. The goal is a repo-agnostic tool that works on any public GitHub repository.

## Project structure

```
ABYSS/
├── src/
│   ├── data/
│   │   ├── scraper.py           — scrapes commits from GitHub API (multi-repo)
│   │   ├── Labeler.py           — keyword-based bug labeling
│   │   ├── szz_labeler.py       — SZZ algorithm via GitHub GraphQL blame API
│   │   ├── hybrid_labeler.py    — combines keyword + SZZ labels
│   │   ├── git_parser.py        — fetches real commit metrics via GitHub REST API
│   │   ├── preprocessor.py      — cleans data + fetches metrics (parallelized)
│   │   ├── feature_engineer.py  — builds 788 features (20 numerical + 768 CodeBERT)
│   │   ├── diff_embedder.py     — generates CodeBERT embeddings from commit diffs
│   │   ├── data_splitter.py     — temporal train/val/test split (70/15/15)
│   │   └── merge_datasets.py    — merges multiple CSV datasets
│   └── models/
│       ├── dataset.py           — PyTorch Dataset, builds sequences per repo
│       ├── lstm_attention.py    — Bi-LSTM + Attention architecture
│       ├── train.py             — training loop with FocalLoss + early stopping
│       ├── evaluate.py          — evaluation on test set + threshold analysis
│       └── explain.py           — Gradient x Input + Attention weights explainability
├── dashboard/
│   ├── backend/
│   │   ├── main.py              — FastAPI with /predict, /history, /health endpoints
│   │   └── predictor.py        — model inference + real commit history sequences
│   └── frontend/                — Next.js 14 App Router + Tailwind + shadcn/ui
│       └── src/
│           ├── app/
│           │   ├── page.tsx         — home with RepoPicker + CommitPicker
│           │   ├── analyze/page.tsx — commit analysis result page
│           │   └── history/page.tsx — repo commit timeline
│           └── components/
│               ├── Navbar.tsx
│               ├── RepoPicker.tsx   — GitHub repo selector (authenticated users)
│               └── CommitPicker.tsx — commit selector with metadata
├── models/
│   ├── best_model.pt            — saved model weights
│   └── model_metadata.json      — threshold, feature_cols, hyperparameters
├── data/
│   ├── raw/
│   │   ├── commits_all_repos.csv   — 20k commits across 10 repos
│   │   └── commits_new_repos.csv   — additional repos dataset
│   └── processed/
│       ├── commits_labeled.csv
│       ├── commits_szz_labeled.csv
│       ├── commits_hybrid_labeled.csv
│       ├── commits_clean.csv
│       ├── commits_features.csv     — final feature matrix (788 features)
│       ├── commit_embeddings.json   — CodeBERT embeddings {sha: [768 floats]}
│       └── splits/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
└── tests/
```

## Data pipeline — exact execution order

```
scraper.py
    → data/raw/commits_all_repos.csv

Labeler.py
    reads:  data/raw/commits_all_repos.csv
    writes: data/processed/commits_labeled.csv

szz_labeler.py
    reads:  data/processed/commits_labeled.csv
    writes: data/processed/commits_szz_labeled.csv

hybrid_labeler.py
    reads:  data/processed/commits_szz_labeled.csv
    writes: data/processed/commits_hybrid_labeled.csv

preprocessor.py
    reads:  data/processed/commits_hybrid_labeled.csv
    writes: data/processed/commits_clean.csv

diff_embedder.py (run separately — long, uses checkpoint)
    reads:  data/raw/commits_all_repos.csv
    writes: data/processed/commit_embeddings.json

feature_engineer.py
    reads:  data/processed/commits_hybrid_labeled.csv
            data/processed/commits_clean.csv (metrics merge)
            data/processed/commit_embeddings.json (CodeBERT)
    writes: data/processed/commits_features.csv

data_splitter.py
    reads:  data/processed/commits_features.csv
    writes: data/processed/splits/train.csv, val.csv, test.csv
```

## Model architecture

- Bi-LSTM with Attention (PyTorch)
- Input: sequences of 20 consecutive commits, each represented by 788 features
- Features: 20 numerical (temporal, author, code metrics) + 768 CodeBERT embeddings from commit diff
- Sequences are built per-repo — never mix commits from different repos
- Loss: FocalLoss (alpha=0.25, gamma=2.0) for class imbalance
- Optimizer: AdamW with weight decay
- Label: introduced_bug_szz (SZZ blame via GraphQL — no circular leakage)
- Current F1: 0.34 on test set (expected to improve significantly with CodeBERT features)
- Threshold: 0.45 (saved in model_metadata.json)
- INPUT_SIZE = 788 (must match after CodeBERT integration)

## GitHub API usage

All GitHub data is fetched via API — no cloning ever. Two APIs used:

REST API (via requests):

- GET /repos/{owner}/{repo}/commits/{sha} — commit metrics
- GET /repos/{owner}/{repo}/commits/{sha} with Accept: application/vnd.github.v3.diff — raw diff for CodeBERT

GraphQL API:

- blame query on Commit object (not Blob) — used in SZZ labeler
- Requires bearer token authentication

Rate limit: 5000 req/hour with token. All heavy scripts use:

- Parallelism via ThreadPoolExecutor (max_workers=10)
- Checkpoint system to resume on interruption
- Automatic wait on 403 rate limit responses using X-RateLimit-Reset header

## Backend API

FastAPI running on port 8000. GitHub token is optional — passed from frontend via proxy.

Key endpoints:

- POST /predict — body: {repo_url, sha, github_token?}
- GET /history — params: repo_url, limit, github_token?
- GET /health

The predictor fetches the 19 previous commits in parallel to build a real sequence, never repeating the same vector.

## Frontend

Next.js 14 App Router, TypeScript, Tailwind CSS, shadcn/ui.
Auth: NextAuth v5 with GitHub OAuth provider.
GitHub token stays server-side — passed via Next.js API proxy routes to FastAPI.

Three main pages:

- / — home with repo + commit picker (authenticated) or manual input (anonymous)
- /analyze — commit risk analysis result
- /history — repo commit timeline with risk scores

## Code style

- No excessive comments
- No emojis in code
- Natural human coding style — not AI-looking
- Python type hints where it adds clarity
- No magic numbers — use named constants at the top of files

## What not to touch

- The SZZ algorithm logic in szz_labeler.py — it uses GraphQL blame, not REST approximation
- The sequence building logic in dataset.py — commits must be grouped by repo
- The label column used for training — must be introduced_bug_szz, not introduced_bug_hybrid (avoids circular leakage from keyword features)
- model_metadata.json — generated automatically by train.py, never edit manually

## Current blockers

The pipeline needs to run end-to-end on 20k commits with CodeBERT embeddings integrated. The diff_embedder has already generated commit_embeddings.json. The remaining steps are hybrid_labeler → preprocessor → feature_engineer → data_splitter → train.
