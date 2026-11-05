import numpy as np
import pandas as pd
from datasets import load_dataset

from .config import PROJECT_ROOT


def _sample_split(ds, n, rng, toxic_ratio=None, threshold=0.5):
    """Sample n rows from a HF split. If toxic_ratio is given, the sample is
    stratified to that ratio, otherwise the natural distribution is kept."""
    tox = np.asarray(ds["toxicity"], dtype=np.float32)

    if toxic_ratio is None:
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    else:
        pos = np.where(tox >= threshold)[0]
        neg = np.where(tox < threshold)[0]
        n_pos = min(int(round(n * toxic_ratio)), len(pos))
        n_neg = min(n - n_pos, len(neg))
        idx = np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ])
        rng.shuffle(idx)

    sub = ds.select(idx.tolist())
    df = pd.DataFrame({"text": sub["text"], "toxicity": sub["toxicity"]})
    return df


def _clean(df, threshold):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["label"] = (df["toxicity"] >= threshold).astype(int)
    return df


def load_civil_comments(cfg):
    """Returns train / validation / test DataFrames with columns
    text, toxicity, label."""
    raw = load_dataset("google/civil_comments")
    raw = raw.select_columns(["text", "toxicity"])
    rng = np.random.default_rng(cfg.seed)

    train = _sample_split(raw["train"], cfg.train_size, rng,
                          toxic_ratio=cfg.train_toxic_ratio,
                          threshold=cfg.toxicity_threshold)
    val = _sample_split(raw["validation"], cfg.val_size, rng)
    test = _sample_split(raw["test"], cfg.test_size, rng)

    train, val, test = (_clean(d, cfg.toxicity_threshold) for d in (train, val, test))
    return train, val, test


def describe_splits(splits: dict) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        rows.append({
            "split": name,
            "rows": len(df),
            "toxic": int(df["label"].sum()),
            "non_toxic": int((1 - df["label"]).sum()),
            "toxic_%": round(100 * df["label"].mean(), 2),
            "avg_words": round(df["text"].str.split().str.len().mean(), 1),
        })
    return pd.DataFrame(rows)


def load_real_world_texts():
    return pd.read_csv(PROJECT_ROOT / "data" / "real_world_texts.csv")
