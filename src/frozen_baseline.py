"""Transfer learning baseline: the pre-trained encoder is frozen and only used
to produce sentence embeddings. A logistic regression classifier is trained
on top of those embeddings."""

import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def load_frozen_encoder(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return tokenizer, encoder


@torch.no_grad()
def extract_embeddings(texts, tokenizer, encoder, device, max_length=128,
                       batch_size=64, desc="Embedding"):
    """Mean-pooled last hidden state, ignoring padding tokens."""
    feats = []
    texts = list(texts)
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(device)
        hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0)


def train_frozen_baseline(train_df, val_df, test_df, cfg, device):
    tokenizer, encoder = load_frozen_encoder(cfg.model_name, device)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())

    start = time.time()
    X_train = extract_embeddings(train_df["text"], tokenizer, encoder, device,
                                 cfg.max_length, cfg.embed_batch_size, "Train embeddings")
    X_val = extract_embeddings(val_df["text"], tokenizer, encoder, device,
                               cfg.max_length, cfg.embed_batch_size, "Val embeddings")
    X_test = extract_embeddings(test_df["text"], tokenizer, encoder, device,
                                cfg.max_length, cfg.embed_batch_size, "Test embeddings")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=cfg.logreg_C, max_iter=2000, random_state=cfg.seed),
    )
    clf.fit(X_train, train_df["label"].values)
    elapsed = time.time() - start

    info = {
        "encoder_params": n_total,
        "trainable_encoder_params": n_trainable,
        "classifier_params": int(X_train.shape[1] + 1),
        "embedding_dim": int(X_train.shape[1]),
        "train_time_sec": elapsed,
    }
    probs = {
        "val": clf.predict_proba(X_val)[:, 1],
        "test": clf.predict_proba(X_test)[:, 1],
    }
    embeddings = {"train": X_train, "test": X_test}

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return clf, tokenizer, probs, embeddings, info


def predict_frozen(texts, clf, cfg, device):
    tokenizer, encoder = load_frozen_encoder(cfg.model_name, device)
    X = extract_embeddings(texts, tokenizer, encoder, device, cfg.max_length,
                           cfg.embed_batch_size, "Embedding")
    return clf.predict_proba(X)[:, 1]
