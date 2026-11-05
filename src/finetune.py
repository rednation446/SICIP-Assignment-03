"""End-to-end fine-tuning of the full encoder plus classification head."""

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .metrics import compute_metrics
from .utils import set_seed


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        if self.labels is None:
            return self.texts[i], -1
        return self.texts[i], int(self.labels[i])


def make_collate(tokenizer, max_length):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


def build_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "not_toxic", 1: "toxic"},
        label2id={"not_toxic": 0, "toxic": 1},
    ).to(device)
    for p in model.parameters():
        p.requires_grad = True
    return tokenizer, model


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device, max_length=128, batch_size=64,
                  show_progress=False):
    model.eval()
    loader = DataLoader(TextDataset(texts), batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(tokenizer, max_length))
    out = []
    for batch in tqdm(loader, disable=not show_progress, desc="Predict"):
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def _val_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        total += loss.item() * batch["labels"].size(0)
        n += batch["labels"].size(0)
    return total / max(n, 1)


def fine_tune(train_df, val_df, cfg, device, learning_rate=None, batch_size=None,
              epochs=None, weight_decay=None, warmup_ratio=None, trial=None,
              select_metric="pr_auc", verbose=True):
    """Fine-tunes all layers. Returns the model restored to its best epoch
    (by validation `select_metric`), the tokenizer and the training history.

    If an Optuna `trial` is passed the validation score is reported after each
    epoch so unpromising trials can be pruned."""
    lr = learning_rate or cfg.learning_rate
    bs = batch_size or cfg.batch_size
    n_epochs = epochs or cfg.epochs
    wd = cfg.weight_decay if weight_decay is None else weight_decay
    warm = cfg.warmup_ratio if warmup_ratio is None else warmup_ratio

    set_seed(cfg.seed)
    tokenizer, model = build_model(cfg.model_name, device)
    collate = make_collate(tokenizer, cfg.max_length)

    train_loader = DataLoader(TextDataset(train_df["text"], train_df["label"]),
                              batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TextDataset(val_df["text"], val_df["label"]),
                            batch_size=64, shuffle=False, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": wd},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warm * total_steps), total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_score, best_state, best_epoch = -np.inf, None, 0
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        step_losses = []
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", disable=not verbose)
        for batch in bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * batch["labels"].size(0)
            seen += batch["labels"].size(0)
            step_losses.append(loss.item())
            bar.set_postfix(loss=f"{running / seen:.4f}")

        val_probs = predict_proba(model, tokenizer, val_df["text"], device, cfg.max_length)
        val_metrics = compute_metrics(val_df["label"], val_probs)
        record = {
            "epoch": epoch,
            "train_loss": running / seen,
            "val_loss": _val_loss(model, val_loader, device),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "threshold"},
            "step_losses": step_losses,
        }
        history.append(record)
        if verbose:
            print(f"  epoch {epoch}: train_loss={record['train_loss']:.4f} "
                  f"val_loss={record['val_loss']:.4f} val_f1={record['val_f1']:.4f} "
                  f"val_pr_auc={record['val_pr_auc']:.4f}")

        score = val_metrics[select_metric]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        if trial is not None:
            import optuna
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    model.load_state_dict(best_state)
    info = {
        "learning_rate": lr, "batch_size": bs, "epochs": n_epochs,
        "weight_decay": wd, "warmup_ratio": warm,
        "best_epoch": best_epoch, f"best_val_{select_metric}": best_score,
        "train_time_sec": time.time() - start,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return model, tokenizer, history, info
