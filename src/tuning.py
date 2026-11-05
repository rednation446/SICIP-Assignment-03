"""Hyperparameter search for the fine-tuned model with Optuna (TPE sampler and
median pruning). Each trial fine-tunes on a smaller training subset and is
scored on the validation set by PR-AUC."""

import gc

import optuna
import pandas as pd
import torch

from .finetune import fine_tune


def run_search(train_df, val_df, cfg, device):
    space = cfg.search_space
    tune_train = train_df.sample(n=min(cfg.tune_train_size, len(train_df)),
                                 random_state=cfg.seed).reset_index(drop=True)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
            "epochs": trial.suggest_categorical("epochs", space["epochs"]),
            "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"]),
            "warmup_ratio": trial.suggest_float("warmup_ratio", *space["warmup_ratio"]),
        }
        print(f"\nTrial {trial.number}: {params}")
        model = None
        try:
            model, _, history, info = fine_tune(tune_train, val_df, cfg, device,
                                                trial=trial, verbose=False, **params)
            best = max(history, key=lambda h: h["val_pr_auc"])
            trial.set_user_attr("val_f1", best["val_f1"])
            trial.set_user_attr("val_roc_auc", best["val_roc_auc"])
            trial.set_user_attr("best_epoch", info["best_epoch"])
            trial.set_user_attr("train_time_sec", info["train_time_sec"])
            print(f"  -> val PR-AUC {info['best_val_pr_auc']:.4f}")
            return info["best_val_pr_auc"]
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study = optuna.create_study(
        direction="maximize",
        study_name="toxicity_finetune",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
    )
    # always try the default setting first so the search can be compared to it
    study.enqueue_trial({
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs if cfg.epochs in space["epochs"] else space["epochs"][0],
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
    })
    study.optimize(objective, n_trials=cfg.n_trials, gc_after_trial=True)
    return study


def trials_table(study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "state": t.state.name,
               "val_pr_auc": t.value}
        row.update(t.params)
        row["val_f1"] = t.user_attrs.get("val_f1")
        row["best_epoch"] = t.user_attrs.get("best_epoch")
        row["time_sec"] = t.user_attrs.get("train_time_sec")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("val_pr_auc", ascending=False, na_position="last").reset_index(drop=True)


def param_importances(study):
    try:
        return optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"Could not compute parameter importances: {e}")
        return {}
