"""Step-by-step pipeline. Every step stores its outputs on a shared `state`
dict and writes figures / tables to disk, so the steps can be run one at a
time from the notebook or all together from run_pipeline.py."""

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import plots
from .data import describe_splits, load_civil_comments, load_real_world_texts
from .explain import (build_explainer, explain_texts, save_text_plot_html,
                      select_examples, token_contributions, top_tokens)
from .finetune import fine_tune, predict_proba
from .frozen_baseline import train_frozen_baseline
from .metrics import best_f1_threshold, compute_metrics
from .tuning import param_importances, run_search, trials_table
from .utils import get_device, save_json, set_seed

FROZEN = "Frozen baseline"
DEFAULT = "Fine-tuned (default)"
TUNED = "Fine-tuned (tuned)"


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_state(cfg):
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    save_json(cfg.to_dict(), cfg.results_dir / "config.json")
    return {"cfg": cfg, "device": device, "probs": {}, "info": {}}


# 1 ------------------------------------------------------------------ data
def step_data(state):
    cfg = state["cfg"]
    train, val, test = load_civil_comments(cfg)
    stats = describe_splits({"train": train, "validation": val, "test": test})
    stats.to_csv(cfg.results_dir / "data_stats.csv", index=False)
    plots.plot_data_overview(train, val, test, cfg.figures_dir / "01_data_overview.png")
    state.update(train=train, val=val, test=test, data_stats=stats)
    print(stats.to_string(index=False))
    return stats


# 2 -------------------------------------------------------- transfer learning
def step_frozen_baseline(state):
    cfg, device = state["cfg"], state["device"]
    clf, _, probs, emb, info = train_frozen_baseline(
        state["train"], state["val"], state["test"], cfg, device)
    state["probs"][FROZEN] = probs
    state["info"][FROZEN] = info
    state["frozen_clf"] = clf
    plots.plot_embedding_tsne(emb["test"], state["test"]["label"].values,
                              cfg.figures_dir / "02_frozen_embeddings_tsne.png", seed=cfg.seed)
    m = compute_metrics(state["val"]["label"], probs["val"])
    print(f"Frozen baseline  val F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
    print(f"Trainable encoder parameters: {info['trainable_encoder_params']:,} "
          f"of {info['encoder_params']:,}; classifier parameters: {info['classifier_params']:,}")
    return m


# 3 ------------------------------------------------------------- fine-tuning
def step_finetune_default(state):
    cfg, device = state["cfg"], state["device"]
    model, tok, history, info = fine_tune(state["train"], state["val"], cfg, device)
    state["probs"][DEFAULT] = {
        "val": predict_proba(model, tok, state["val"]["text"], device, cfg.max_length),
        "test": predict_proba(model, tok, state["test"]["text"], device, cfg.max_length),
    }
    state["info"][DEFAULT] = info
    state["history_default"] = history
    plots.plot_training_history(history, "Fine-tuning with default hyperparameters",
                                cfg.figures_dir / "03_training_default.png")
    save_json([{k: v for k, v in h.items() if k != "step_losses"} for h in history],
              cfg.results_dir / "history_default.json")
    del model
    _free()
    return history


# 4 ---------------------------------------------------- hyperparameter tuning
def step_tuning(state):
    cfg, device = state["cfg"], state["device"]
    study = run_search(state["train"], state["val"], cfg, device)
    table = trials_table(study)
    imps = param_importances(study)
    table.to_csv(cfg.results_dir / "tuning_trials.csv", index=False)
    save_json({"best_params": study.best_params, "best_val_pr_auc": study.best_value,
               "importances": imps}, cfg.results_dir / "tuning_best.json")

    plots.plot_tuning_results(table, imps, cfg.figures_dir / "04_tuning_results.png")
    shown = table.drop(columns=["time_sec"]).copy()
    shown["learning_rate"] = shown["learning_rate"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
    plots.render_table_image(shown, cfg.figures_dir / "04_tuning_table.png",
                             title=f"Optuna trials sorted by validation PR-AUC "
                                   f"({cfg.n_trials} trials, best row highlighted)")
    state.update(study=study, tuning_table=table, importances=imps,
                 best_params=study.best_params)
    print("\nBest parameters:", study.best_params)
    print(f"Best validation PR-AUC (tuning subset): {study.best_value:.4f}")
    return table


# 5 ----------------------------------------- retrain with the best parameters
def step_finetune_tuned(state):
    cfg, device = state["cfg"], state["device"]
    model, tok, history, info = fine_tune(state["train"], state["val"], cfg, device,
                                          **state["best_params"])
    state["probs"][TUNED] = {
        "val": predict_proba(model, tok, state["val"]["text"], device, cfg.max_length),
        "test": predict_proba(model, tok, state["test"]["text"], device, cfg.max_length),
    }
    state["info"][TUNED] = info
    state["history_tuned"] = history
    state["model"], state["tokenizer"] = model, tok
    plots.plot_training_history(history, "Fine-tuning with tuned hyperparameters",
                                cfg.figures_dir / "05_training_tuned.png")
    save_json([{k: v for k, v in h.items() if k != "step_losses"} for h in history],
              cfg.results_dir / "history_tuned.json")
    out = cfg.models_dir / "finetuned_tuned"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return history


# 6 -------------------------------------------------------------- evaluation
def step_evaluate(state):
    cfg = state["cfg"]
    y_val, y_test = state["val"]["label"].values, state["test"]["label"].values
    rows_default, rows_tuned_thr, thresholds = {}, {}, {}

    for name, p in state["probs"].items():
        thr = best_f1_threshold(y_val, p["val"])
        thresholds[name] = thr
        rows_default[name] = compute_metrics(y_test, p["test"], 0.5)
        rows_tuned_thr[name] = compute_metrics(y_test, p["test"], thr)

    m05 = pd.DataFrame(rows_default).T
    mthr = pd.DataFrame(rows_tuned_thr).T
    m05.to_csv(cfg.results_dir / "test_metrics_threshold_0.5.csv")
    mthr.to_csv(cfg.results_dir / "test_metrics_val_threshold.csv")

    test_probs = {k: v["test"] for k, v in state["probs"].items()}
    plots.plot_metric_comparison(m05, cfg.figures_dir / "06_metric_comparison.png")
    plots.plot_roc_pr(y_test, test_probs, cfg.figures_dir / "07_roc_pr_curves.png")
    plots.plot_confusion_matrices(
        y_test, {k: (v >= 0.5).astype(int) for k, v in test_probs.items()},
        cfg.figures_dir / "08_confusion_matrices.png")
    plots.plot_probability_distributions(y_test, test_probs,
                                         cfg.figures_dir / "09_probability_distributions.png")

    state.update(metrics_05=m05, metrics_thr=mthr, thresholds=thresholds)
    save_json({"thresholds": thresholds, "info": state["info"]},
              cfg.results_dir / "model_info.json")
    cols = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]
    print("Test metrics at threshold 0.5")
    print(m05[cols].round(4).to_string())
    print("\nTest metrics at the F1-optimal threshold chosen on validation")
    print(mthr[cols + ["threshold"]].round(4).to_string())
    return m05, mthr


# 7 ------------------------------------------------------- real-world testing
def step_real_world(state):
    cfg, device = state["cfg"], state["device"]
    df = load_real_world_texts()
    thr = state["thresholds"][TUNED]
    p = predict_proba(state["model"], state["tokenizer"], df["text"], device, cfg.max_length)
    out = df.copy()
    out["prob_toxic"] = p
    out["threshold"] = thr
    out["predicted_label"] = (p >= thr).astype(int)
    out["predicted_class"] = out["predicted_label"].map({0: "not toxic", 1: "toxic"})
    out["matches_expected"] = out["predicted_label"] == out["expected_label"]
    out.to_csv(cfg.results_dir / "real_world_predictions.csv", index=False)
    plots.plot_real_world(out, cfg.figures_dir / "10_real_world_predictions.png")
    state["real_world"] = out
    acc = out["matches_expected"].mean()
    print(f"Real-world texts: {out['matches_expected'].sum()}/{len(out)} match the expected label ({acc:.0%})")
    return out[["text", "prob_toxic", "predicted_class", "expected_label", "matches_expected"]]


# 8 ---------------------------------------------------------- interpretability
def step_explain(state, per_group=2, n_real_world=4):
    cfg, device = state["cfg"], state["device"]
    thr = state["thresholds"][TUNED]
    examples = select_examples(state["test"], state["probs"][TUNED]["test"], thr,
                               per_group=per_group, seed=cfg.seed)

    rw = state["real_world"]
    # real-world texts the model got wrong are the most interesting ones to explain
    wrong = rw[~rw["matches_expected"]]
    right_toxic = rw[rw["matches_expected"] & (rw["expected_label"] == 1)]
    right_clean = rw[rw["matches_expected"] & (rw["expected_label"] == 0)]
    half = n_real_world // 2
    rw_pick = pd.concat([wrong, right_toxic.head(half), right_clean.tail(half)]).head(n_real_world)
    rw_rows = pd.DataFrame({
        "group": "real_world",
        "text": rw_pick["text"].values,
        "toxicity": np.nan,
        "label": rw_pick["expected_label"].values,
        "prob_toxic": rw_pick["prob_toxic"].values,
        "pred": rw_pick["predicted_label"].values,
    })
    examples = pd.concat([examples, rw_rows]).reset_index(drop=True)

    explainer = build_explainer(state["model"], state["tokenizer"], device, cfg.max_length)
    sv = explain_texts(explainer, examples["text"], max_evals=cfg.shap_max_evals)
    save_text_plot_html(sv, cfg.figures_dir / "shap_text_plots.html")

    summaries = []
    exp_dir = cfg.figures_dir / "explanations"
    exp_dir.mkdir(exist_ok=True)
    for i, row in examples.iterrows():
        tokens, values = token_contributions(sv, i)
        pred = "toxic" if row["pred"] == 1 else "not toxic"
        actual = "toxic" if row["label"] == 1 else "not toxic"
        title = (f"[{row['group']}] predicted: {pred} (P={row['prob_toxic']:.2f}), "
                 f"actual: {actual}")
        for fn, suffix in ((plots.plot_token_heatmap, "heatmap"), (plots.plot_top_token_bars, "bars")):
            plt.close(fn(tokens, values, title, exp_dir / f"example_{i:02d}_{suffix}.png"))
        pos, neg = top_tokens(sv, i, k=4)
        summaries.append({
            "id": i, "group": row["group"], "text": row["text"],
            "actual": actual, "predicted": pred, "prob_toxic": float(row["prob_toxic"]),
            "toward_toxic": ", ".join(f"{t} ({v:+.3f})" for t, v in zip(pos["token"], pos["shap"])),
            "toward_not_toxic": ", ".join(f"{t} ({v:+.3f})" for t, v in zip(neg["token"], neg["shap"])),
            "base_value": float(np.asarray(sv[i].base_values)[1]),
        })
    plots.plot_global_token_importance(sv, cfg.figures_dir / "11_shap_global_tokens.png")

    summary = pd.DataFrame(summaries)
    summary.to_csv(cfg.results_dir / "explanations.csv", index=False)
    state.update(explanations=summary, shap_values=sv)
    return summary
