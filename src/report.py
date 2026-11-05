"""Writes the numbers produced by a run into the marked sections of README.md,
so the report always matches the latest results."""

import re

import pandas as pd

from .config import PROJECT_ROOT
from .pipeline import DEFAULT, FROZEN, TUNED

README = PROJECT_ROOT / "README.md"


def _md_table(df, floatfmt=4):
    df = df.copy()
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append(f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _replace(text, key, body):
    pattern = re.compile(rf"(<!-- AUTO:{key}:START -->)(.*?)(<!-- AUTO:{key}:END -->)", re.S)
    return pattern.sub(lambda m: f"{m.group(1)}\n{body.strip()}\n{m.group(3)}", text)


def _setup_section(state):
    cfg = state["cfg"]
    stats = state["data_stats"].rename(columns={"toxic_%": "toxic (%)"})
    dev = state["device"].type
    return (
        f"Run settings: model `{cfg.model_name}`, max length {cfg.max_length} tokens, "
        f"toxicity threshold **{cfg.toxicity_threshold}**, seed {cfg.seed}, device `{dev}`.\n\n"
        + _md_table(stats, 2)
        + "\n\nThe training subset was sampled with "
        f"{int(cfg.train_toxic_ratio * 100)}% toxic comments. Validation and test keep the "
        "natural class balance of Civil Comments, so the test numbers reflect the real distribution.\n\n"
        "![Data overview](figures/01_data_overview.png)"
    )


def _results_section(state):
    cols = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc"]
    m05 = state["metrics_05"][cols].reset_index().rename(columns={"index": "model"})
    mthr = state["metrics_thr"][cols + ["threshold"]].reset_index().rename(columns={"index": "model"})
    info = state["info"]

    a, b, c = (state["metrics_05"].loc[n] for n in (FROZEN, DEFAULT, TUNED))
    params = pd.DataFrame([
        {"model": FROZEN, "trainable parameters": f"{info[FROZEN]['classifier_params']:,} (logistic regression only)",
         "training time (s)": round(info[FROZEN]["train_time_sec"], 1)},
        {"model": DEFAULT, "trainable parameters": f"{info[DEFAULT]['trainable_params']:,}",
         "training time (s)": round(info[DEFAULT]["train_time_sec"], 1)},
        {"model": TUNED, "trainable parameters": f"{info[TUNED]['trainable_params']:,}",
         "training time (s)": round(info[TUNED]["train_time_sec"], 1)},
    ])

    def d(x, y, k):
        return f"{y[k] - x[k]:+.3f}"

    text = (
        "**Test set, decision threshold 0.5**\n\n" + _md_table(m05) +
        "\n\n**Test set, F1-optimal threshold picked on the validation set**\n\n" + _md_table(mthr) +
        "\n\n" + _md_table(params, 1) +
        "\n\nMeasured differences on the test set (threshold 0.5):\n\n"
        f"- Fine-tuning vs frozen baseline: F1 {d(a, b, 'f1')}, ROC-AUC {d(a, b, 'roc_auc')}, "
        f"PR-AUC {d(a, b, 'pr_auc')}.\n"
        f"- Tuned vs default fine-tuning: F1 {d(b, c, 'f1')}, ROC-AUC {d(b, c, 'roc_auc')}, "
        f"PR-AUC {d(b, c, 'pr_auc')}.\n"
        f"- Tuned model vs frozen baseline: F1 {d(a, c, 'f1')}, PR-AUC {d(a, c, 'pr_auc')}.\n\n"
        "![Metric comparison](figures/06_metric_comparison.png)\n\n"
        "![ROC and PR curves](figures/07_roc_pr_curves.png)\n\n"
        "![Confusion matrices](figures/08_confusion_matrices.png)\n\n"
        "![Probability distributions](figures/09_probability_distributions.png)\n\n"
        "Training curves:\n\n"
        "![Default fine-tuning](figures/03_training_default.png)\n\n"
        "![Tuned fine-tuning](figures/05_training_tuned.png)\n\n"
        "Frozen embeddings (t-SNE). The two classes overlap heavily, which is why a linear "
        "classifier on frozen features has limited room:\n\n"
        "![t-SNE](figures/02_frozen_embeddings_tsne.png)"
    )
    return text


def _tuning_section(state):
    cfg = state["cfg"]
    best = state["best_params"]
    t = state["tuning_table"]
    n_done = int((t["state"] == "COMPLETE").sum())
    n_pruned = int((t["state"] == "PRUNED").sum())
    default_row = t[t["trial"] == 0]
    default_score = float(default_row["val_pr_auc"].iloc[0]) if len(default_row) else float("nan")
    imps = state["importances"]
    imp_txt = ", ".join(f"{k} ({v:.2f})" for k, v in imps.items()) if imps else "not available"
    def rng(k, v):
        if isinstance(v, tuple):
            scale = " (log scale)" if k == "learning_rate" else ""
            return f"{v[0]:g} to {v[1]:g}{scale}"
        return "one of " + ", ".join(str(x) for x in v)

    def fmt(k, v):
        if k == "learning_rate":
            return f"{v:.2e}"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    space = "\n".join(f"| {k} | {rng(k, v)} |" for k, v in cfg.search_space.items())
    best_rows = "\n".join(f"| {k} | {fmt(k, v)} |" for k, v in best.items())
    return (
        "| hyperparameter | search range |\n|---|---|\n" + space +
        f"\n\nMethod: Optuna with the TPE sampler, {cfg.n_trials} trials, median pruning after "
        f"each epoch, objective = validation PR-AUC. Each trial fine-tunes on a "
        f"{cfg.tune_train_size}-comment subset of the training data to keep the search affordable. "
        "Trial 0 is the default setting, so the search is measured against it.\n\n"
        f"Completed trials: {n_done}, pruned: {n_pruned}. Default setting (trial 0) scored "
        f"{default_score:.4f}; the best trial scored {t['val_pr_auc'].max():.4f}.\n\n"
        "| best hyperparameter | value |\n|---|---|\n" + best_rows +
        f"\n\nParameter importance (fANOVA): {imp_txt}.\n\n"
        "Screenshot of the tuning results:\n\n"
        "![Tuning table](figures/04_tuning_table.png)\n\n"
        "![Tuning plots](figures/04_tuning_results.png)"
    )


def _real_world_section(state):
    rw = state["real_world"].copy()
    rw["expected"] = rw["expected_label"].map({0: "not toxic", 1: "toxic"})
    rw["check"] = rw.apply(lambda r: ("correct: " if r["matches_expected"] else "wrong: ") + r["note"], axis=1)
    tbl = rw[["text", "source", "prob_toxic", "predicted_class", "expected", "check"]]
    n_ok = int(rw["matches_expected"].sum())
    return (
        f"Model: tuned fine-tuned model, decision threshold {rw['threshold'].iloc[0]:.3f} "
        "(chosen on validation). The texts are in `data/real_world_texts.csv`; the expected "
        "label is my own judgement of each sentence.\n\n"
        + _md_table(tbl, 3) +
        f"\n\n{n_ok} of {len(rw)} predictions agree with the expected label.\n\n"
        "![Real-world predictions](figures/10_real_world_predictions.png)"
    )


def _explain_section(state):
    ex = state["explanations"]
    parts = []
    for _, r in ex.iterrows():
        i = int(r["id"])
        parts.append(
            f"**Example {i} ({r['group'].replace('_', ' ')})**: \"{r['text']}\"\n\n"
            f"Predicted {r['predicted']} (P(toxic) = {r['prob_toxic']:.3f}), actual / expected {r['actual']}.\n"
            f"Tokens pushing towards toxic: {r['toward_toxic'] or 'none'}. "
            f"Tokens pushing towards not toxic: {r['toward_not_toxic'] or 'none'}.\n\n"
            f"![example {i}](figures/explanations/example_{i:02d}_heatmap.png)\n"
        )
    return (
        "\n".join(parts) +
        "\nAverage contribution of tokens across all explained examples:\n\n"
        "![Global tokens](figures/11_shap_global_tokens.png)\n\n"
        "The interactive SHAP text plots for all examples are in `figures/shap_text_plots.html` "
        "(open it in a browser)."
    )


def update_readme(state):
    text = README.read_text(encoding="utf-8")
    text = _replace(text, "SETUP", _setup_section(state))
    text = _replace(text, "RESULTS", _results_section(state))
    text = _replace(text, "TUNING", _tuning_section(state))
    text = _replace(text, "REALWORLD", _real_world_section(state))
    text = _replace(text, "EXPLAIN", _explain_section(state))
    README.write_text(text, encoding="utf-8")
    print(f"README updated: {README}")
