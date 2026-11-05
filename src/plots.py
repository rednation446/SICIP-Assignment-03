import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

# fixed colour per model so it means the same thing in every figure
MODEL_COLORS = {
    "Frozen baseline": "#2a78d6",
    "Fine-tuned (default)": "#eb6834",
    "Fine-tuned (tuned)": "#1baf7a",
}
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_NAMES = {0: "not toxic", 1: "toxic"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_div", ["#2a78d6", "#f4f3f0", "#e34948"])
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f7fc", "#a8c8ef", "#2a78d6", "#0f3d75"])

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "font.size": 10,
})


def _save(fig, path):
    if path is not None:
        fig.savefig(path)
    return fig


# ---------------------------------------------------------------- data

def plot_data_overview(train, val, test, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    names = ["train", "validation", "test"]
    dfs = [train, val, test]
    x = np.arange(len(names))
    pos = [d["label"].mean() * 100 for d in dfs]
    neg = [100 - p for p in pos]
    ax.bar(x - 0.18, neg, 0.34, color=CLASS_COLORS[0], label="not toxic")
    ax.bar(x + 0.18, pos, 0.34, color=CLASS_COLORS[1], label="toxic")
    for i, p in enumerate(pos):
        ax.text(x[i] + 0.18, p + 1.5, f"{p:.1f}%", ha="center", color=TEXT, fontsize=9)
    ax.set_xticks(x, [f"{n}\n(n={len(d)})" for n, d in zip(names, dfs)])
    ax.set_ylabel("% of comments")
    ax.set_title("Class balance per split")
    ax.legend()
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    full = pd.concat([val, test])
    ax.hist(full["toxicity"], bins=40, color="#2a78d6", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.axvline(0.5, color=TEXT, ls="--", lw=1.2)
    ax.text(0.52, ax.get_ylim()[1] * 0.4, "threshold = 0.5", color=TEXT, fontsize=9)
    ax.set_xlabel("toxicity score (fraction of annotators)")
    ax.set_ylabel("comments (log scale)")
    ax.set_title("Toxicity score distribution (val + test)")

    ax = axes[2]
    lengths = train["text"].str.split().str.len()
    for lab in (0, 1):
        ax.hist(lengths[train["label"] == lab].clip(upper=300), bins=40, alpha=0.6,
                color=CLASS_COLORS[lab], label=CLASS_NAMES[lab], density=True)
    ax.set_xlabel("words per comment (clipped at 300)")
    ax.set_ylabel("density")
    ax.set_title("Comment length (train)")
    ax.legend()

    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- baseline

def plot_embedding_tsne(X, y, path=None, n=1500, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Z = TSNE(n_components=2, random_state=seed, init="pca",
             perplexity=30).fit_transform(X[idx])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for lab in (0, 1):
        m = y[idx] == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, color=CLASS_COLORS[lab],
                   label=CLASS_NAMES[lab], edgecolors="none")
    ax.set_title("t-SNE of frozen encoder embeddings (test)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2)
    return _save(fig, path)


# ---------------------------------------------------------------- training

def plot_training_history(history, title, path=None):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    steps = np.concatenate([h["step_losses"] for h in history])
    win = max(1, len(steps) // 50)
    smooth = pd.Series(steps).rolling(win, min_periods=1).mean()
    axes[0].plot(steps, color=GRID, lw=1)
    axes[0].plot(smooth, color="#eb6834", lw=2, label=f"moving avg ({win} steps)")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("loss")
    axes[0].set_title("Training loss per step")
    axes[0].legend()

    axes[1].plot(ep, [h["train_loss"] for h in history], "o-", color="#eb6834", lw=2, ms=8, label="train")
    axes[1].plot(ep, [h["val_loss"] for h in history], "o-", color="#2a78d6", lw=2, ms=8, label="validation")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("cross-entropy loss")
    axes[1].set_title("Loss per epoch"); axes[1].set_xticks(ep)
    axes[1].legend()

    for key, col, lab in [("val_f1", "#2a78d6", "F1"), ("val_roc_auc", "#eb6834", "ROC-AUC"),
                          ("val_pr_auc", "#1baf7a", "PR-AUC")]:
        axes[2].plot(ep, [h[key] for h in history], "o-", color=col, lw=2, ms=8, label=lab)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("score")
    axes[2].set_title("Validation metrics per epoch"); axes[2].set_xticks(ep)
    axes[2].legend()

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- evaluation

def plot_confusion_matrices(y_true, preds: dict, path=None):
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1%})", ha="center", va="center",
                        color="white" if pct[i, j] > 0.55 else TEXT, fontsize=11)
        ax.set_xticks([0, 1], ["not toxic", "toxic"])
        ax.set_yticks([0, 1], ["not toxic", "toxic"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        ax.set_title(name)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test set (row-normalised)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_roc_pr(y_true, probs: dict, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, p in probs.items():
        c = MODEL_COLORS.get(name)
        fpr, tpr, _ = roc_curve(y_true, p)
        prec, rec, _ = precision_recall_curve(y_true, p)
        axes[0].plot(fpr, tpr, color=c, lw=2, label=name)
        axes[1].plot(rec, prec, color=c, lw=2, label=name)
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC curve (test)")
    base = np.mean(y_true)
    axes[1].axhline(base, color=MUTED, ls="--", lw=1)
    axes[1].text(0.02, base + 0.02, f"chance = {base:.2f}", color=MUTED, fontsize=9)
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall curve (test)")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_comparison(metrics_df, path=None,
                           cols=("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    models = list(metrics_df.index)
    x = np.arange(len(cols))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [metrics_df.loc[m, c] for c in cols]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w * 0.92,
                      color=MODEL_COLORS.get(m), label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(x, [c.replace("_", "-").upper() if "auc" in c else c.capitalize() for c in cols])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Test metrics: frozen baseline vs fine-tuned vs tuned")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(models))
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_probability_distributions(y_true, probs: dict, path=None):
    fig, axes = plt.subplots(1, len(probs), figsize=(5 * len(probs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 26)
    for ax, (name, p) in zip(axes, probs.items()):
        for lab in (0, 1):
            ax.hist(p[np.asarray(y_true) == lab], bins=bins, alpha=0.65,
                    color=CLASS_COLORS[lab], label=f"actual {CLASS_NAMES[lab]}", density=True)
        ax.set_title(name); ax.set_xlabel("predicted P(toxic)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.suptitle("How well each model separates the two classes (test)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- tuning

def plot_tuning_results(trials_df, importances, path=None):
    done = trials_df[trials_df["state"] == "COMPLETE"].sort_values("trial")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(done["trial"], done["val_pr_auc"], "o", color="#2a78d6", ms=8, label="trial")
    ax.plot(done["trial"], done["val_pr_auc"].cummax(), "-", color="#eb6834", lw=2, label="best so far")
    pruned = trials_df[trials_df["state"] == "PRUNED"]
    if len(pruned):
        ax.plot(pruned["trial"], [done["val_pr_auc"].min()] * len(pruned), "x",
                color=MUTED, ms=8, label="pruned")
    ax.set_xlabel("trial"); ax.set_ylabel("validation PR-AUC")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Optimisation history"); ax.legend()

    ax = axes[1]
    sc = ax.scatter(done["learning_rate"], done["val_pr_auc"], c=done["epochs"],
                    cmap=SEQ_CMAP, s=70, edgecolors=TEXT, linewidths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log)"); ax.set_ylabel("validation PR-AUC")
    ax.set_title("Learning rate vs score")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("epochs")

    ax = axes[2]
    if importances:
        names = list(importances.keys())[::-1]
        vals = [importances[n] for n in names]
        ax.barh(names, vals, color="#2a78d6", height=0.6)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=TEXT)
        ax.set_xlim(0, max(vals) * 1.25 + 1e-6)
    ax.set_title("Hyperparameter importance (fANOVA)")
    ax.set_xlabel("importance"); ax.grid(axis="y", visible=False)

    fig.tight_layout()
    return _save(fig, path)


def render_table_image(df, path=None, title=None, highlight_first=True, float_fmt="{:.4g}"):
    """Renders a DataFrame as an image (used for the tuning results screenshot)."""
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    show = show.fillna("")
    n_rows, n_cols = show.shape
    fig, ax = plt.subplots(figsize=(max(8, 1.35 * n_cols), 0.42 * (n_rows + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=list(show.columns),
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#0f3d75"); cell.set_text_props(color="white", weight="bold")
        elif highlight_first and r == 1:
            cell.set_facecolor("#d6f2e7")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f6f3")
    if title:
        ax.set_title(title, fontweight="bold", pad=12)
    return _save(fig, path)


# ---------------------------------------------------------------- interpretability

def plot_token_heatmap(tokens, values, title, path=None, max_per_line=14):
    """Draws the comment with every token coloured by its SHAP value
    (red pushes towards toxic, blue towards not toxic)."""
    tokens = [str(t).strip() for t in tokens]
    vmax = max(np.abs(values).max(), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    lines = [list(range(i, min(i + max_per_line, len(tokens))))
             for i in range(0, len(tokens), max_per_line)]
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(lines) + 1.1))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, len(lines))
    renderer = fig.canvas.get_renderer()
    for li, idxs in enumerate(lines):
        x = 0.0
        y = len(lines) - li - 0.5
        for i in idxs:
            t = ax.text(x, y, tokens[i], fontsize=11, va="center", ha="left", color=TEXT,
                        bbox=dict(boxstyle="round,pad=0.25", fc=SHAP_CMAP(norm(values[i])),
                                  ec="none"))
            bb = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
            x = bb.x1 + 0.008
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SHAP_CMAP)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.08, pad=0.05, aspect=50)
    cb.set_label("SHAP value  (blue = towards not toxic, red = towards toxic)")
    ax.set_title(title, fontsize=10, loc="left")
    return _save(fig, path)


def plot_top_token_bars(tokens, values, title, path=None, k=10):
    df = pd.DataFrame({"token": [str(t).strip() for t in tokens], "v": values})
    df = df.reindex(df["v"].abs().sort_values(ascending=False).index).head(k)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(df) + 1.2))
    colors = ["#e34948" if v > 0 else "#2a78d6" for v in df["v"]]
    ax.barh(range(len(df)), df["v"], color=colors, height=0.65)
    ax.set_yticks(range(len(df)), df["token"])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("SHAP value for the toxic class")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_global_token_importance(shap_values, path=None, k=15, class_idx=1):
    """Mean SHAP value per token across all explained examples."""
    rows = []
    for i in range(len(shap_values)):
        toks = shap_values[i].data
        vals = np.asarray(shap_values[i].values)[:, class_idx]
        for t, v in zip(toks, vals):
            t = str(t).strip().lower()
            if t:
                rows.append((t, v))
    df = pd.DataFrame(rows, columns=["token", "v"]).groupby("token")["v"].mean()
    top_pos = df.sort_values(ascending=False).head(k)
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(top_pos) + 1.2))
    ax.barh(range(len(top_pos))[::-1], top_pos.values, color="#e34948", height=0.65)
    ax.set_yticks(range(len(top_pos))[::-1], top_pos.index)
    ax.set_xlabel("mean SHAP value for the toxic class")
    ax.set_title("Tokens that push predictions towards toxic\n(across the explained examples)", fontsize=10)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_real_world(results_df, path=None):
    df = results_df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df) + 1.3))
    y = np.arange(len(df))[::-1]
    colors = [CLASS_COLORS[int(p)] for p in df["predicted_label"]]
    ax.barh(y, df["prob_toxic"], color=colors, height=0.6)
    ax.axvline(df["threshold"].iloc[0], color=TEXT, ls="--", lw=1)
    labels = [(t[:60] + "...") if len(t) > 60 else t for t in df["text"]]
    ax.set_yticks(y, labels, fontsize=8.5)
    for yi, (p, ok) in zip(y, zip(df["prob_toxic"], df["matches_expected"])):
        ax.text(min(p + 0.01, 0.93), yi, "match" if ok else "mismatch",
                va="center", fontsize=8, color=TEXT if ok else "#b5302f")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("predicted P(toxic)")
    ax.set_title("Real-world text: predicted toxicity (dashed line = decision threshold)")
    ax.legend(handles=[Patch(color=CLASS_COLORS[0], label="predicted not toxic"),
                       Patch(color=CLASS_COLORS[1], label="predicted toxic")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
