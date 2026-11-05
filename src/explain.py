"""Explaining individual predictions with SHAP.

SHAP's text masker hides groups of tokens and measures how the toxic
probability changes, so every token gets an additive contribution:
positive values push the prediction towards "toxic", negative values push it
towards "not toxic"."""

import numpy as np
import pandas as pd
import shap

from .finetune import predict_proba


def build_explainer(model, tokenizer, device, max_length=128):
    def f(texts):
        texts = [str(t) for t in texts]
        p = predict_proba(model, tokenizer, texts, device, max_length, batch_size=64)
        return np.stack([1 - p, p], axis=1)

    masker = shap.maskers.Text(tokenizer)
    return shap.Explainer(f, masker, output_names=["not_toxic", "toxic"])


def explain_texts(explainer, texts, max_evals=300):
    return explainer(list(texts), max_evals=max_evals, batch_size=32)


def token_contributions(shap_values, i, class_idx=1):
    """Tokens and their SHAP values (towards `class_idx`) for example i,
    with empty special tokens removed."""
    tokens = np.asarray(shap_values[i].data)
    values = np.asarray(shap_values[i].values)[:, class_idx]
    keep = np.array([len(str(t).strip()) > 0 for t in tokens])
    return list(tokens[keep]), values[keep]


def top_tokens(shap_values, i, k=5, class_idx=1):
    tokens, values = token_contributions(shap_values, i, class_idx)
    df = pd.DataFrame({"token": [t.strip() for t in tokens], "shap": values})
    pos = df.sort_values("shap", ascending=False).head(k)
    neg = df.sort_values("shap", ascending=True).head(k)
    return pos[pos["shap"] > 0], neg[neg["shap"] < 0]


def select_examples(test_df, probs, threshold, per_group=2, seed=42):
    """Picks confident correct predictions and the clearest mistakes so the
    explanations cover both what the model gets right and where it fails."""
    df = test_df.copy()
    df["prob_toxic"] = probs
    df["pred"] = (df["prob_toxic"] >= threshold).astype(int)
    df["words"] = df["text"].str.split().str.len()
    df = df[(df["words"] >= 5) & (df["words"] <= 45)]

    groups = {
        "true_positive": df[(df.label == 1) & (df.pred == 1)].sort_values("prob_toxic", ascending=False),
        "true_negative": df[(df.label == 0) & (df.pred == 0)].sort_values("prob_toxic"),
        "false_positive": df[(df.label == 0) & (df.pred == 1)].sort_values("prob_toxic", ascending=False),
        "false_negative": df[(df.label == 1) & (df.pred == 0)].sort_values("prob_toxic"),
    }
    picked = []
    for name, g in groups.items():
        g = g.head(per_group * 5)
        if len(g) == 0:
            continue
        g = g.sample(n=min(per_group, len(g)), random_state=seed).copy()
        g["group"] = name
        picked.append(g)
    out = pd.concat(picked).reset_index(drop=True)
    return out[["group", "text", "toxicity", "label", "prob_toxic", "pred"]]


def save_text_plot_html(shap_values, path, class_idx=1):
    html = shap.plots.text(shap_values[:, :, class_idx], display=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>")
        f.write(html)
        f.write("</body></html>")
