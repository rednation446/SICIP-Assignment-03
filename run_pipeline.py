import argparse

import matplotlib

matplotlib.use("Agg")

from src.config import get_config
from src import pipeline
from src.report import update_readme


def main():
    parser = argparse.ArgumentParser(description="Toxic comment classification pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="small run that finishes on a CPU")
    parser.add_argument("--trials", type=int, default=None,
                        help="number of hyperparameter search trials")
    parser.add_argument("--model", type=str, default=None,
                        help="Hugging Face model name, e.g. bert-base-uncased")
    args = parser.parse_args()

    cfg = get_config("quick" if args.quick else "full")
    if args.trials:
        cfg.n_trials = args.trials
    if args.model:
        cfg.model_name = args.model

    state = pipeline.init_state(cfg)

    print("\n=== 1. Data ===")
    pipeline.step_data(state)
    print("\n=== 2. Transfer learning: frozen encoder baseline ===")
    pipeline.step_frozen_baseline(state)
    print("\n=== 3. Fine-tuning with default hyperparameters ===")
    pipeline.step_finetune_default(state)
    print("\n=== 4. Hyperparameter tuning ===")
    pipeline.step_tuning(state)
    print("\n=== 5. Fine-tuning with the best hyperparameters ===")
    pipeline.step_finetune_tuned(state)
    print("\n=== 6. Evaluation on the test set ===")
    pipeline.step_evaluate(state)
    print("\n=== 7. Real-world text ===")
    print(pipeline.step_real_world(state).to_string(index=False))
    print("\n=== 8. Interpretability (SHAP) ===")
    print(pipeline.step_explain(state)[["group", "predicted", "actual", "toward_toxic"]].to_string(index=False))

    update_readme(state)
    print("\nDone. Figures are in figures/, tables in results/.")


if __name__ == "__main__":
    main()
