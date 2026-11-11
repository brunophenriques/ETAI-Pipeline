"""Run the existing models sequentially using one Adult YAML, without editing it."""
import argparse
from copy import deepcopy
from pathlib import Path

import main
from scripts.run_week7 import MODELS as WEEK7_MODELS


MODELS = {**WEEK7_MODELS, "xgboost": {
    "random_state": 42, "n_jobs": 1, "tree_method": "hist", "eval_metric": "logloss",
}}


def run(config_path="config_adult.yaml", models=None, *, untuned=False, trials=None):
    """Run complete model blocks sequentially; return the saved scoreboard rows.

    Configuration overrides are copied in memory. The source YAML, shared
    preprocessing and folds stay unchanged; predictions are saved per model.
    """
    base = main.load_config(config_path)
    names = list(MODELS) if models is None else list(models)
    unknown = set(names) - MODELS.keys()
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    if trials is not None and trials < 1:
        raise ValueError("--trials must be positive.")
    # Validate every recipe before starting the first potentially long run.
    recipes = []
    for name in names:
        config = deepcopy(base)
        params = base['model']['params'] if base['model']['type'] == name else MODELS[name]
        config['model'] = {'type': name, 'params': deepcopy(params)}
        config['tuning']['enabled'] = base['tuning']['enabled'] and not untuned and name != 'dummy'
        if trials is not None:
            config['tuning']['n_trials'] = trials
        if config['tuning']['enabled'] and not config['tuning']['search_spaces'].get(name):
            raise ValueError(f"No tuning search space configured for {name}.")
        config['output']['layout'] = 'model'
        config['output']['predictions'] = str(
            Path(base['output']['results_dir']) / name / 'predictions.csv')
        recipes.append(config)
    rows = []
    for i, config in enumerate(recipes, 1):
        name = config['model']['type']
        print(f"\n[{i}/{len(recipes)}] Running {name}", flush=True)
        row = main.main(config_path, config=config)
        rows.append(row)
        print(f"Completed {name}: {row['metric']} = {row['mean']:.4f}", flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_adult.yaml')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--untuned', action='store_true', help='Disable tuning for a quick comparison.')
    parser.add_argument('--trials', type=int, help='Override trials per search for this command only.')
    args = parser.parse_args()
    run(args.config, args.models, untuned=args.untuned, trials=args.trials)
