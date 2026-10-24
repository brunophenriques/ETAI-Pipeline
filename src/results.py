"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so youcan open it again later, or compare two runs side by side after changing something in config.yaml.
"""
import os
from datetime import datetime
from pathlib import Path
import pandas as pd


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(results_dir, f"{config['model']['type']}_{timestamp}.txt")

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model']['params']}\n"
        f"Reserved test size: {config['split']['test_size']}  "
        f"random_state: {config['split']['random_state']}\n"
        f"CV settings: {config.get('cv', {})}\n"
        f"Preprocessing settings: {config.get('preprocessing', {})}\n"
        f"Tuning settings: {config.get('tuning', {})}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    import json
    Path(path).with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return path


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Append a run, retaining earlier runs and an extensible CSV schema."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(destination, dtype={"run": str}) if destination.exists() else pd.DataFrame()
    runs = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    runs.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return runs
