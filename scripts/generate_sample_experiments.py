#!/usr/bin/env python3
"""
Example script showing how to save experiment results in the format expected
by the automated reporting system.

This demonstrates the integration between experimental code and the reporting system.
"""
import json
import random
from pathlib import Path
from typing import Dict


def run_mock_experiment(
    experiment_name: str, run_id: str, config: Dict
) -> Dict[str, float]:
    """
    Mock experiment that simulates training a model and computing metrics.

    Args:
        experiment_name: Name of the experiment (e.g., 'baseline', 'improved_model')
        run_id: Unique identifier for this run (e.g., 'run_001', 'seed_42')
        config: Configuration parameters for the experiment

    Returns:
        Dictionary of metrics
    """
    # Simulate some variability in results
    random.seed(hash(run_id))
    base_accuracy = config.get("base_accuracy", 0.8)
    noise = random.uniform(-0.05, 0.05)

    accuracy = base_accuracy + noise
    f1 = accuracy - 0.03 + random.uniform(-0.02, 0.02)
    precision = accuracy + 0.01 + random.uniform(-0.02, 0.02)
    recall = accuracy - 0.02 + random.uniform(-0.02, 0.02)
    rmse = 1.0 - accuracy + random.uniform(-0.1, 0.1)

    return {
        "accuracy": max(0, min(1, accuracy)),
        "f1": max(0, min(1, f1)),
        "precision": max(0, min(1, precision)),
        "recall": max(0, min(1, recall)),
        "rmse": max(0, rmse),
    }


def save_experiment_results(
    experiment_name: str,
    run_id: str,
    metrics: Dict[str, float],
    config: Dict = None,
    results_dir: Path = Path("results"),
):
    """
    Save experiment results in the format expected by aggregate_results.py

    Args:
        experiment_name: Name of the experiment
        run_id: Unique identifier for this run
        metrics: Dictionary of metric name -> value
        config: Optional configuration to save alongside metrics
        results_dir: Root directory for results
    """
    run_dir = results_dir / experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    # Save config if provided
    if config:
        config_path = run_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

    print(f"Saved results for {experiment_name}/{run_id}")


def main():
    """Run multiple experiments to demonstrate the reporting system."""

    # Experiment 1: Baseline model
    baseline_config = {
        "base_accuracy": 0.75,
        "model": "logistic_regression",
        "learning_rate": 0.01,
    }
    for i in range(5):
        run_id = f"run_{i+1:03d}"
        metrics = run_mock_experiment("baseline", run_id, baseline_config)
        save_experiment_results("baseline", run_id, metrics, baseline_config)

    # Experiment 2: Improved model
    improved_config = {
        "base_accuracy": 0.85,
        "model": "random_forest",
        "n_estimators": 100,
    }
    for i in range(5):
        run_id = f"run_{i+1:03d}"
        metrics = run_mock_experiment("improved_model", run_id, improved_config)
        save_experiment_results("improved_model", run_id, metrics, improved_config)

    # Experiment 3: Alternative approach
    alternative_config = {
        "base_accuracy": 0.82,
        "model": "neural_network",
        "hidden_size": 64,
    }
    for i in range(4):  # fewer runs to test min-runs filtering
        run_id = f"run_{i+1:03d}"
        metrics = run_mock_experiment("neural_network", run_id, alternative_config)
        save_experiment_results("neural_network", run_id, metrics, alternative_config)

    print("\nGenerated experiment results!")
    print("Run the following command to generate a report:")
    print("python scripts/aggregate_results.py --baseline baseline")


if __name__ == "__main__":
    main()
