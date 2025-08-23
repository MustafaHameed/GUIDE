import pandas as pd
from pathlib import Path

from src.transfer.uci_transfer import run_bidirectional_transfer
import src.transfer.uci_transfer as uci_transfer


def test_run_bidirectional_transfer_aligns_and_saves(tmp_path, monkeypatch):
    # Create synthetic OULAD dataset
    oulad_df = pd.DataFrame(
        {
            "sex": ["Male", "Female", "Male"],
            "age_band": ["0-35", "35-55", "0-35"],
            "imd_band": ["0-10%", "20-30%", "30-40%"],
            "vle_total_clicks": [100, 200, 150],
            "label_pass": [1, 0, 1],
        }
    )
    oulad_path = tmp_path / "oulad.parquet"
    oulad_df.to_parquet(oulad_path)

    # Create synthetic UCI dataset
    uci_df = pd.DataFrame(
        {
            "sex": ["M", "F", "M"],
            "age": [18, 20, 22],
            "Medu": [2, 3, 1],
            "absences": [5, 3, 10],
            "G3": [12, 8, 15],
        }
    )
    uci_path = tmp_path / "uci.csv"
    uci_df.to_csv(uci_path, index=False)

    calls = {}

    # Track file loading and mapping
    original_read_parquet = pd.read_parquet

    def patched_read_parquet(path, *args, **kwargs):
        calls["parquet"] = path
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", patched_read_parquet)

    import src.data as data_module

    original_load_data = data_module.load_data

    def patched_load_data(path, *args, **kwargs):
        calls["csv"] = path
        return original_load_data(path, *args, **kwargs)

    monkeypatch.setattr(uci_transfer, "load_data", patched_load_data)

    original_mapping = uci_transfer.create_shared_feature_mapping

    def patched_mapping():
        calls["mapping"] = True
        return original_mapping()

    monkeypatch.setattr(uci_transfer, "create_shared_feature_mapping", patched_mapping)

    # Speed up MLP training during tests
    monkeypatch.setattr(uci_transfer, "MLP_PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(uci_transfer, "MLP_FINETUNE_EPOCHS", 1)

    table_path = tmp_path / "tables" / "transfer_results.csv"
    figure_path = tmp_path / "figures" / "transfer_performance.png"
    output_dir = tmp_path / "output"

    run_bidirectional_transfer(
        str(oulad_path),
        str(uci_path),
        output_dir,
        table_path=table_path,
        figure_path=figure_path,
    )

    assert calls["parquet"] == str(oulad_path)
    assert calls["csv"] == str(uci_path)
    assert calls["mapping"] is True

    assert table_path.exists()
    results_df = pd.read_csv(table_path)
    assert "accuracy" in results_df.columns
    assert "worst_group_accuracy" in results_df.columns

    assert figure_path.exists()


def test_transfer_experiment_mlp(monkeypatch):
    """Ensure the MLP branch trains and returns metrics."""
    # Minimal numeric dataset with encoded sex column
    source = pd.DataFrame(
        {
            "feat": [0.1, 0.9, 0.3, 0.7],
            "sex": [0, 1, 0, 1],
            "label": [0, 1, 0, 1],
        }
    )
    target = pd.DataFrame(
        {
            "feat": [0.2, 0.8],
            "sex": [0, 1],
            "label": [0, 1],
        }
    )

    # Speed up training for unit test
    monkeypatch.setattr(uci_transfer, "MLP_PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(uci_transfer, "MLP_FINETUNE_EPOCHS", 1)

    results = uci_transfer.transfer_experiment(source, target, model_type="mlp")
    assert "accuracy" in results
    assert "auc" in results
    # Should include sex-specific accuracy metrics
    assert any(k.startswith("accuracy_") for k in results)
