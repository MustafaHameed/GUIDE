"""Test the aggregate_results.py script functionality."""
import tempfile
import json
from pathlib import Path
import pytest
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from aggregate_results import load_metrics_json, load_metrics_csv, discover_runs, get_primary_metric


def test_load_metrics_json():
    """Test loading metrics from JSON format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"accuracy": 0.85, "f1": 0.82}, f)
        temp_path = Path(f.name)
    
    try:
        metrics = load_metrics_json(temp_path)
        assert metrics == {"accuracy": 0.85, "f1": 0.82}
    finally:
        temp_path.unlink()


def test_load_metrics_json_nested():
    """Test loading metrics from nested JSON format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"metrics": {"accuracy": 0.85, "f1": 0.82}}, f)
        temp_path = Path(f.name)
    
    try:
        metrics = load_metrics_json(temp_path)
        assert metrics == {"accuracy": 0.85, "f1": 0.82}
    finally:
        temp_path.unlink()


def test_load_metrics_csv():
    """Test loading metrics from CSV format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("metric,value\naccuracy,0.85\nf1,0.82\n")
        temp_path = Path(f.name)
    
    try:
        metrics = load_metrics_csv(temp_path)
        assert metrics == {"accuracy": 0.85, "f1": 0.82}
    finally:
        temp_path.unlink()


def test_discover_runs():
    """Test discovering experiment runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test structure
        exp1_run1 = tmpdir / "experiment1" / "run1"
        exp1_run2 = tmpdir / "experiment1" / "run2"
        exp2_run1 = tmpdir / "experiment2" / "run1"
        
        exp1_run1.mkdir(parents=True)
        exp1_run2.mkdir(parents=True)
        exp2_run1.mkdir(parents=True)
        
        runs = discover_runs(tmpdir)
        assert len(runs) == 3
        
        # Check structure
        exp_names = {run[0] for run in runs}
        assert exp_names == {"experiment1", "experiment2"}


def test_get_primary_metric():
    """Test primary metric selection logic."""
    # Test explicit metric selection
    metrics = ["accuracy", "f1", "precision"]
    assert get_primary_metric(metrics, "f1") == "f1"
    
    # Test case-insensitive matching
    assert get_primary_metric(metrics, "ACCURACY") == "accuracy"
    
    # Test auto-detection
    assert get_primary_metric(["custom", "accuracy", "other"], None) == "accuracy"
    assert get_primary_metric(["rmse", "mae", "r2"], None) == "rmse"
    
    # Test fallback to first metric
    assert get_primary_metric(["custom_metric"], None) == "custom_metric"


if __name__ == "__main__":
    pytest.main([__file__])