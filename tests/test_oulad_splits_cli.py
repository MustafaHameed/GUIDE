import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _assert_no_leakage(split, total_ids):
    train, val, test = map(set, (split['train'], split['val'], split['test']))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == total_ids
    assert len(train) + len(val) + len(test) == len(total_ids)


def test_cli_creates_valid_splits(tmp_path):
    # Create a mock OULAD-like dataset
    df = pd.DataFrame({
        'id_student': range(12),
        'code_module': ['AAA'] * 6 + ['BBB'] * 6,
        'code_presentation': ['2013B', '2013J', '2014B', '2014J'] * 3,
        'label_pass': [0, 1] * 6,
        'sex': ['Female', 'Male'] * 6,
    })

    dataset_path = tmp_path / 'dataset.parquet'
    df.to_parquet(dataset_path)

    output_dir = tmp_path / 'splits'
    cmd = [sys.executable, 'src/oulad/splits.py', '--dataset', str(dataset_path), '--output-dir', str(output_dir)]
    subprocess.run(cmd, check=True)

    total_ids = set(df['id_student'])

    # Chronological split
    with open(output_dir / 'chronological_split.json') as f:
        chrono = json.load(f)
    _assert_no_leakage(chrono, total_ids)

    # Random split
    with open(output_dir / 'random_split.json') as f:
        rnd = json.load(f)
    _assert_no_leakage(rnd, total_ids)

    # Module holdout splits for each module
    for module in df['code_module'].unique():
        with open(output_dir / f'module_holdout_{module}.json') as f:
            holdout = json.load(f)
        _assert_no_leakage(holdout, total_ids)
        # Ensure test split size equals number of students in that module
        expected_test = df[df['code_module'] == module]['id_student'].nunique()
        assert len(holdout['test']) == expected_test
