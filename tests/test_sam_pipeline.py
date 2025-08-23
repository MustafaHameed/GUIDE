from pathlib import Path
import types

import pandas as pd


def _make_schema(tmp_path: Path) -> Path:
    schema_text = (
        "## Required Fields\n"
        "### case_id\n"
        "### activity\n"
        "### timestamp\n"
        "## Optional Fields\n"
        "### resource\n"
    )
    schema_path = tmp_path / "schema.md"
    schema_path.write_text(schema_text)
    return schema_path


def _make_sam_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "case_id": [1, 1, 2],
            "activity": ["start", "end", "start"],
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="H"),
            "resource": ["r1", "r1", "r2"],
        }
    )
    csv_path = tmp_path / "sam.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_and_convert(monkeypatch, tmp_path):
    from src.process_mining import sam_pipeline

    sam_pipeline.HAS_PM4PY = True

    calls: dict[str, list[str]] = {}

    class DummyVariants:
        class TO_EVENT_LOG:
            value = types.SimpleNamespace(
                Parameters=types.SimpleNamespace(CASE_ID_KEY="case:concept:name")
            )

    def dummy_apply(df, parameters=None, variant=None):
        calls["columns"] = list(df.columns)
        return ["event_log"]

    sam_pipeline.log_converter = types.SimpleNamespace(
        Variants=DummyVariants, apply=dummy_apply
    )

    miner = sam_pipeline.SAMProcessMiner(
        output_dir=tmp_path / "out",
        figures_dir=tmp_path / "figures",
        tables_dir=tmp_path / "tables",
    )

    sam_csv = _make_sam_csv(tmp_path)
    schema_path = _make_schema(tmp_path)

    df = miner.load_and_validate_sam_data(sam_csv, schema_path)
    assert set(["case_id", "activity", "timestamp", "resource"]).issubset(df.columns)

    event_log = miner.convert_to_event_log(df)
    assert event_log == ["event_log"]
    assert {"case:concept:name", "concept:name", "time:timestamp", "org:resource"}.issubset(
        set(calls["columns"])
    )

