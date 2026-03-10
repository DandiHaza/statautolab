from __future__ import annotations

import shutil
from pathlib import Path

from run_analysis import main


def test_smoke_pipeline_runs_for_eda_and_model(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    eda_output = (repo_root / "outputs_smoke_eda").resolve()
    model_output = (repo_root / "outputs_smoke_model").resolve()
    shutil.rmtree(eda_output, ignore_errors=True)
    shutil.rmtree(model_output, ignore_errors=True)

    try:
        eda_csv = (repo_root / "data/examples/eda_sample.csv").resolve()
        model_csv = (repo_root / "data/examples/classification_sample.csv").resolve()

        monkeypatch.setattr(
            "sys.argv",
            ["run_analysis.py", "--input", str(eda_csv), "--output-dir", str(eda_output)],
        )
        main()

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_analysis.py",
                "--input",
                str(model_csv),
                "--target",
                "buy",
                "--output-dir",
                str(model_output),
            ],
        )
        main()

        assert any(eda_output.rglob("report.md"))
        assert any(model_output.rglob("report.md"))
        assert any(model_output.rglob("model_comparison.csv"))
    finally:
        shutil.rmtree(eda_output, ignore_errors=True)
        shutil.rmtree(model_output, ignore_errors=True)
