from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.experiment import append_experiment_log, save_config_snapshot, save_data_summary
from app.io import load_dataset
from app.preprocessing import save_preprocessing_summary
from app.profiling import ProfileResult, profile_dataset
from app.report import build_markdown_report, save_html_report, save_markdown_report
from app.train import ModelResult, save_model_results, train_and_compare_models
from app.visualization import generate_boxplots, generate_correlation_heatmap, generate_histograms
from app.warnings_log import WarningRecord, collect_data_warnings, dedupe_warnings, save_warnings_summary


@dataclass
class AnalysisRunContext:
    settings: dict[str, object]
    config_path: str | None
    run_timestamp: datetime
    input_path: Path
    output_dir: Path
    run_id: str


@dataclass
class AnalysisRunResult:
    context: AnalysisRunContext
    profile: ProfileResult
    report_path: Path
    charts_dir: Path
    warnings_markdown_path: Path
    warnings_json_path: Path
    experiment_log_path: Path
    model_result: ModelResult | None
    model_artifacts: dict[str, Path]
    warnings: list[WarningRecord]


def build_output_dir(base_output_dir: str | Path, run_time: datetime | None = None) -> Path:
    timestamp = run_time or datetime.now()
    base_path = Path(base_output_dir)
    return base_path / timestamp.strftime("%Y%m%d") / timestamp.strftime("%H%M%S")


def build_run_context(settings: dict[str, object], config_path: str | None = None) -> AnalysisRunContext:
    run_timestamp = datetime.now()
    input_path = Path(str(settings["input_path"]))
    output_dir = build_output_dir(str(settings["output_dir"]), run_timestamp)
    run_id = f"{output_dir.parent.name}_{output_dir.name}"
    return AnalysisRunContext(
        settings=settings,
        config_path=config_path,
        run_timestamp=run_timestamp,
        input_path=input_path,
        output_dir=output_dir,
        run_id=run_id,
    )


def execute_analysis(context: AnalysisRunContext) -> AnalysisRunResult:
    settings = context.settings
    charts_dir = context.output_dir / "charts"
    warning_records: list[WarningRecord] = []

    df = load_dataset(context.input_path)
    profile = profile_dataset(df)
    save_config_snapshot(_build_config_snapshot(context), context.output_dir)
    save_data_summary(profile, context.input_path, context.output_dir)

    histogram_paths = generate_histograms(df, charts_dir)
    boxplot_paths = generate_boxplots(df, charts_dir)
    correlation_path = generate_correlation_heatmap(profile.correlation, charts_dir)
    profile.outliers.to_csv(context.output_dir / "outlier_summary.csv", index=False)

    model_result = None
    preprocessing_summary = None
    model_artifacts: dict[str, Path] = {}

    if settings["target"]:
        model_result = train_and_compare_models(
            df,
            str(settings["target"]),
            feature_columns=list(settings["feature_columns"]) if settings.get("feature_columns") is not None else None,
            selected_model=str(settings["selected_model"]) if settings.get("selected_model") is not None else None,
            test_size=float(settings["test_size"]),
            task_type=str(settings["task_type"]),
            random_state=int(settings["random_state"]),
            eval_method=str(settings["eval_method"]),
            cv_folds=int(settings["cv_folds"]),
        )
        preprocessing_summary = model_result.preprocessing_summary
        warning_records.extend(model_result.warnings)
        save_preprocessing_summary(preprocessing_summary, context.output_dir)
        _, _, model_path, metadata_path = save_model_results(model_result, context.output_dir)
        model_artifacts = {"model": model_path, "metadata": metadata_path}
    elif settings["task_type"] != "auto":
        raise ValueError("`--task-type regression|classification`를 사용하려면 `--target`도 함께 지정해야 합니다.")

    target_series = None
    problem_type = None
    if settings["target"]:
        target_series = df[str(settings["target"])].dropna()
        problem_type = model_result.problem_type if model_result is not None else None

    warning_records.extend(
        collect_data_warnings(
            profile=profile,
            preprocessing_summary=preprocessing_summary,
            target_series=target_series,
            problem_type=problem_type,
        )
    )
    warning_records = dedupe_warnings(warning_records)
    warnings_md_path, warnings_json_path = save_warnings_summary(warning_records, context.output_dir)

    report_content = build_markdown_report(
        source_name=context.input_path.name,
        profile=profile,
        histogram_paths=histogram_paths,
        boxplot_paths=boxplot_paths,
        correlation_path=correlation_path,
        preprocessing_summary=preprocessing_summary,
        model_result=model_result,
        warnings=warning_records,
        model_artifacts=model_artifacts,
    )

    report_path = _save_report(report_content, context)
    log_path = append_experiment_log(
        base_output_dir=str(settings["output_dir"]),
        run_id=context.run_id,
        timestamp=context.run_timestamp.isoformat(timespec="seconds"),
        input_file=context.input_path,
        target=str(settings["target"]) if settings["target"] is not None else None,
        task_type=str(settings["task_type"]),
        eval_method=str(settings["eval_method"]),
        cv_folds=int(settings["cv_folds"]),
        model_result=model_result,
        output_path=context.output_dir,
        success=True,
        warning_count=len(warning_records),
    )

    return AnalysisRunResult(
        context=context,
        profile=profile,
        report_path=report_path,
        charts_dir=charts_dir,
        warnings_markdown_path=warnings_md_path,
        warnings_json_path=warnings_json_path,
        experiment_log_path=log_path,
        model_result=model_result,
        model_artifacts=model_artifacts,
        warnings=warning_records,
    )


def record_failed_run(
    context: AnalysisRunContext,
    exc: Exception,
    warning_records: list[WarningRecord] | None = None,
) -> None:
    warnings = list(warning_records or [])
    warnings.append(
        WarningRecord(
            code="run_failed",
            level="error",
            message="실행 중 예외가 발생해 분석이 완료되지 않았습니다.",
            details={"error": str(exc)},
        )
    )
    warnings = dedupe_warnings(warnings)
    save_warnings_summary(warnings, context.output_dir)
    append_experiment_log(
        base_output_dir=str(context.settings["output_dir"]),
        run_id=context.run_id,
        timestamp=context.run_timestamp.isoformat(timespec="seconds"),
        input_file=context.input_path,
        target=str(context.settings["target"]) if context.settings["target"] is not None else None,
        task_type=str(context.settings["task_type"]),
        eval_method=str(context.settings["eval_method"]),
        cv_folds=int(context.settings["cv_folds"]),
        model_result=None,
        output_path=context.output_dir,
        success=False,
        warning_count=len(warnings),
    )


def _build_config_snapshot(context: AnalysisRunContext) -> dict[str, object]:
    settings = context.settings
    return {
        "run_id": context.run_id,
        "timestamp": context.run_timestamp.isoformat(timespec="seconds"),
        "input_file": str(context.input_path),
        "config_path": context.config_path,
        "target": settings["target"],
        "feature_columns": settings["feature_columns"],
        "selected_model": settings["selected_model"],
        "task_type": settings["task_type"],
        "report_format": settings["report_format"],
        "random_state": settings["random_state"],
        "test_size": settings["test_size"],
        "eval_method": settings["eval_method"],
        "cv_folds": settings["cv_folds"],
        "output_dir": str(context.output_dir),
    }


def _save_report(report_content: str, context: AnalysisRunContext) -> Path:
    if context.settings["report_format"] == "html":
        return save_html_report(
            report_content,
            context.output_dir / "report.html",
            f"데이터 분석 리포트: {context.input_path.name}",
        )
    return save_markdown_report(report_content, context.output_dir / "report.md")
