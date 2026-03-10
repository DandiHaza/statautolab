from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from app.config import load_config_file, resolve_settings
from app.experiment import append_experiment_log, save_config_snapshot, save_data_summary
from app.io import load_dataset
from app.preprocessing import save_preprocessing_summary
from app.profiling import profile_dataset
from app.report import build_markdown_report, save_html_report, save_markdown_report
from app.train import save_model_results, train_and_compare_models
from app.visualization import generate_boxplots, generate_correlation_heatmap, generate_histograms


class SmartHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV/XLSX 데이터에 대해 EDA와 baseline 모델 비교를 수행합니다.",
        epilog=(
            "예시:\n"
            "  python run_analysis.py --input data/sample.csv\n"
            "  python run_analysis.py --config configs/default.yaml\n"
            "  python run_analysis.py --input data/sample.csv --target buy\n"
            "  python run_analysis.py --input my_data.csv --target target_column --task-type classification --report-format html"
        ),
        formatter_class=SmartHelpFormatter,
    )
    parser.add_argument("input_file", nargs="?", help="입력 데이터 파일 경로(.csv, .xlsx, .xls)")

    required_or_common = parser.add_argument_group("주요 입력 옵션")
    required_or_common.add_argument(
        "--config",
        default=None,
        help="YAML 설정 파일 경로. CLI 인자가 동시에 주어지면 CLI 값이 우선합니다.",
    )
    required_or_common.add_argument(
        "--input",
        dest="input_file_option",
        default=None,
        help="입력 데이터 파일 경로(.csv, .xlsx, .xls)",
    )
    required_or_common.add_argument(
        "--target",
        default=None,
        help="모델 자동화를 실행할 때 사용할 타깃 컬럼명",
    )

    optional = parser.add_argument_group("선택 옵션")
    optional.add_argument(
        "--output-dir",
        default=None,
        help="결과 파일이 저장될 기본 폴더",
    )
    optional.add_argument(
        "--report-format",
        choices=["md", "html"],
        default=None,
        help="최종 리포트 저장 형식",
    )
    optional.add_argument(
        "--task-type",
        choices=["auto", "regression", "classification"],
        default=None,
        help="문제 유형 강제 지정. 기본값 auto는 타깃 dtype으로 자동 판별",
    )
    optional.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="학습/검증 분할과 랜덤 모델에 사용할 시드",
    )
    optional.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="검증 데이터 비율(0과 1 사이)",
    )
    args = parser.parse_args(argv)
    args.input_file = args.input_file_option or args.input_file
    return args


def build_output_dir(base_output_dir: str | Path, run_time: datetime | None = None) -> Path:
    timestamp = run_time or datetime.now()
    base_path = Path(base_output_dir)
    return base_path / timestamp.strftime("%Y%m%d") / timestamp.strftime("%H%M%S")


def main() -> None:
    try:
        args = parse_args()
        config_values = load_config_file(args.config)
        cli_values = {
            "input_path": args.input_file_option or args.input_file,
            "target": args.target,
            "output_dir": args.output_dir,
            "report_format": args.report_format,
            "task_type": args.task_type,
            "random_state": args.random_state,
            "test_size": args.test_size,
        }
        settings = resolve_settings(cli_values, config_values)
        run_timestamp = datetime.now()
        input_path = Path(str(settings["input_path"]))
        output_dir = build_output_dir(str(settings["output_dir"]), run_timestamp)
        charts_dir = output_dir / "charts"
        run_id = f"{output_dir.parent.name}_{output_dir.name}"

        df = load_dataset(input_path)
        profile = profile_dataset(df)
        config_snapshot = {
            "run_id": run_id,
            "timestamp": run_timestamp.isoformat(timespec="seconds"),
            "input_file": str(input_path),
            "config_path": args.config,
            "target": settings["target"],
            "task_type": settings["task_type"],
            "report_format": settings["report_format"],
            "random_state": settings["random_state"],
            "test_size": settings["test_size"],
            "output_dir": str(output_dir),
        }
        save_config_snapshot(config_snapshot, output_dir)
        save_data_summary(profile, input_path, output_dir)

        histogram_paths = generate_histograms(df, charts_dir)
        boxplot_paths = generate_boxplots(df, charts_dir)
        correlation_path = generate_correlation_heatmap(profile.correlation, charts_dir)
        profile.outliers.to_csv(output_dir / "outlier_summary.csv", index=False)
        model_result = None
        preprocessing_summary = None

        if settings["target"]:
            model_result = train_and_compare_models(
                df,
                str(settings["target"]),
                test_size=float(settings["test_size"]),
                task_type=str(settings["task_type"]),
                random_state=int(settings["random_state"]),
            )
            preprocessing_summary = model_result.preprocessing_summary
            save_preprocessing_summary(preprocessing_summary, output_dir)
            save_model_results(model_result, output_dir)
        elif settings["task_type"] != "auto":
            raise ValueError("`--task-type regression|classification`를 사용하려면 `--target`도 함께 지정해야 합니다.")

        report_content = build_markdown_report(
            source_name=input_path.name,
            profile=profile,
            histogram_paths=histogram_paths,
            boxplot_paths=boxplot_paths,
            correlation_path=correlation_path,
            preprocessing_summary=preprocessing_summary,
            model_result=model_result,
        )
        if settings["report_format"] == "html":
            report_path = save_html_report(report_content, output_dir / "report.html", f"데이터 분석 리포트: {input_path.name}")
        else:
            report_path = save_markdown_report(report_content, output_dir / "report.md")

        log_path = append_experiment_log(
            base_output_dir=str(settings["output_dir"]),
            run_id=run_id,
            timestamp=run_timestamp.isoformat(timespec="seconds"),
            input_file=input_path,
            target=str(settings["target"]) if settings["target"] is not None else None,
            task_type=str(settings["task_type"]),
            model_result=model_result,
            output_path=output_dir,
        )

        print(f"분석이 완료되었습니다: {input_path}")
        print(f"리포트 저장 위치: {report_path}")
        print(f"차트 저장 위치: {charts_dir}")
        print(f"실험 설정 스냅샷 저장 위치: {output_dir / 'config_snapshot.json'}")
        print(f"데이터 요약 저장 위치: {output_dir / 'data_summary.json'}")
        print(f"이상치 요약 CSV 저장 위치: {output_dir / 'outlier_summary.csv'}")
        print(f"실험 로그 저장 위치: {log_path}")
        if model_result is not None:
            print(f"전처리 요약 Markdown 저장 위치: {output_dir / 'preprocessing_summary.md'}")
            print(f"모델 비교 CSV 저장 위치: {output_dir / 'model_comparison.csv'}")
            print(f"모델 요약 Markdown 저장 위치: {output_dir / 'model_summary.md'}")
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        print(
            "입력 파일 경로, config 설정값, `--target` 컬럼명, `--report-format`, 그리고 출력 폴더 권한을 확인하세요. "
            "예시: `python run_analysis.py --config configs/default.yaml`",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
