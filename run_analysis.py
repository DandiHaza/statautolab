from __future__ import annotations

import argparse
import sys

from app.analysis_runner import AnalysisRunContext, build_output_dir, build_run_context, execute_analysis, record_failed_run
from app.config import load_config_file, resolve_settings


class SmartHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def parse_feature_columns(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    columns = [column.strip() for column in raw_value.split(",") if column.strip()]
    return columns or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV/XLSX 데이터를 대상으로 EDA와 baseline 모델 비교를 수행합니다.",
        epilog=(
            "예시:\n"
            "  python run_analysis.py --input data/examples/eda_sample.csv\n"
            "  python run_analysis.py --input data/examples/classification_sample.csv --target buy\n"
            "  python run_analysis.py --input data/examples/regression_sample.csv --target spending_score --features age,income,city,visits\n"
            "  python run_analysis.py --config configs/default.yaml"
        ),
        formatter_class=SmartHelpFormatter,
    )
    parser.add_argument("input_file", nargs="?", help="입력 데이터 파일 경로(.csv, .xlsx, .xls)")

    required_or_common = parser.add_argument_group("주요 입력 옵션")
    required_or_common.add_argument(
        "--config",
        default=None,
        help="YAML 설정 파일 경로. CLI 인자와 함께 주어지면 CLI 값이 우선합니다.",
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
        help="종속변수로 사용할 타깃 컬럼명",
    )
    required_or_common.add_argument(
        "--features",
        default=None,
        help="독립변수로 사용할 컬럼 목록. 쉼표로 구분합니다. 예: age,income,city",
    )

    optional = parser.add_argument_group("선택 옵션")
    optional.add_argument("--output-dir", default=None, help="결과 파일을 저장할 기본 폴더")
    optional.add_argument("--report-format", choices=["md", "html"], default=None, help="최종 리포트 형식")
    optional.add_argument(
        "--task-type",
        choices=["auto", "regression", "classification"],
        default=None,
        help="문제 유형을 강제로 지정합니다. 기본값 auto는 타깃 dtype 기준으로 자동 판별합니다.",
    )
    optional.add_argument("--random-state", type=int, default=None, help="분할과 모델 학습에 사용할 랜덤 시드")
    optional.add_argument("--test-size", type=float, default=None, help="holdout 검증 데이터 비율(0과 1 사이)")
    optional.add_argument("--eval-method", choices=["holdout", "cv"], default=None, help="모델 평가 방식")
    optional.add_argument("--cv-folds", type=int, default=None, help="교차검증 fold 수")
    args = parser.parse_args(argv)
    args.input_file = args.input_file_option or args.input_file
    return args


def build_cli_values(args: argparse.Namespace) -> dict[str, object]:
    return {
        "input_path": args.input_file_option or args.input_file,
        "target": args.target,
        "feature_columns": parse_feature_columns(args.features),
        "output_dir": args.output_dir,
        "report_format": args.report_format,
        "task_type": args.task_type,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "eval_method": args.eval_method,
        "cv_folds": args.cv_folds,
    }


def print_run_summary(result) -> None:
    context = result.context
    print(f"분석이 완료되었습니다: {context.input_path}")
    print(f"리포트 저장 위치: {result.report_path}")
    print(f"차트 저장 위치: {result.charts_dir}")
    print(f"실행 설정 스냅샷 저장 위치: {context.output_dir / 'config_snapshot.json'}")
    print(f"데이터 요약 저장 위치: {context.output_dir / 'data_summary.json'}")
    print(f"이상치 요약 CSV 저장 위치: {context.output_dir / 'outlier_summary.csv'}")
    print(f"경고 요약 Markdown 저장 위치: {result.warnings_markdown_path}")
    print(f"경고 JSON 저장 위치: {result.warnings_json_path}")
    print(f"실험 로그 저장 위치: {result.experiment_log_path}")
    if result.model_result is not None:
        print(f"전처리 요약 Markdown 저장 위치: {context.output_dir / 'preprocessing_summary.md'}")
        print(f"모델 비교 CSV 저장 위치: {context.output_dir / 'model_comparison.csv'}")
        print(f"모델 요약 Markdown 저장 위치: {context.output_dir / 'model_summary.md'}")
        print(f"선택된 독립변수: {', '.join(result.model_result.preprocessing_summary.selected_feature_columns)}")
        print(f"best model artifact 저장 위치: {result.model_artifacts['model']}")
        print(f"모델 metadata 저장 위치: {result.model_artifacts['metadata']}")


def main() -> None:
    context: AnalysisRunContext | None = None
    try:
        args = parse_args()
        config_values = load_config_file(args.config)
        settings = resolve_settings(build_cli_values(args), config_values)
        context = build_run_context(settings, config_path=args.config)
        result = execute_analysis(context)
        print_run_summary(result)
    except Exception as exc:
        if context is not None:
            record_failed_run(context, exc)
        print(f"오류: {exc}", file=sys.stderr)
        print(
            "입력 파일 경로, config 설정값, 타깃 컬럼, 독립변수 목록, 출력 폴더 권한을 확인해 주세요. "
            "예시: `python run_analysis.py --input data/examples/regression_sample.csv --target spending_score --features age,income,city,visits`",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
