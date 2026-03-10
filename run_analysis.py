from __future__ import annotations

import argparse
from pathlib import Path

from app.io import load_dataset
from app.modeling import save_model_results, train_baseline_models
from app.profiling import profile_dataset
from app.report import build_markdown_report, save_markdown_report
from app.visualization import generate_correlation_heatmap, generate_histograms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic exploratory data analysis.")
    parser.add_argument("input_file", help="Path to a CSV/XLSX dataset")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where charts and report will be written",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column for automatic baseline modeling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "charts"

    df = load_dataset(input_path)
    profile = profile_dataset(df)

    histogram_paths = generate_histograms(df, charts_dir)
    correlation_path = generate_correlation_heatmap(profile.correlation, charts_dir)
    model_result = None

    if args.target:
        model_result = train_baseline_models(df, args.target)
        save_model_results(model_result, output_dir)

    report_content = build_markdown_report(
        source_name=input_path.name,
        profile=profile,
        histogram_paths=histogram_paths,
        correlation_path=correlation_path,
        model_result=model_result,
    )
    report_path = save_markdown_report(report_content, output_dir / "report.md")

    print(f"Analysis completed for: {input_path}")
    print(f"Report saved to: {report_path}")
    print(f"Charts saved to: {charts_dir}")
    if model_result is not None:
        print(f"Model comparison saved to: {output_dir / 'model_comparison.csv'}")
        print(f"Model comparison markdown saved to: {output_dir / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
