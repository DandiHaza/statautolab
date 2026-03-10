# StatAutoLab MVP

StatAutoLab MVP is a minimal exploratory data analysis and baseline modeling tool for CSV and Excel files.

## Features

- Load `.csv`, `.xlsx`, and `.xls` datasets
- Generate dataset preview and column summary
- Analyze missing values
- Produce descriptive statistics for numeric and categorical columns
- Save histograms for numeric columns
- Save a correlation matrix heatmap when numeric columns are available
- Build a Markdown report with all analysis outputs
- Detect regression/classification tasks from `--target`
- Train baseline models and compare validation performance

## Project Structure

```text
StatAutoLab/
|-- app/
|   |-- __init__.py
|   |-- io.py
|   |-- modeling.py
|   |-- profiling.py
|   |-- report.py
|   `-- visualization.py
|-- run_analysis.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_analysis.py path/to/data.csv --output-dir outputs
python run_analysis.py path/to/data.csv --target target_column --output-dir outputs
```

## Output

- `outputs/report.md`
- `outputs/charts/histogram_<column>.png`
- `outputs/charts/correlation_matrix.png`
- `outputs/model_comparison.csv`
- `outputs/model_comparison.md`

## Notes

- Numeric targets are treated as regression; non-numeric targets are treated as classification.
- Correlation analysis and histograms are generated only for numeric columns.
