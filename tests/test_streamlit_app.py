from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "streamlit_app.py"


def _run_app(timeout: int = 120) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=timeout).run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def _cleanup_session_dirs(session_id: str) -> None:
    for base in ("outputs", ".streamlit_uploads"):
        shutil.rmtree(REPO_ROOT / base / session_id, ignore_errors=True)


@pytest.fixture
def app() -> AppTest:
    instance = _run_app()
    yield instance
    _cleanup_session_dirs(instance.session_state["session_id"])


class TestLandingScreen:
    def test_offers_sample_datasets_before_any_upload(self, app: AppTest) -> None:
        # A visitor without a CSV on hand must still have a way into the demo.
        start_buttons = [button for button in app.button if button.label == "이 데이터로 시작"]

        assert len(start_buttons) >= 1

    def test_every_advertised_sample_file_exists(self) -> None:
        from ui.constants import SAMPLE_DATASETS

        missing = [sample["path"] for sample in SAMPLE_DATASETS if not (REPO_ROOT / sample["path"]).exists()]

        assert missing == [], f"예제 파일이 저장소에 없습니다: {missing}"


class TestSessionIsolation:
    def test_session_id_exists_from_first_render(self, app: AppTest) -> None:
        assert app.session_state["session_id"]

    def test_concurrent_sessions_get_separate_ids(self, app: AppTest) -> None:
        # Two visitors uploading the same file name must not overwrite each other.
        other = _run_app()
        try:
            assert app.session_state["session_id"] != other.session_state["session_id"]
        finally:
            _cleanup_session_dirs(other.session_state["session_id"])

    def test_working_paths_are_scoped_to_the_session(self, app: AppTest) -> None:
        from ui.constants import OUTPUT_ROOT, UPLOAD_DIR

        session_id = app.session_state["session_id"]

        assert str(OUTPUT_ROOT / session_id).endswith(session_id)
        assert str(UPLOAD_DIR / session_id).endswith(session_id)


class TestSampleSelection:
    def test_choosing_a_sample_prefills_its_target(self, app: AppTest) -> None:
        app.button(key="sample-classification").click().run()

        assert not app.exception
        assert app.session_state["target_option"] == "buy"

    def test_eda_only_sample_leaves_target_unset(self, app: AppTest) -> None:
        from ui.constants import NO_TARGET

        app.button(key="sample-eda").click().run()

        assert app.session_state["target_option"] == NO_TARGET

    def test_returning_to_the_picker_clears_the_selection(self, app: AppTest) -> None:
        app.button(key="sample-classification").click().run()
        app.button(key="clear-sample").click().run()

        assert not app.exception
        assert "sample_dataset_key" not in app.session_state
        assert any(button.label == "이 데이터로 시작" for button in app.button)


class TestFeatureDefaults:
    def test_identifier_columns_are_not_preselected(self, app: AppTest) -> None:
        # Passing an explicit column list turns off the pipeline's identifier detection,
        # so preselecting every column would quietly feed customer_id to the model.
        app.button(key="sample-regression").click().run()

        assert not app.exception
        assert "customer_id" not in app.session_state["selected_features"]
        assert app.session_state["selected_features"] == ["age", "income", "city", "visits"]

    def test_auto_exclusion_is_explained_on_screen(self, app: AppTest) -> None:
        app.button(key="sample-regression").click().run()

        notes = [caption.value for caption in app.caption if "customer_id" in caption.value]
        assert notes, "무엇이 왜 제외됐는지 화면에 표시되지 않습니다."

    def test_regression_dashboard_reports_real_numbers(self, app: AppTest) -> None:
        # With customer_id included the design matrix is rank-deficient and the
        # dashboard used to show nan for Adj. R2 and the F statistic.
        app.button(key="sample-regression").click().run()
        app.button(key="run_analysis").click().run()

        assert not app.exception
        values = {metric.label: metric.value for metric in app.metric}
        assert "nan" not in values.get("Adj. R2", "nan").lower()


class TestEvaluationOptions:
    def test_holdout_exposes_test_size_and_cv_exposes_folds(self, app: AppTest) -> None:
        app.button(key="sample-classification").click().run()

        assert "test_size" in app.session_state, "홀드아웃에서 검증 비율 슬라이더가 없습니다."

        app.selectbox(key="eval_method").select("cv").run()

        assert not app.exception
        assert "cv_folds" in app.session_state, "교차검증에서 fold 슬라이더가 없습니다."


class TestAnalysisRun:
    def test_sample_dataset_runs_end_to_end_and_reports_metrics(self, app: AppTest) -> None:
        app.button(key="sample-classification").click().run()
        app.button(key="run_analysis").click().run()

        assert not app.exception, [error.value for error in app.exception]

        metric_labels = [metric.label for metric in app.metric]
        assert "Accuracy" in metric_labels
