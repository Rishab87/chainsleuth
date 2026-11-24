"""Headless smoke test of the Streamlit dashboard (needs pipeline artifacts)."""

import pytest

from chainsleuth.config import ARTIFACTS_DIR, ROOT

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "xgb_af.joblib").exists(), reason="run the pipeline first")


def test_dashboard_renders_and_investigates():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "ChainSleuth"
    labels = [m.label for m in at.metric]
    assert {"Time step", "Ground truth", "XGBoost risk"} <= set(labels)

    # type a known transaction id and check the page updates to it
    at.text_input[0].input("230425980").run()  # first row of the dataset, time step 1
    assert not at.exception, at.exception
    step = next(m for m in at.metric if m.label == "Time step")
    assert step.value == "1"

    at.text_input[0].input("999").run()  # not a transaction
    assert any("not in the Elliptic dataset" in e.value for e in at.error)
