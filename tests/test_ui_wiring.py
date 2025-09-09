import importlib.util
import sys
from types import SimpleNamespace

import pandas as pd


# helper to load module by path
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_load_rebuild(monkeypatch, tmp_path):
    # stub streamlit
    class STubs:
        session_state = {}

        def cache_resource(self, func=None, **k):
            if func is None:
                return lambda f: f
            return func

        def cache_data(self, func=None, **k):
            if func is None:
                return lambda f: f
            return func

        def file_uploader(self, *a, **k):
            return None

        def title(self, *a, **k):
            return None

        def subheader(self, *a, **k):
            return None

        def markdown(self, *a, **k):
            return None

        def info(self, *a, **k):
            return None

        def success(self, *a, **k):
            return None

        def spinner(self, *a, **k):
            class Dummy:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            return Dummy()

        def button(self, label, *a, **k):
            return label == "Rebuild Market Probs"

        def write(self, *a, **k):
            return None

        def error(self, *a, **k):
            return None

        def code(self, *a, **k):
            return None

        def code(self, *a, **k):
            return None

        def code(self, *a, **k):
            return None

        def code(self, *a, **k):
            return None

        def dataframe(self, *a, **k):
            return None

    stubs = STubs()
    monkeypatch.setitem(sys.modules, "streamlit", stubs)
    # patch read_duck to provide minimal tables
    expected_matches = pd.DataFrame({"MatchId": [1], "event_time_local": ["2020-01-01"]})
    expected_pre = pd.DataFrame({"MatchId": [1], "AvgH": [2.0], "AvgD": [3.0], "AvgA": [4.0]})
    expected_close = pd.DataFrame({"MatchId": [1], "AvgCH": [2.0], "AvgCD": [3.0], "AvgCA": [4.0]})

    def fake_read(query):
        if "matches" in query:
            return expected_matches
        if "odds_1x2_pre" in query:
            return expected_pre
        if "odds_1x2_close" in query:
            return expected_close
        return pd.DataFrame()

    monkeypatch.setattr("ui._io.read_duck", fake_read)

    called = {}

    def fake_cmp(matches, pre, close):
        called["args"] = (matches, pre, close)
        return {}

    def fake_save(tables):
        called["saved"] = True

    monkeypatch.setattr("engine.data.ingest.compute_market_probs", fake_cmp)
    monkeypatch.setattr("engine.data.ingest.save_tables", fake_save)

    load_module("ui/pages/01_📥_Data_Load.py", "data_load")
    assert "args" in called and "saved" in called
    m, p, c = called["args"]
    assert m.equals(expected_matches)
    assert p.equals(expected_pre)
    assert c.equals(expected_close)


def test_backtest_calls_engine(monkeypatch, tmp_path):
    # stub streamlit with needed functions
    class Tab:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class STub:
        session_state = {}

        def title(self, *a, **k):
            return None

        def tabs(self, labels):
            return Tab(), Tab()

        def checkbox(self, *a, **k):
            return True

        def slider(self, *a, **k):
            return 0.1

        def multiselect(self, label, options, default=None, **k):
            return default or []

        def number_input(self, *a, **k):
            return 1

        def write(self, *a, **k):
            return None

        def button(self, label, *a, **k):
            return label == "Run backtest"

        def header(self, *a, **k):
            return None

        def subheader(self, *a, **k):
            return None

        def selectbox(self, *a, **k):
            return None

        def components(self):
            return None

        class components:
            class v1:
                @staticmethod
                def html(*a, **k):
                    return None

        def success(self, *a, **k):
            return None

        def info(self, *a, **k):
            return None

        def warning(self, *a, **k):
            return None

        def error(self, *a, **k):
            return None

        def code(self, *a, **k):
            return None

    st = STub()
    monkeypatch.setitem(sys.modules, "streamlit", st)

    monkeypatch.setattr("ui._widgets.equity_plot", lambda df: None)
    monkeypatch.setattr("ui._widgets.metric_card", lambda *a, **k: None)

    called = {}

    def fake_apply(df, cfg):
        called["apply"] = True
        return df.assign(pnl=[0.1, -0.05])

    monkeypatch.setattr("engine.eval.backtester.apply_conformal_guard", fake_apply)
    monkeypatch.setattr("engine.eval.diagnostics.run_diagnostics", lambda: str(tmp_path / "d.html"))
    monkeypatch.setattr("engine.utils.mlflow_utils.start_run", lambda *a, **k: None)
    monkeypatch.setattr("engine.utils.mlflow_utils.log_artifact", lambda *a, **k: None)
    monkeypatch.setattr("engine.utils.mlflow_utils.end_run", lambda *a, **k: None)
    (tmp_path / "d.html").write_text("<html></html>")

    class DummyClient:
        def get_run(self, run_id):
            return SimpleNamespace(info=SimpleNamespace(experiment_id="0", artifact_uri="file:/tmp"))

        def get_metric_history(self, run_id, key):
            return []

        def get_experiment_by_name(self, name):
            return None

        def search_runs(self, *a, **k):
            return []

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: DummyClient())

    load_module("ui/pages/04_📈_Backtest.py", "backtest")
    assert called.get("apply")
