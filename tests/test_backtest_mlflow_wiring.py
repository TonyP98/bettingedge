import sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import importlib.util


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backtest_page_uses_mlflow_client(monkeypatch, tmp_path):
    art_dir = tmp_path / "mlruns" / "0" / "r1" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "equity.csv").write_text("equity\n1")
    (art_dir / "diagnostics.html").write_text("<html></html>")

    called = {"download_keys": [], "metric_cards": [], "equity": None}

    def fake_read_csv(path, *a, **k):
        called["equity"] = Path(path)
        return pd.DataFrame({"equity": [1, 2]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    orig_read_text = Path.read_text
    orig_read_bytes = Path.read_bytes

    def fake_read_text(self, *a, **k):
        if self.suffix in {".html", ".csv"}:
            return "<html></html>"
        return orig_read_text(self, *a, **k)

    def fake_read_bytes(self, *a, **k):
        return b"x"

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    class DummyClient:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="0")

        def search_runs(self, experiment_ids, order_by, max_results):
            run = SimpleNamespace(
                info=SimpleNamespace(run_id="r1", artifact_uri=art_dir.as_uri())
            )
            return [run]

        def get_run(self, run_id):
            return SimpleNamespace(info=SimpleNamespace(experiment_id="0", artifact_uri=art_dir.as_uri()))

        def get_metric_history(self, run_id, key):
            return [SimpleNamespace(value=0.2)] if key == "ECE" else []

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: DummyClient())

    class Tab:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class STub:
        session_state = {"show_artifacts": True}
        secrets = {}

        def title(self, *a, **k):
            return None

        def tabs(self, labels):
            return Tab(), Tab()

        def checkbox(self, *a, **k):
            return False

        def slider(self, *a, **k):
            return 0.1

        def multiselect(self, *a, **k):
            return []

        def number_input(self, *a, **k):
            return 1

        def write(self, *a, **k):
            return None

        def button(self, *a, **k):
            return False

        def header(self, *a, **k):
            return None

        def subheader(self, *a, **k):
            return None

        def selectbox(self, *a, **k):
            return None

        def line_chart(self, *a, **k):
            return None

        def download_button(self, label, *a, **k):
            called["download_keys"].append(k.get("key"))
            return None

        class components:
            class v1:
                @staticmethod
                def html(*a, **k):
                    return None

        def info(self, *a, **k):
            return None

        def success(self, *a, **k):
            return None

        def warning(self, *a, **k):
            return None

        def error(self, *a, **k):
            return None

        def code(self, *a, **k):
            return None

        def metric(self, *a, **k):
            return None

    st = STub()
    monkeypatch.setitem(sys.modules, "streamlit", st)

    def fake_metric_card(title, value, help=None):
        called["metric_cards"].append((title, value))

    monkeypatch.setattr("ui._widgets.metric_card", fake_metric_card)

    load_module("ui/pages/04_📈_Backtest.py", "backtest_page_client")

    assert called["equity"] == art_dir / "equity.csv"
    assert any(k.startswith("dl_equity_r1") for k in called["download_keys"])
    assert any(k.startswith("dl_diag_r1") for k in called["download_keys"])
    assert len(set(called["download_keys"])) == len(called["download_keys"])
    assert ("ECE (ensemble)", 0.2) in called["metric_cards"]
