import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
import pandas as pd


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mlflow_artifacts_viewer(monkeypatch, tmp_path):
    # stub streamlit with minimal API
    class SessionState(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    class STub:
        session_state = SessionState()
        secrets = {}

        def set_page_config(self, *a, **k):
            return None

        def tabs(self, labels):
            class Tab:
                def __enter__(self_inner):
                    return self

                def __exit__(self_inner, *exc):
                    return False

            return [Tab() for _ in labels]

        def write(self, *a, **k):
            return None

        def selectbox(self, *a, **k):
            return 0

        def line_chart(self, *a, **k):
            return None

        def download_button(self, *a, **k):
            return None

        def warning(self, *a, **k):
            return None

        def info(self, *a, **k):
            return None

        def caption(self, *a, **k):
            return None

        def divider(self, *a, **k):
            return None

        def title(self, *a, **k):
            return None

    st = STub()

    class Sidebar:
        def __enter__(self):
            return st

        def __exit__(self, *exc):
            return False

    st.sidebar = Sidebar()

    class Components:
        class v1:
            @staticmethod
            def html(*a, **k):
                return None

    st.components = Components()

    monkeypatch.setitem(sys.modules, "streamlit", st)

    # stub theme/state utilities
    monkeypatch.setitem(sys.modules, "ui._theme", SimpleNamespace(inject_theme=lambda: None))
    monkeypatch.setitem(sys.modules, "ui._state", SimpleNamespace(DUCK_PATH="duck", init_defaults=lambda: None))

    # stub mlflow and client
    called = {}

    def fake_read_csv(path, *a, **k):
        called["equity"] = Path(path)
        return pd.DataFrame({"equity": [1, 2]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    def fake_read_text(self, *a, **k):
        called.setdefault("html", []).append(self)
        return "<html></html>"

    def fake_read_bytes(self, *a, **k):
        return b"x"

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    class DummyClient:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, experiment_ids, max_results, order_by):
            art_dir = tmp_path / "mlruns" / "1" / "run1" / "artifacts"
            art_dir.mkdir(parents=True, exist_ok=True)
            (art_dir / "equity.csv").write_text("eq\n1")
            (art_dir / "diagnostics.html").write_text("<html></html>")
            (art_dir / "playbook.html").write_text("<html></html>")
            run1 = SimpleNamespace(
                info=SimpleNamespace(
                    run_id="run1",
                    status="FINISHED",
                    start_time=0,
                    artifact_uri=art_dir.as_uri(),
                )
            )
            run2 = SimpleNamespace(
                info=SimpleNamespace(
                    run_id="run2",
                    status="FAILED",
                    start_time=0,
                    artifact_uri=art_dir.as_uri(),
                )
            )
            return [run1, run2]

    tracking_ns = SimpleNamespace(MlflowClient=DummyClient)
    mlflow_stub = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        tracking=tracking_ns,
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_ns)

    load_module("ui/streamlit_app.py", "streamlit_app")

    art_dir = tmp_path / "mlruns" / "1" / "run1" / "artifacts"
    assert called["equity"] == art_dir / "equity.csv"
    assert art_dir / "diagnostics.html" in called["html"]
    assert art_dir / "playbook.html" in called["html"]
