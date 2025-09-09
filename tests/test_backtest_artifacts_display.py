import importlib.util
import sys
from pathlib import Path
import pandas as pd


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backtest_page_reads_artifacts(monkeypatch, tmp_path):
    art_dir = tmp_path / "mlruns" / "0" / "r1" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "equity.csv").write_text("equity\n1")
    (art_dir / "diagnostics.html").write_text("<html></html>")

    called = {}

    def fake_read_csv(path, *a, **k):
        called["equity"] = Path(path)
        return pd.DataFrame({"equity": [1, 2]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    orig_read_text = Path.read_text
    orig_read_bytes = Path.read_bytes

    def fake_read_text(self, *a, **k):
        if self.suffix in {".html", ".csv"}:
            called.setdefault("html", []).append(self)
            return "<html></html>"
        return orig_read_text(self, *a, **k)

    def fake_read_bytes(self, *a, **k):
        if self.suffix in {".html", ".csv"}:
            return b"x"
        return orig_read_bytes(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    class Tab:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class STub:
        session_state = {
            "mlflow_run_id": "r1",
            "artifacts_root": art_dir,
            "show_artifacts": True,
        }
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

        def selectbox(self, *a, **k):
            return None

        def line_chart(self, *a, **k):
            return None

        def download_button(self, label, *a, **k):
            called.setdefault("download", []).append(label)
            return None

        class components:
            class v1:
                @staticmethod
                def html(*a, **k):
                    called.setdefault("html_embed", []).append(a[0])
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

    load_module("ui/pages/04_📈_Backtest.py", "backtest_page")

    assert called["equity"] == art_dir / "equity.csv"
    assert art_dir / "diagnostics.html" in called["html"]
    assert "Scarica equity.csv" in called["download"]
    assert "Scarica diagnostics.html" in called["download"]

