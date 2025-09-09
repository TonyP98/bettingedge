"""Streamlit backtest page with diagnostics tab."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import traceback

import pandas as pd
import streamlit as st
from omegaconf import OmegaConf
from mlflow.tracking import MlflowClient

try:  # pragma: no cover - optional
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

from engine.eval import backtester
from engine.eval.diagnostics import run_diagnostics
from engine.utils import mlflow_utils as mlf
from ui._state import set
from ui._widgets import metric_card

# -----------------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("mlflow_run_id", None)
ss.setdefault("artifacts_root", None)
ss.setdefault("show_artifacts", False)

client = MlflowClient()


def _resolve_last_run() -> None:
    """Populate session state with the latest backtest run if missing."""
    if ss.get("mlflow_run_id"):
        return
    try:
        exp = client.get_experiment_by_name("backtest")
    except Exception:
        exp = None
    if not exp:
        return
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return
    run = runs[0]
    ss["mlflow_run_id"] = run.info.run_id
    set("last_run_id", run.info.run_id)
    parsed = urlparse(run.info.artifact_uri)
    if parsed.scheme == "file":
        ss["artifacts_root"] = Path(parsed.path)
    else:  # pragma: no cover - handled visually
        st.warning("artifact store non locale: preview e download disabilitati")
        ss["artifacts_root"] = None


_resolve_last_run()
run_id = ss.get("mlflow_run_id") or "no_run"
art_root = ss.get("artifacts_root")

st.title("📈 Backtest")

backtest_tab, diag_tab = st.tabs(["Backtest", "Diagnostics"])

# -----------------------------------------------------------------------------
# Backtest tab
# -----------------------------------------------------------------------------
with backtest_tab:
    enable = st.checkbox("Enable Conformal Guard", value=False)
    if enable:
        alpha = st.slider("Alpha", 0.01, 0.5, 0.1)
        mondrian = st.multiselect(
            "Mondrian keys",
            ["Div", "Season", "Month", "OverroundBin", "RegimeId"],
            default=["Div"],
        )
        window = st.number_input(
            "Calibration window", min_value=1, max_value=50, value=10
        )
        edge_thr = st.number_input("Edge threshold", 0.0, 1.0, 0.02)
        kelly_cap = st.number_input("Kelly cap", 0.0, 1.0, 0.15)
        max_width = st.number_input("Max width", 0.0, 1.0, 0.35)
        st.write(
            f"Conformal guard enabled with α={alpha}, keys={mondrian}, "
            f"window={window}, edge≥{edge_thr}"
        )
    else:
        st.write("Standard Kelly strategy without conformal guard.")

    if st.button("Run backtest"):
        try:
            cfg = OmegaConf.create(
                {
                    "conformal": {
                        "alpha": alpha if enable else 0.1,
                        "mondrian_keys": mondrian if enable else [],
                        "min_group_size": int(window) if enable else 10,
                        "width_cap": max_width if enable else 0.35,
                    },
                    "policy": {
                        "edge_thr": edge_thr if enable else 0.02,
                        "max_width": max_width if enable else 0.35,
                        "kelly_cap": kelly_cap if enable else 0.15,
                        "variance_penalty": 0.0,
                    },
                }
            )
            df = pd.DataFrame(
                {
                    "pH": [0.4, 0.35],
                    "pD": [0.3, 0.3],
                    "pA": [0.3, 0.35],
                    "oH": [2.1, 2.2],
                    "oD": [3.1, 3.0],
                    "oA": [3.2, 3.4],
                    "y": [0, 1],
                    "pnl": [0.1, -0.05],
                }
            )
            mlf.start_run("backtest")
            out = backtester.apply_conformal_guard(df, cfg)
            eq_df = pd.DataFrame({"step": range(len(out)), "equity": out["pnl"].cumsum()})
            eq_path = Path("data/processed/equity.csv")
            eq_path.parent.mkdir(parents=True, exist_ok=True)
            eq_df.to_csv(eq_path, index=False)
            mlf.log_artifact(str(eq_path))
            diag_path = run_diagnostics()
            if isinstance(diag_path, str) and Path(diag_path).exists():
                mlf.log_artifact(diag_path)
            if mlflow and mlflow.active_run():  # active run available
                run = mlflow.active_run()
                ss["mlflow_run_id"] = run.info.run_id
                set("last_run_id", run.info.run_id)
                parsed = urlparse(run.info.artifact_uri)
                if parsed.scheme == "file":
                    ss["artifacts_root"] = Path(parsed.path)
                else:
                    st.info(
                        "artifact store non locale: download/preview disabilitati"
                    )
                    ss["artifacts_root"] = None
                st.success(f"MLflow run {run.info.run_id}")
            mlf.end_run()
            st.rerun()
        except Exception as exc:  # pragma: no cover - display errors
            st.error(f"Backtest failed: {exc}")
            st.code(traceback.format_exc())

    art_root = ss.get("artifacts_root")
    if art_root:
        eq_file = Path(art_root) / "equity.csv"
        if eq_file.exists():
            try:
                df = pd.read_csv(eq_file)
                col = (
                    "equity"
                    if "equity" in df.columns
                    else df.select_dtypes("number").columns.to_list()[0]
                    if not df.select_dtypes("number").empty
                    else None
                )
                if col:
                    st.line_chart(df[col])
                else:
                    st.write(df)
                st.download_button(
                    "Scarica equity.csv",
                    data=eq_file.read_bytes(),
                    file_name="equity.csv",
                    mime="text/csv",
                    key=f"dl_equity_{run_id}",
                )
            except Exception as e:  # pragma: no cover - display errors
                st.error(f"Errore nel leggere equity.csv: {e}")
                st.code(traceback.format_exc())
        else:
            st.warning("equity.csv non trovato per questo run")
        if st.button("Apri cartella artifact"):
            ss["show_artifacts"] = True
        if ss.get("show_artifacts"):
            files = [p for p in Path(art_root).rglob("*") if p.is_file()]
            for idx, p in enumerate(files):
                label = (
                    f"{p.relative_to(art_root)} "
                    f"({p.stat().st_size} B, {datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
                )
                mime = (
                    "text/html"
                    if p.suffix == ".html"
                    else "text/csv" if p.suffix == ".csv" else None
                )
                st.download_button(
                    label,
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime=mime,
                    key=f"dl_art_{run_id}_{idx}_{p.name}",
                )

    st.header("Bandit (online)")
    algo = st.selectbox("Algoritmo", ["linucb", "thompson"], index=0)
    alpha_b = st.slider("Alpha", 0.1, 5.0, 1.0)
    epsilon_b = st.slider("Epsilon", 0.0, 0.5, 0.02)
    gamma = st.slider("Gamma CRRA", 1.0, 3.0, 2.0)
    decay = st.slider("Decay", 0.9, 1.0, 0.995)
    edge_grid = st.multiselect(
        "edge_thr grid",
        [0.01, 0.02, 0.03, 0.05, 0.08],
        default=[0.01, 0.02, 0.03, 0.05, 0.08],
    )
    kelly_grid = st.multiselect(
        "kelly_cap grid", [0.10, 0.15, 0.20, 0.25], default=[0.10, 0.15, 0.20, 0.25]
    )
    use_conf = st.checkbox("Usa Conformal Guard", value=True)

# -----------------------------------------------------------------------------
# Diagnostics tab
# -----------------------------------------------------------------------------
with diag_tab:
    st.header("Diagnostics")
    art_root = ss.get("artifacts_root")
    diag_file = Path(art_root) / "diagnostics.html" if art_root else None
    play_file = Path(art_root) / "playbook.html" if art_root else None
    if diag_file and diag_file.exists():
        try:
            st.components.v1.html(
                diag_file.read_text(), height=700, scrolling=True
            )
            st.download_button(
                "Scarica diagnostics.html",
                data=diag_file.read_bytes(),
                file_name="diagnostics.html",
                mime="text/html",
                key=f"dl_diag_{run_id}",
            )
        except Exception as e:  # pragma: no cover - display errors
            st.error(f"Errore nel leggere diagnostics.html: {e}")
            st.code(traceback.format_exc())
    else:
        st.warning("diagnostics.html non trovato")

    if play_file and play_file.exists():
        try:
            st.components.v1.html(play_file.read_text(), height=700, scrolling=True)
            st.download_button(
                "Scarica playbook.html",
                data=play_file.read_bytes(),
                file_name="playbook.html",
                mime="text/html",
                key=f"dl_play_{run_id}",
            )
        except Exception as e:  # pragma: no cover - display errors
            st.error(f"Errore nel leggere playbook.html: {e}")
            st.code(traceback.format_exc())

    if st.button("Compute diagnostics"):
        try:
            diag_path = run_diagnostics()
            if isinstance(diag_path, str) and Path(diag_path).exists():
                mlf.log_artifact(diag_path)
                st.rerun()
        except Exception as exc:  # pragma: no cover - display errors
            st.error(f"Diagnostics failed: {exc}")
            st.code(traceback.format_exc())

    if st.button("Open last report") and art_root:
        ss["show_artifacts"] = True

    metrics = {"ECE": "n/d", "Brier": "n/d", "KS-p": "n/d", "Regimes": "n/d"}
    if ss.get("mlflow_run_id"):
        name_map = {
            "ECE": ["ECE", "ece"],
            "Brier": ["Brier", "brier"],
            "KS-p": ["KS_p", "KS-p", "ks_p"],
            "Regimes": ["Regimes", "regimes"],
        }
        for disp, names in name_map.items():
            for mname in names:
                try:
                    hist = client.get_metric_history(ss["mlflow_run_id"], mname)
                except Exception:
                    hist = []
                if hist:
                    metrics[disp] = hist[-1].value
                    break

    metric_card(
        "ECE (ensemble)",
        metrics["ECE"],
        help=None if metrics["ECE"] != "n/d" else "metrica non esportata dal motore",
    )
    metric_card(
        "Brier",
        metrics["Brier"],
        help=None if metrics["Brier"] != "n/d" else "metrica non esportata dal motore",
    )
    metric_card(
        "KS-p (PIT)",
        metrics["KS-p"],
        help=None if metrics["KS-p"] != "n/d" else "metrica non esportata dal motore",
    )
    metric_card(
        "# Regimes",
        metrics["Regimes"],
        help=None if metrics["Regimes"] != "n/d" else "metrica non esportata dal motore",
    )

# -----------------------------------------------------------------------------
# Debug information
# -----------------------------------------------------------------------------
try:
    current_run = client.get_run(ss["mlflow_run_id"]) if ss.get("mlflow_run_id") else None
except Exception:
    current_run = None

st.subheader("Debug MLflow state")
st.write("experiment_id:", current_run.info.experiment_id if current_run else "n/d")
st.write("run_id:", ss.get("mlflow_run_id") or "n/d")
st.write("artifact_uri:", current_run.info.artifact_uri if current_run else "n/d")
if art_root:
    files = [str(p.relative_to(art_root)) for p in Path(art_root).rglob("*") if p.is_file()]
    st.write("artifacts:", files)
else:
    st.write("artifacts:", [])
