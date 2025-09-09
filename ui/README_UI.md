# BettingEdge UI

Launch the dashboard with:

```bash
streamlit run ui/streamlit_app.py
```

The sidebar provides navigation across the pages:

1. **Data Load** – ingest raw CSVs and view summary statistics.
2. **EDA & Features** – quick exploratory plots and feature preview.
3. **Model Lab** – experiment with models.
4. **Backtest** – run toy backtests and see diagnostics.
5. **Edge Finder** – list potential bets and send them to the sizing page.
6. **Bet Sizing Simulator** – simulate stake policies.
7. **Explain** – placeholder for explainability tools.

The main page also exposes a **Results / MLflow Artifacts** tab showing outputs of
MLflow runs for the `backtest` experiment.

The UI resolves a single `DATA_ROOT` at start from `DATA_ROOT` env var or
`st.secrets`; if unset it defaults to `./data`. The value is stored in
`st.session_state["DATA_ROOT"]` and all pages communicate paths via
`st.session_state["raw_paths"]` and `st.session_state["processed_paths"]`.
This avoids mismatches when moving from **Data Load** to **Rebuild Market Probs**
to **Backtest**.

Before any interaction with MLflow the `MLFLOW_TRACKING_URI` environment
variable is set (env → `st.secrets` → `file:/var/tmp/mlruns`). The Backtest page
reads metrics and artifacts for the current run through the MLflow API; results
such as `equity.csv`, `diagnostics.html` and `playbook.html` are embedded
directly from the run's artifact store.

## UI → Engine wiring

| UI Action | Engine function | Artifacts shown |
| --- | --- | --- |
| **Ingest data/raw** | `engine.data.ingest.ingest` | DuckDB tables (`matches`, `odds_1x2_pre`, ...) |
| **Rebuild Market Probs** | `engine.data.ingest.compute_market_probs` / `engine.data.ingest.save_tables` | `market_probs_pre`, `market_probs_close` tables |
| **Build Features** | `engine.features.builder.build_features` | `features` table |
| **Fit DC (state-space)** | `engine.models.dc_state_space.DCStateSpace` | predictions, state snapshot |
| **Fit Bivariate Poisson (rolling)** | `engine.models.bivar_poisson_corr.BivarPoisson` | predicted probabilities |
| **Build Ensemble** | `engine.eval.feature_assembly.assemble_ensemble_features` / `engine.models.meta_learner.MetaEnsemble` | blended probabilities, reliability plot |
| **Run backtest** | `engine.eval.backtester.apply_conformal_guard` (+ `engine.eval.diagnostics.run_diagnostics`) | `equity.csv`, `diagnostics.html` |
| **Compute diagnostics** | `engine.eval.diagnostics.run_diagnostics` | `diagnostics.html` |

## MLflow Artifacts Viewer

- The viewer reads the `MLFLOW_TRACKING_URI` from the environment or `st.secrets`.
- If unset, it falls back to `file:/var/tmp/mlruns` (only local `file:` stores are supported).
- Select a run to display `equity.csv`, `diagnostics.html` and `playbook.html` and download them.
