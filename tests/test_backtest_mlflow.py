from pathlib import Path

import mlflow

from engine.utils import mlflow_utils as mlf


def test_mlflow_logging(tmp_path):
    tracking = tmp_path / "mlruns"
    mlf.start_run("test-exp", run_name="t1", tracking_uri=str(tracking))
    mlf.log_params({"p": 1})
    mlf.log_metrics({"m": 0.5})
    art = tmp_path / "artifact.txt"
    art.write_text("hi")
    mlf.log_artifact(str(art))
    run = mlflow.active_run()
    run_id = run.info.run_id
    exp_id = run.info.experiment_id
    mlf.end_run()
    base = tracking / exp_id / run_id
    assert (base / "params" / "p").exists()
    assert (base / "metrics" / "m").exists()
    assert (base / "artifacts" / "artifact.txt").exists()
