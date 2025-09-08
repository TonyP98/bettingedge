import subprocess
import sys
import pathlib
from importlib.resources import files


def test_engine_ingest_keys_cli_runs():
    cmd = [sys.executable, "-m", "engine.ingest.keys", "--check"]
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    assert rc in (0, 2)  # 0 ok, 2 warning when sample missing


def test_engine_ingest_keys_cli_with_sample(tmp_path):
    sample = tmp_path / "mini.csv"
    sample.write_text(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA\n"
        "E0,01/08/24,15:00,A,B,1,0,H,2.1,3.4,3.2\n"
    )
    spec = files("engine.data.specs") / "football_data_keys.yaml"
    cmd = [
        sys.executable,
        "-m",
        "engine.ingest.keys",
        "--spec",
        str(spec),
        "--sample",
        str(sample),
        "--check",
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    assert rc in (0, 2)
