import subprocess
from pathlib import Path
import duckdb


def test_directories_exist():
    assert Path("data/raw").exists()
    assert Path("data/processed").exists()


def test_ingest_creates_tables():
    cmd = ["python", "-m", "engine.data.ingest", "--source", "data/raw/l1_24_25.csv", "--commit"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    conn = duckdb.connect("data/processed/football.duckdb")
    tables = {t[0] for t in conn.execute("SHOW TABLES").fetchall()}
    assert {"matches", "odds_1x2_pre", "market_probs_pre"}.issubset(tables)
    assert "odds_1x2_close" in tables

    df = conn.execute(
        "SELECT pH_mul+pD_mul+pA_mul AS mul, pH_shin+pD_shin+pA_shin AS shin FROM market_probs_pre"
    ).fetchdf()
    assert ((df["mul"] - 1).abs() < 1e-6).all()
    assert ((df["shin"] - 1).abs() < 1e-6).all()
    conn.close()
