import pandas as pd
import pandera as pa
import pytest

from engine.data.contracts import (
    MatchesSchema,
    Odds1x2PreSchema,
    MarketProbsSchema,
    validate_or_raise,
)


@pytest.fixture
def valid_matches():
    return pd.DataFrame(
        {
            "Div": ["I1"],
            "event_time": [
                pd.Timestamp("2024-08-01 12:00", tz="Europe/Rome").strftime(
                    "%Y-%m-%dT%H:%M:%S%z"
                )
            ],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "FTHG": [1],
            "FTAG": [0],
            "FTR": ["H"],
        }
    )


@pytest.fixture
def invalid_matches(valid_matches):
    df = valid_matches.copy()
    df.loc[0, "FTHG"] = -1
    return df


def test_matches_valid(valid_matches):
    validate_or_raise(valid_matches, MatchesSchema, "matches")


def test_matches_invalid(invalid_matches):
    with pytest.raises(pa.errors.SchemaError):
        validate_or_raise(invalid_matches, MatchesSchema, "matches")


@pytest.fixture
def valid_odds():
    return pd.DataFrame(
        {"H": [2.0], "D": [3.0], "A": [4.0], "overround_pre": [1 / 2 + 1 / 3 + 1 / 4]}
    )


@pytest.fixture
def invalid_odds(valid_odds):
    df = valid_odds.copy()
    df.loc[0, "H"] = 0.5
    return df


def test_odds_valid(valid_odds):
    validate_or_raise(valid_odds, Odds1x2PreSchema, "odds")


def test_odds_invalid(invalid_odds):
    with pytest.raises(pa.errors.SchemaError):
        validate_or_raise(invalid_odds, Odds1x2PreSchema, "odds")


@pytest.fixture
def valid_probs():
    return pd.DataFrame({"pH": [0.5], "pD": [0.3], "pA": [0.2]})


@pytest.fixture
def invalid_probs(valid_probs):
    df = valid_probs.copy()
    df.loc[0, "pA"] = 0.9
    return df


def test_probs_valid(valid_probs):
    validate_or_raise(valid_probs, MarketProbsSchema, "probs")


def test_probs_invalid(invalid_probs):
    with pytest.raises(pa.errors.SchemaError):
        validate_or_raise(invalid_probs, MarketProbsSchema, "probs")


def test_ingest_valid():
    from engine.data.ingest import ingest

    tables = ingest("data/raw/l1_24_25.csv", commit=False)
    assert len(tables["matches"]) > 0


def test_ingest_invalid(tmp_path):
    src = pd.read_csv("data/raw/l1_24_25.csv")
    src.loc[0, "FTHG"] = -1
    bad = tmp_path / "bad.csv"
    src.to_csv(bad, index=False)
    from engine.data.ingest import ingest

    with pytest.raises(pa.errors.SchemaError):
        ingest(str(bad), commit=False)
