from engine.data import ingest


def test_ingest_drops_all_nan_rows(tmp_path):
    csv_content = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA\n"
        "I1,01/01/2020,15:00,TeamA,TeamB,1,0,H,1.5,3.5,5.1\n"
        "I1,02/01/2020,18:00,TeamC,TeamD,2,2,D,2.0,3.0,4.2\n"
        + "," * 10 + "\n"
    )
    path = tmp_path / "sample.csv"
    path.write_text(csv_content)

    tables = ingest.ingest(str(path), commit=False)
    matches = tables["matches"]

    assert len(matches) == 2
    assert matches["HomeTeam"].isna().sum() == 0
