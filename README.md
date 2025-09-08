# bettingedge

## Verifica mappa chiavi

Per validare la spec delle chiavi di ingestione:

```
python -m engine.ingest.keys --check
python -m engine.ingest.keys --sample data/raw/l1_24_25.csv
# oppure
bettingedge-keys --check
```

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m pytest -q
```
