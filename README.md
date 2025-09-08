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

## Quality & Tracking

Per abilitare MLflow localmente:

```bash
export MLFLOW_TRACKING_URI=mlruns
python -m engine.eval.backtester
```

Gli artifact (es. `equity.csv`, `diagnostics.html`, `playbook.html`) vengono salvati nella cartella indicata da MLflow.

L'ingestione fallisce se i dati violano gli schemi Pandera; ad esempio, modificare una colonna con valori non validi provocherà un errore e l'operazione non verrà completata.
