# bettingedge

## Verifica mappa chiavi

Per validare la spec delle chiavi di ingestione:

```
python -m engine.ingest.keys --check
python -m engine.ingest.keys --sample data/raw/l1_24_25.csv
# oppure
bettingedge-keys --check
```
