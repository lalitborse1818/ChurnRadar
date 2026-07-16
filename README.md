# data/

- `sample_raw_customers.csv` / `sample_features.csv` — 200-row previews
  committed to the repo so the schema is browsable on GitHub without
  running anything.
- `raw_customers.csv`, `features.csv`, `churnradar.db` — the full
  52,000-row dataset, SQL feature table, and SQLite database. These are
  **generated, not committed** (see `.gitignore`) — run `python
  src/pipeline.py` from the repo root to produce them locally.
