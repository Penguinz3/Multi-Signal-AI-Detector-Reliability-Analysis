# Operational Beta Demo

This self-contained demo exercises the black-box workflow with synthetic text
and simulated detector scores. It demonstrates software behavior only; its
scores are not research evidence.

From the repository root, choose a new output location and run:

```powershell
python -m fprint init-audit `
  --records .\examples\operational-demo\records.csv `
  --audit-root .\outputs\product\operational-demo\audit `
  --endpoint demo-opaque-detector `
  --minimum-sites 2

python .\examples\operational-demo\simulate_scores.py `
  .\outputs\product\operational-demo\audit\challenge.csv `
  .\outputs\product\operational-demo\scores

python -m fprint import-run --audit-root .\outputs\product\operational-demo\audit `
  --run-id reference-a --role reference `
  --scores .\outputs\product\operational-demo\scores\reference-a.csv `
  --metadata .\outputs\product\operational-demo\scores\reference-a-metadata.json

python -m fprint import-run --audit-root .\outputs\product\operational-demo\audit `
  --run-id reference-b --role reference `
  --scores .\outputs\product\operational-demo\scores\reference-b.csv `
  --metadata .\outputs\product\operational-demo\scores\reference-b-metadata.json

python -m fprint import-run --audit-root .\outputs\product\operational-demo\audit `
  --run-id current-changed --role current `
  --scores .\outputs\product\operational-demo\scores\current-changed.csv `
  --metadata .\outputs\product\operational-demo\scores\current-changed-metadata.json

python -m fprint compare-runs --audit-root .\outputs\product\operational-demo\audit `
  --reference reference-a reference-b --current current-changed `
  --output-dir .\outputs\product\operational-demo\report
```

Open `report/index.html`. The simulated current run changes paragraph-response
geometry, so the report should say `CHANGED` and localize the affected behavior
to paragraph resegmentation.
