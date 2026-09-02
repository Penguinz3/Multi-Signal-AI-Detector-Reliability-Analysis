# Privacy-Safe Demonstration

This bundle uses synthetic aggregate metrics and contains no source passages or
real detector outputs. It demonstrates the versioned evaluation contract and
the standalone report shown to an operator.

Rebuild it from the repository root:

```powershell
python -m fprint render-fault-audit-report `
  --evaluation .\examples\fault-audit-demo\evaluation.json `
  --output .\examples\fault-audit-demo\index.html

python -m fprint export-fault-audit-contracts `
  --output-dir .\examples\fault-audit-demo\contracts
```

The numbers are illustrative and must not be cited as study findings. The
validated study results are documented in `docs/fault_audit_results.md`.
