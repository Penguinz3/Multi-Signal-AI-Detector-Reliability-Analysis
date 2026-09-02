from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conformance import (
    evaluate_fault_audit,
    fault_audit_readiness,
    import_score_table,
    prepare_fault_audit,
    score_fault_audit,
)
from .detectors import SPECS
from .product import build_release_bundle, export_contracts, write_evaluation_html


def prepare_conformance(args: argparse.Namespace) -> None:
    manifest = prepare_fault_audit(args.source_root, args.audit_root, args.config, args.evaluation)
    print(
        f"Locked {len(manifest['triplet_ids'])} discovery triplets and "
        f"{len(manifest['confirmation_candidate_ids'])} prospective candidates at {args.audit_root.resolve()}"
    )


def score_conformance(args: argparse.Namespace) -> None:
    if args.import_score_table:
        print(f"Imported {import_score_table(args.audit_root, args.import_score_table)} canonical score rows.")
        return
    if not args.endpoint or not args.fault:
        raise ValueError("Provide --endpoint and --fault, or --import-score-table")
    counts = score_fault_audit(
        args.audit_root,
        args.endpoint,
        args.fault,
        device=args.device,
        mage_repo=args.mage_repo,
        source_kind=args.source_kind,
    )
    print("Fault-audit scoring: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


def evaluate_conformance(args: argparse.Namespace) -> None:
    report = evaluate_fault_audit(args.audit_root, args.output_dir)
    gate = report["success_gates"]
    print(f"Fault audit evaluated: claim={gate['permitted_primary_claim']}; artifacts={report['artifacts']}")


def conformance_status(args: argparse.Namespace) -> None:
    print(json.dumps(fault_audit_readiness(args.audit_root), indent=2, sort_keys=True))


def package_conformance(args: argparse.Namespace) -> None:
    readiness = fault_audit_readiness(args.audit_root)
    if not readiness["ready"]:
        raise ValueError(f"Fault audit is incomplete: {readiness['missing']}")
    evaluation = args.audit_root / "results" / "fault_audit_evaluation.json"
    print(f"Published FPRINT release bundle: {build_release_bundle(evaluation, args.output_dir)}")


def export_conformance_contracts(args: argparse.Namespace) -> None:
    print(f"Exported FPRINT production contracts: {export_contracts(args.output_dir)}")


def render_conformance_report(args: argparse.Namespace) -> None:
    print(f"Wrote privacy-preserving FPRINT report: {write_evaluation_html(args.evaluation, args.output)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fprint",
        description="Behavioral conformance testing for black-box AI-text detectors",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-fault-audit", help="Create and hash-lock a conformance challenge")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--audit-root", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=Path("fault_audit_config.json"))
    prepare.add_argument("--evaluation", type=Path, required=True)
    prepare.set_defaults(func=prepare_conformance)

    score = sub.add_parser("score-fault-audit", help="Resume scoring or import canonical scores")
    score.add_argument("--audit-root", type=Path, required=True)
    score.add_argument("--endpoint", choices=sorted(SPECS))
    score.add_argument("--fault")
    score.add_argument("--device", type=int, default=0)
    score.add_argument("--mage-repo")
    score.add_argument(
        "--source-kind",
        choices=("all", "discovery", "confirmation_candidate"),
        default="all",
    )
    score.add_argument("--import-score-table", type=Path)
    score.set_defaults(func=score_conformance)

    evaluate = sub.add_parser("evaluate-fault-audit", help="Run held-out-corpus conformance evaluation")
    evaluate.add_argument("--audit-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.set_defaults(func=evaluate_conformance)

    status = sub.add_parser("fault-audit-status", help="Verify the lock and scoring completeness")
    status.add_argument("--audit-root", type=Path, required=True)
    status.set_defaults(func=conformance_status)

    package = sub.add_parser("package-fault-audit", help="Publish a verified privacy-safe release bundle")
    package.add_argument("--audit-root", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.set_defaults(func=package_conformance)

    contracts = sub.add_parser("export-fault-audit-contracts", help="Export versioned interchange contracts")
    contracts.add_argument("--output-dir", type=Path, required=True)
    contracts.set_defaults(func=export_conformance_contracts)

    render = sub.add_parser("render-fault-audit-report", help="Render a validated standalone HTML report")
    render.add_argument("--evaluation", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.set_defaults(func=render_conformance_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
