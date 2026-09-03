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
from .operational import compare_runs, export_challenge, import_run, initialize_audit
from .product import build_release_bundle, export_contracts, write_evaluation_html
from .validation import lock_prospective_panel, prepare_prospective_validation, replay_operational_validation
from .validation_evaluate import evaluate_prospective_validation
from .validation_scoring import (
    lock_execution_integrity_patch,
    lock_scoring_integrity_amendment,
    lock_scoring_protocol_from_database,
    score_validation_run,
)


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


def initialize_operational(args: argparse.Namespace) -> None:
    result = initialize_audit(
        args.records,
        args.audit_root,
        args.endpoint,
        minimum_triplets_per_probe=args.minimum_triplets,
        minimum_sites=args.minimum_sites,
        alpha=args.alpha,
        absolute_tolerance=args.absolute_tolerance,
        noise_multiplier=args.noise_multiplier,
        minimum_affected_fraction=args.minimum_affected_fraction,
        maximum_reference_noise=args.maximum_reference_noise,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def export_operational(args: argparse.Namespace) -> None:
    print(f"Exported locked challenge: {export_challenge(args.audit_root, args.output_dir)}")


def import_operational_run(args: argparse.Namespace) -> None:
    print(f"Locked detector run: {import_run(args.audit_root, args.run_id, args.role, args.scores, args.metadata)}")


def compare_operational(args: argparse.Namespace) -> None:
    print(f"Published operational report: {compare_runs(args.audit_root, args.reference, args.current, args.output_dir)}")


def replay_operational(args: argparse.Namespace) -> None:
    print(f"Published retrospective operational replay: {replay_operational_validation(args.audit_root, args.output_dir)}")


def prepare_prospective(args: argparse.Namespace) -> None:
    print(f"Locked prospective validation candidates: {prepare_prospective_validation(args.source_root, args.prior_fault_root, args.validation_root)}")


def lock_panel(args: argparse.Namespace) -> None:
    print(f"Locked all-endpoint-valid prospective panel: {lock_prospective_panel(args.validation_root, args.mage_repo)}")


def lock_scoring(args: argparse.Namespace) -> None:
    print(f"Locked prospective scoring protocol: {lock_scoring_protocol_from_database(args.validation_root, args.reference_database)}")


def score_prospective(args: argparse.Namespace) -> None:
    print(f"Locked prospective score run: {score_validation_run(args.validation_root, args.endpoint, args.condition_code, args.run_label, device=args.device, mage_repo=args.mage_repo)}")


def amend_scoring_integrity(args: argparse.Namespace) -> None:
    print(f"Locked pre-score integrity amendment: {lock_scoring_integrity_amendment(args.validation_root)}")


def patch_scoring_execution(args: argparse.Namespace) -> None:
    print(f"Locked score-preserving execution patch: {lock_execution_integrity_patch(args.validation_root)}")


def evaluate_prospective(args: argparse.Namespace) -> None:
    print(f"Published prospective validation: {evaluate_prospective_validation(args.validation_root, args.output_dir)}")


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

    initialize = sub.add_parser("init-audit", help="Lock an operational challenge from approved records")
    initialize.add_argument("--records", type=Path, required=True)
    initialize.add_argument("--audit-root", type=Path, required=True)
    initialize.add_argument("--endpoint", required=True)
    initialize.add_argument("--minimum-triplets", type=int, default=10)
    initialize.add_argument("--minimum-sites", type=int, default=4)
    initialize.add_argument("--alpha", type=float, default=.05)
    initialize.add_argument("--absolute-tolerance", type=float, default=.01)
    initialize.add_argument("--noise-multiplier", type=float, default=3.0)
    initialize.add_argument("--minimum-affected-fraction", type=float, default=.20)
    initialize.add_argument("--maximum-reference-noise", type=float, default=.02)
    initialize.set_defaults(func=initialize_operational)

    challenge = sub.add_parser("export-challenge", help="Export the locked query table and score template")
    challenge.add_argument("--audit-root", type=Path, required=True)
    challenge.add_argument("--output-dir", type=Path, required=True)
    challenge.set_defaults(func=export_operational)

    run = sub.add_parser("import-run", help="Validate and lock one complete detector score run")
    run.add_argument("--audit-root", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--role", choices=("reference", "current"), required=True)
    run.add_argument("--scores", type=Path, required=True)
    run.add_argument("--metadata", type=Path, required=True)
    run.set_defaults(func=import_operational_run)

    compare = sub.add_parser("compare-runs", help="Compare a current run with two locked reference repeats")
    compare.add_argument("--audit-root", type=Path, required=True)
    compare.add_argument("--reference", nargs=2, required=True, metavar=("REFERENCE_A", "REFERENCE_B"))
    compare.add_argument("--current", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(func=compare_operational)

    replay = sub.add_parser(
        "replay-operational-validation",
        help="Benchmark the operational alarm rule on prior locked fault scores",
    )
    replay.add_argument("--audit-root", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.set_defaults(func=replay_operational)

    prospective = sub.add_parser(
        "prepare-operational-validation",
        help="Lock fresh group-disjoint candidates and opaque validation conditions",
    )
    prospective.add_argument("--source-root", type=Path, required=True)
    prospective.add_argument("--prior-fault-root", type=Path, required=True)
    prospective.add_argument("--validation-root", type=Path, required=True)
    prospective.set_defaults(func=prepare_prospective)

    panel = sub.add_parser(
        "lock-operational-validation-panel",
        help="Token-check candidates and lock the prospective scoring panel",
    )
    panel.add_argument("--validation-root", type=Path, required=True)
    panel.add_argument("--mage-repo", type=Path, required=True)
    panel.set_defaults(func=lock_panel)

    scoring_lock = sub.add_parser(
        "lock-operational-validation-scoring",
        help="Bind scoring code and frozen normalization before inference",
    )
    scoring_lock.add_argument("--validation-root", type=Path, required=True)
    scoring_lock.add_argument("--reference-database", type=Path, required=True)
    scoring_lock.set_defaults(func=lock_scoring)

    scoring_amendment = sub.add_parser(
        "lock-operational-validation-integrity-amendment",
        help="Bind legacy panel bytes before prospective inference",
    )
    scoring_amendment.add_argument("--validation-root", type=Path, required=True)
    scoring_amendment.set_defaults(func=amend_scoring_integrity)

    execution_patch = sub.add_parser(
        "lock-operational-validation-execution-patch",
        help="Bind score-preserving collection guard fixes before unblinding",
    )
    execution_patch.add_argument("--validation-root", type=Path, required=True)
    execution_patch.set_defaults(func=patch_scoring_execution)

    validation_score = sub.add_parser(
        "score-operational-validation",
        help="Resume one opaque prospective endpoint condition",
    )
    validation_score.add_argument("--validation-root", type=Path, required=True)
    validation_score.add_argument("--endpoint", required=True)
    validation_score.add_argument("--condition-code", required=True)
    validation_score.add_argument("--run-label", choices=("reference-a", "reference-b", "current"), required=True)
    validation_score.add_argument("--device", type=int, default=0)
    validation_score.add_argument("--mage-repo", type=Path)
    validation_score.set_defaults(func=score_prospective)

    validation_evaluate = sub.add_parser(
        "evaluate-operational-validation",
        help="Lock blinded reports before unblinding prospective metrics",
    )
    validation_evaluate.add_argument("--validation-root", type=Path, required=True)
    validation_evaluate.add_argument("--output-dir", type=Path)
    validation_evaluate.set_defaults(func=evaluate_prospective)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
