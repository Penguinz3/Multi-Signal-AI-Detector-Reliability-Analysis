import json
import tempfile
import unittest
from pathlib import Path

from fprint.deferral import (
    DEV_CORPORA,
    ENDPOINT_ROLES,
    LOGRANK_ENDPOINT,
    MAGE_ENDPOINT,
    PROBES,
    RADAR_ENDPOINT,
    DeferralPaths,
    ScoreRow,
    authorize_final_stage,
    assemble_evaluation_rows,
    build_conditional_worklist,
    build_reflow_variants,
    calibrate_radar_threshold,
    export_manual_audit,
    import_generation_outputs,
    import_canonical_scores,
    import_manual_audit,
    line_wrap_variant,
    lock_human_token_panels,
    mage_effective_input_hash,
    prepare_generation_requests,
    prepare_pilot_manifest,
    query_accounting,
    radar_positive,
    require_pilot_authorization,
    select_human_panel,
    sentence_blocks_variant,
    sentence_per_paragraph_variant,
    validate_reflow_variant,
    verify_pilot_lock,
)
from fprint.core import lock_forecasts


def _records():
    rows = []
    for index in range(2):
        rows.append({
            "record_id": f"cal-{index}", "corpus": "corpus-a", "group_id": f"cal-group-{index}",
            "text": "A calibration passage with three sentences. It has enough words to wrap. The final sentence closes it.",
            "provenance_label": "human", "partition": "calibration",
        })
    for index in range(2):
        rows.append({
            "record_id": f"pilot-{index}", "corpus": "corpus-a", "group_id": f"pilot-group-{index}",
            "text": "A pilot passage with three sentences. It has enough words to wrap at a small width. The final sentence closes the passage.",
            "provenance_label": "human", "partition": "pilot",
        })
    return rows


class DeferralProbeTests(unittest.TestCase):
    def test_active_reflows_are_deterministic_and_preserve_non_whitespace(self):
        text = "One sentence, with punctuation. A second sentence follows here. A third sentence ends here."
        self.assertEqual(line_wrap_variant(text, width=20), line_wrap_variant(text, width=20))
        self.assertEqual(sentence_blocks_variant(text, block_size=2), sentence_blocks_variant(text, block_size=2))
        self.assertEqual(sentence_per_paragraph_variant(text), sentence_per_paragraph_variant(text))
        variants = build_reflow_variants(text, width=20, block_size=2)
        self.assertEqual({row.probe for row in variants}, set(PROBES))
        self.assertTrue(all(row.changed and row.non_whitespace_preserved and row.eligible for row in variants))
        for row in variants:
            self.assertEqual(validate_reflow_variant(text, row.text)["eligible"], True)

    def test_eligibility_rejects_a_noop(self):
        with self.assertRaises(ValueError):
            validate_reflow_variant("One sentence.", "One sentence.")


class DeferralManifestTests(unittest.TestCase):
    def test_manifest_is_group_disjoint_and_lock_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = DeferralPaths.from_root(Path(temporary))
            manifest = prepare_pilot_manifest(
                _records(), paths, calibration_cap=2, pilot_cap=2,
                width=20, block_size=2,
                endpoint_revisions={endpoint: "rev-1" for endpoint in ENDPOINT_ROLES},
            )
            self.assertTrue(paths.lock.exists())
            self.assertEqual(verify_pilot_lock(paths)["payload"], manifest)
            calibration_groups = {row["group_id"] for row in manifest["calibration"]}
            pilot_groups = {row["group_id"] for row in manifest["pilot"]}
            self.assertFalse(calibration_groups & pilot_groups)
            self.assertEqual(manifest["candidate_order"], "corpus,group_id,record_id")
            copy = json.loads(paths.manifest.read_text(encoding="utf-8"))
            copy["seed"] += 1
            paths.manifest.write_text(json.dumps(copy), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_pilot_lock(paths)

    def test_default_selection_is_exact_and_balanced(self):
        records = []
        for corpus in DEV_CORPORA:
            for index in range(1_750):
                records.append({
                    "record_id": f"{corpus}-{index}", "corpus": corpus,
                    "group_id": f"{corpus}-group-{index}",
                    "text": f"A human passage {corpus} {index}.", "provenance_label": "human",
                })
        from fprint.deferral import read_canonical_table
        calibration, pilot = select_human_panel(read_canonical_table(records))
        self.assertEqual(len(calibration), 2_000)
        self.assertEqual(len(pilot), 5_000)
        self.assertEqual({corpus: sum(row.corpus == corpus for row in calibration) for corpus in DEV_CORPORA}, {corpus: 500 for corpus in DEV_CORPORA})
        self.assertTrue(all(sum(row.corpus == corpus for row in pilot) >= 1_000 for corpus in DEV_CORPORA))
        self.assertFalse({(row.corpus, row.group_id) for row in calibration} & {(row.corpus, row.group_id) for row in pilot})


class DeferralWorklistTests(unittest.TestCase):
    def _manifest(self, temporary):
        paths = DeferralPaths.from_root(Path(temporary))
        manifest = prepare_pilot_manifest(
            _records(), paths, calibration_cap=2, pilot_cap=2,
            width=20, block_size=2,
            endpoint_revisions={endpoint: "rev-1" for endpoint in ENDPOINT_ROLES},
        )
        return paths, manifest

    def test_worklist_only_expands_original_positives_and_accounts_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, manifest = self._manifest(temporary)
            endpoints = tuple(ENDPOINT_ROLES)
            scores = {
                (record_id, RADAR_ENDPOINT): 0.0
                for record_id in ("pilot-0", "pilot-1")
            }
            scores[("pilot-0", RADAR_ENDPOINT)] = 0.9
            lock_forecasts(paths.threshold_lock, {"threshold": .5})
            worklist = build_conditional_worklist(
                paths, scores, thresholds={endpoint: 0.5 for endpoint in endpoints},
            )
            originals = [row for row in worklist if row["variant_id"] == "original"]
            variants = [row for row in worklist if row["variant_id"] != "original"]
            self.assertEqual(len(originals), 4)
            self.assertEqual(len(variants), 3)
            self.assertTrue(all(row["endpoint"] == RADAR_ENDPOINT for row in variants))
            self.assertEqual(query_accounting(2, 1), 5)
            self.assertEqual(sum(row["endpoint"] == RADAR_ENDPOINT for row in worklist), 5)
            self.assertEqual(sum(row["endpoint"] == MAGE_ENDPOINT for row in worklist), 1)
            self.assertEqual(sum(row["endpoint"] == LOGRANK_ENDPOINT for row in worklist), 1)
            self.assertNotIn(("pilot-1", "line_wrap", RADAR_ENDPOINT), {
                (row["record_id"], row["variant_id"], row["endpoint"]) for row in worklist
            })

    def test_final_stage_refuses_without_authorization_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = self._manifest(temporary)
            with self.assertRaises(RuntimeError):
                require_pilot_authorization(paths)
            authorize_final_stage(paths, {"status": "pilot_passed", "passed": True})
            self.assertEqual(require_pilot_authorization(paths)["payload"]["stage"], "pilot_authorization")

    def test_authorization_rejects_unpassed_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = self._manifest(temporary)
            with self.assertRaises(ValueError):
                authorize_final_stage(paths, {"status": "pilot_passed", "passed": False})


class DeferralGenerationTests(unittest.TestCase):
    def test_topic_may_contain_literal_braces(self):
        from fprint.deferral import _render_generation_prompt

        prompt = _render_generation_prompt(
            "Write about {topic} in {target_length} words ({min_word_count}-{max_word_count}).",
            "sets such as {1, 2, 3}", 25, 20, 30,
        )
        self.assertIn("{1, 2, 3}", prompt)

    def _manifest(self, temporary):
        paths = DeferralPaths.from_root(Path(temporary))
        prepare_pilot_manifest(
            _records(), paths, calibration_cap=2, pilot_cap=2,
            width=20, block_size=2,
            endpoint_revisions={endpoint: "rev-1" for endpoint in ENDPOINT_ROLES},
        )
        return paths

    def test_requests_are_locked_balanced_and_outputs_require_atomic_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._manifest(temporary)
            counts = {
                endpoint: {"original": 20, "wrap_80": 21, "sentence_blocks_2": 22, "sentence_per_paragraph": 23}
                for endpoint in (RADAR_ENDPOINT, MAGE_ENDPOINT)
            }
            lock_human_token_panels(paths, {record_id: counts for record_id in ("pilot-0", "pilot-1")})
            requests = prepare_generation_requests(
                paths, {"pilot-0": "topic zero", "pilot-1": "topic one"},
                generator_families=(("a", "ra"), ("b", "rb"), ("c", "rc")),
                seed=7, target_length=25,
            )
            self.assertEqual(len(requests), 2)
            self.assertTrue(paths.generation_lock.exists())
            output = "This is a generated passage with enough words to wrap around the fixed width. It has a second sentence. It has a third sentence."
            rows = [
                {
                    "request_id": request.request_id,
                    "generator_family": request.generator_family,
                    "generator_revision": request.generator_revision,
                    "retry": request.retry,
                    "text": output,
                    "token_counts": counts,
                }
                for request in requests
            ]
            with self.assertRaisesRegex(ValueError, "length tolerance"):
                import_generation_outputs(paths, [dict(row, text="Too short.") for row in rows])
            panels = import_generation_outputs(paths, rows)
            self.assertEqual(len(panels), 2)
            self.assertTrue(all(panel["output_word_count"] for panel in panels))
            lock_forecasts(paths.threshold_lock, {"threshold": .5})
            original_scores = {
                (record_id, RADAR_ENDPOINT): .9
                for record_id in (
                    "pilot-0", "pilot-1", panels[0]["ai_record_id"], panels[1]["ai_record_id"],
                )
            }
            work = build_conditional_worklist(
                paths, original_scores, thresholds={RADAR_ENDPOINT: .5},
            )
            self.assertEqual(sum(row["endpoint"] == RADAR_ENDPOINT and row["variant_id"] == "original" for row in work), 4)
            self.assertEqual(sum(row["endpoint"] == RADAR_ENDPOINT and row["variant_id"] in PROBES for row in work), 12)
            self.assertEqual(sum(row["endpoint"] == MAGE_ENDPOINT for row in work), 4)
            self.assertEqual(sum(row["endpoint"] == LOGRANK_ENDPOINT for row in work), 4)
            manifest = verify_pilot_lock(paths)["payload"]
            hashes = {
                row["record_id"]: {
                    "original": row["text_sha256"],
                    **{variant["variant_id"]: variant["text_sha256"] for variant in row["variants"]},
                }
                for row in manifest["pilot"]
            }
            for panel in panels:
                hashes[panel["ai_record_id"]] = {
                    "original": panel["base_text_sha256"],
                    **{probe: panel["variants"][probe]["text_sha256"] for probe in PROBES},
                }
            score_rows = []
            for record_id, variants in hashes.items():
                label = "ai" if record_id.startswith("ai:") else "human"
                for variant, text_hash in variants.items():
                    score_rows.append(ScoreRow(record_id, variant, RADAR_ENDPOINT, "rev-1", text_hash, label, .9 if variant == "original" else .7))
                score_rows.append(ScoreRow(record_id, "original", MAGE_ENDPOINT, "rev-1", variants["original"], label, .4))
                score_rows.append(ScoreRow(record_id, "original", LOGRANK_ENDPOINT, "rev-1", variants["original"], label, .3))
            import_canonical_scores(score_rows, paths)
            evaluation = assemble_evaluation_rows(paths, [
                {"record_id": "pilot-0", "text": _records()[2]["text"]},
                {"record_id": "pilot-1", "text": _records()[3]["text"]},
                *({"request_id": request.request_id, "text": output} for request in requests),
            ])
            self.assertEqual(len(evaluation), 4)
            self.assertEqual({row["label"] for row in evaluation}, {0, 1})
            with self.assertRaises(ValueError):
                import_generation_outputs(paths, [dict(rows[0], token_counts={RADAR_ENDPOINT: {"original": 20}}), rows[1]])

    def test_mage_effective_input_hash_is_invariant_to_reflow(self):
        text = "A sentence with enough words to wrap at the fixed width. A second sentence follows. A third sentence closes the passage."
        variants = build_reflow_variants(text)
        self.assertEqual(
            len({mage_effective_input_hash(text), *(mage_effective_input_hash(row.text) for row in variants)}), 1,
        )

    def test_manual_audit_export_import_is_exact_and_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._manifest(temporary)
            rows = export_manual_audit(
                paths, probe="wrap_80", count=1,
                texts=[{"record_id": row["record_id"], "text": row["text"]} for row in _records()[2:]],
            )
            self.assertEqual(len(rows), 1)
            self.assertIn("variant_text", rows[0])
            judged = [dict(rows[0], valid="1")]
            result = import_manual_audit(paths, judged, probe="wrap_80", count=1, minimum_valid=1)
            self.assertEqual(result["valid"], 1)


class DeferralThresholdTests(unittest.TestCase):
    def test_threshold_is_strictly_greater_and_requires_complete_2000(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = DeferralPaths.from_root(Path(temporary))
            rows = []
            for index in range(2_000):
                rows.append({
                    "record_id": f"cal-{index}", "corpus": DEV_CORPORA[index % 4],
                    "group_id": f"cal-group-{index}",
                    "text": f"Calibration text {index}.", "provenance_label": "human", "partition": "calibration",
                })
            rows.append({
                "record_id": "pilot", "corpus": DEV_CORPORA[0], "group_id": "pilot-group",
                "text": "A pilot sentence. A second sentence. A third sentence.",
                "provenance_label": "human", "partition": "pilot",
            })
            prepare_pilot_manifest(rows, paths, calibration_cap=2_000, pilot_cap=1, width=10, block_size=2)
            scores = {f"cal-{index}": float(index // 100) for index in range(2_000)}
            payload = calibrate_radar_threshold(paths, scores)
            self.assertEqual(payload["reference_count"], 2_000)
            self.assertFalse(radar_positive(payload["threshold"], payload["threshold"]))


class DeferralScoreRowTests(unittest.TestCase):
    def test_score_row_carries_provenance_revision_and_text_hash_fields(self):
        row = ScoreRow("r", "original", RADAR_ENDPOINT, "rev", "hash", "human", .2)
        self.assertEqual(row.key, ("r", "original", RADAR_ENDPOINT))
        self.assertEqual(row.provenance_label, "human")


if __name__ == "__main__":
    unittest.main()
