import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_deferral_generation import (
    FakeBackend,
    GenerationFailure,
    IMPORT_FIELDS,
    _fit_word_envelope,
    read_locked_requests,
    request_seed,
    run_generation,
    verify_model_artifacts,
)


def _request_csv(path: Path, rows):
    fields = ["request_id", "record_id", "corpus", "generator_family", "generator_revision", "prompt", "prompt_sha256", "seed", "retry", "target_length", "min_word_count", "max_word_count", "decoding"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RunDeferralGenerationTests(unittest.TestCase):
    def test_model_artifact_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "config.json"
            artifact.write_bytes(b"pinned")
            expected = __import__("hashlib").sha256(b"pinned").hexdigest()
            verify_model_artifacts(root, {"config.json": expected})
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                verify_model_artifacts(root, {"config.json": expected})

    def test_overlength_output_uses_nearest_complete_sentence_prefix(self):
        text = "One two three four. Five six seven eight. Nine ten eleven twelve."
        self.assertEqual(
            _fit_word_envelope(text, target_length=8, min_word_count=6, max_word_count=10),
            "One two three four. Five six seven eight.",
        )

    def test_incomplete_suffix_is_removed_inside_envelope(self):
        text = "One two three four. Five six seven eight. Nine ten unfinished"
        self.assertEqual(
            _fit_word_envelope(text, target_length=10, min_word_count=7, max_word_count=12),
            "One two three four. Five six seven eight.",
        )

    def test_seed_is_stable_and_attempt_specific(self):
        self.assertEqual(request_seed(5, "r", 0), request_seed(5, "r", 0))
        self.assertNotEqual(request_seed(5, "r", 0), request_seed(5, "r", 1))
        self.assertNotEqual(request_seed(5, "r", 0), request_seed(6, "r", 0))

    def test_fake_generation_writes_exact_import_shape_and_never_needs_human_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "generation_requests.csv"
            _request_csv(requests, [{
                "request_id": "r1", "record_id": "human-1", "corpus": "asap_aes",
                "generator_family": "fake-family", "generator_revision": "rev-1",
                "prompt": "Write about public transport.", "prompt_sha256": "x", "seed": 7,
                "retry": 0, "target_length": 5, "min_word_count": 5, "max_word_count": 5,
                "decoding": json.dumps({"temperature": 0.7}),
            }])
            seen = []
            backend = FakeBackend("fake-family", "rev-1")
            rows = run_generation(requests, root / "outputs.csv", root / "checkpoint.jsonl", backend_factory=lambda f, r: (seen.append((f, r)) or backend))
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), set(IMPORT_FIELDS))
            self.assertEqual(seen, [("fake-family", "rev-1")])
            self.assertNotIn("human-1", backend.calls[0][0])
            with (root / "outputs.csv").open(encoding="utf-8", newline="") as handle:
                imported = list(csv.DictReader(handle))
            self.assertEqual(imported[0]["request_id"], "r1")
            self.assertEqual(imported[0]["retry"], "0")

    def test_resume_skips_checkpointed_request_and_append_tail_is_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "generation_requests.csv"
            common = {
                "record_id": "human", "corpus": "asap_aes", "generator_family": "fake",
                "generator_revision": "rev", "prompt": "Write a passage.", "prompt_sha256": "x",
                "seed": 9, "retry": 0, "target_length": 3, "min_word_count": 3, "max_word_count": 3,
                "decoding": "{}",
            }
            _request_csv(requests, [{"request_id": "r1", **common}, {"request_id": "r2", **common}])
            checkpoint = root / "checkpoint.jsonl"
            first = {"request_id": "r1", "generator_family": "fake", "generator_revision": "rev", "retry": 0, "attempt": 0, "seed": request_seed(9, "r1", 0), "target_length": 3, "decoding": "{}", "text": "generated generated generated."}
            checkpoint.write_text(json.dumps(first) + "\n{" , encoding="utf-8")
            backend = FakeBackend("fake", "rev")
            rows = run_generation(requests, root / "outputs.csv", checkpoint, backend_factory=lambda f, r: backend)
            self.assertEqual({row["request_id"] for row in rows}, {"r1", "r2"})
            self.assertEqual(len(backend.calls), 1)

    def test_retry_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "generation_requests.csv"
            _request_csv(requests, [{
                "request_id": "r", "record_id": "human", "corpus": "asap_aes", "generator_family": "fake", "generator_revision": "rev", "prompt": "Write.", "prompt_sha256": "x", "seed": 1, "retry": 1, "target_length": 5, "min_word_count": 5, "max_word_count": 5, "decoding": "{}",
            }])
            class TooShort(FakeBackend):
                def generate(self, *args, **kwargs):
                    self.calls.append((args[0] if args else "", kwargs["seed"]))
                    return "short"
            backend = TooShort("fake", "rev")
            with self.assertRaises(GenerationFailure):
                run_generation(requests, root / "outputs.csv", root / "checkpoint.jsonl", backend_factory=lambda f, r: backend)
            self.assertEqual(len(backend.calls), 2)

    def test_token_panel_rejects_attempt_atomically_and_is_checkpointed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "generation_requests.csv"
            _request_csv(requests, [{
                "request_id": "r", "record_id": "human", "corpus": "asap_aes",
                "generator_family": "fake", "generator_revision": "rev",
                "prompt": "Write.", "prompt_sha256": "x", "seed": 1, "retry": 1,
                "target_length": 5, "min_word_count": 5, "max_word_count": 5,
                "decoding": "{}",
            }])
            calls = []

            def panel_counter(text):
                calls.append(text)
                if len(calls) == 1:
                    raise ValueError("one transformed view truncates")
                return {"radar": {"original": 5}, "mage": {"original": 5}}

            backend = FakeBackend("fake", "rev")
            rows = run_generation(
                requests, root / "outputs.csv", root / "checkpoint.jsonl",
                backend_factory=lambda f, r: backend, panel_counter=panel_counter,
            )
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(rows[0]["attempt"], 1)
            self.assertEqual(json.loads(rows[0]["token_counts"])["radar"]["original"], 5)


if __name__ == "__main__":
    unittest.main()
