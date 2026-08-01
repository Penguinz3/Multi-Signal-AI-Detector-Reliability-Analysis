from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import fprint.detectors as detectors


class DetectorTests(unittest.TestCase):
    def tearDown(self):
        detectors._TOKEN_SCORERS.clear()

    def test_specs_pin_orientations_implementations_and_precision_ids(self):
        self.assertEqual(detectors.SPECS["openai_roberta_base__gpt2_legacy"].ai_label, 0)
        radar = detectors.SPECS["radar_roberta_large__vicuna7b_training"]
        self.assertEqual(radar.ai_label, 0)
        self.assertEqual(radar.orientation_source, detectors.RADAR_ORIENTATION_SOURCE)
        mage = detectors.SPECS["mage_longformer__paper"]
        self.assertEqual(mage.ai_label, 0)
        self.assertEqual(mage.preprocessing_revision, detectors.MAGE_REPOSITORY_REVISION)
        observers = [
            detectors.SPECS[f"{method}__qwen2_5_0_5b_fp32"]
            for method in ("logrank", "lastde")
        ]
        self.assertEqual(len({
            (spec.model_id, spec.revision, spec.tokenizer_revision, spec.precision)
            for spec in observers
        }), 1)
        self.assertEqual(observers[0].precision, "fp32")
        self.assertFalse(any(spec.candidate for spec in observers))
        self.assertEqual(
            detectors.SPECS["lastde__qwen2_5_0_5b_fp32"].implementation_revision,
            detectors.LASTDE_IMPLEMENTATION_REVISION,
        )
        detectors.validate_specs()

    def test_orientation_pilot_empirically_guards_direction(self):
        config_id = "radar_roberta_large__vicuna7b_training"
        detectors.assert_orientation_pilot(config_id, [.8, .9, .7], [.1, .3, .2])
        with self.assertRaises(RuntimeError):
            detectors.assert_orientation_pilot(config_id, [.1, .3, .2], [.8, .9, .7])
        with self.assertRaises(ValueError):
            detectors.assert_orientation_pilot(config_id, [.8], [.2])

    def test_labeled_pilot_requires_50_per_class_variation_and_repeatability(self):
        human = [index / 1000 for index in range(50)]
        ai = [.5 + index / 1000 for index in range(50)]
        self.assertEqual(
            detectors.validate_labeled_pilot(
                "radar_roberta_large__vicuna7b_training",
                human, ai, human, ai,
            ),
            0,
        )
        with self.assertRaisesRegex(RuntimeError, "deterministic"):
            detectors.validate_labeled_pilot(
                "radar_roberta_large__vicuna7b_training",
                human, ai, human, [value + .01 for value in ai],
            )

    def test_token_count_preprocesses_without_truncation(self):
        calls = []

        def tokenizer(text, **kwargs):
            calls.append((text, kwargs))
            return {"input_ids": [1, 2, 3]}

        adapter = object.__new__(detectors.SequenceClassifierAdapter)
        adapter.tokenizer = tokenizer
        adapter.preprocessor = str.upper
        self.assertEqual(adapter.token_count("mixed"), 3)
        self.assertEqual(calls, [("MIXED", {"add_special_tokens": True, "truncation": False})])

    def test_nonfinite_values_fail_closed(self):
        finite_result = Mock()
        finite_result.all.return_value = False
        torch = types.SimpleNamespace(isfinite=Mock(return_value=finite_result))
        with self.assertRaises(FloatingPointError):
            detectors._require_finite(torch, object(), "Observer logits")

    def test_logrank_and_lastde_share_one_precision_specific_scorer(self):
        created = []

        class FakeScorer:
            def __init__(self, spec):
                created.append(spec.config_id)

        with patch.object(detectors, "CausalTokenScorer", FakeScorer):
            logrank = detectors.build_adapter("logrank__qwen2_5_0_5b_fp32")
            lastde = detectors.build_adapter("lastde__qwen2_5_0_5b_fp32")
        self.assertIs(logrank.scorer, lastde.scorer)
        self.assertEqual(len(created), 1)

    def test_observer_load_is_pinned_eager_fp32_and_deterministic(self):
        model_calls = []

        class FakeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(use_cache=True)

            def eval(self):
                return self

        class ModelFactory:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                model_calls.append((model_id, kwargs))
                return FakeModel()

        class TokenizerFactory:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        transformers = types.SimpleNamespace(
            AutoModelForCausalLM=ModelFactory,
            AutoTokenizer=TokenizerFactory,
        )
        torch = types.SimpleNamespace(
            float32="float32",
            use_deterministic_algorithms=Mock(),
            backends=types.SimpleNamespace(
                cudnn=types.SimpleNamespace(benchmark=True),
            ),
        )
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            observer = detectors.CausalTokenScorer(
                detectors.SPECS["logrank__qwen2_5_0_5b_fp32"],
            )
        self.assertFalse(observer.model.config.use_cache)
        self.assertEqual(model_calls[0][1]["attn_implementation"], "eager")
        self.assertEqual(model_calls[0][1]["torch_dtype"], "float32")
        torch.use_deterministic_algorithms.assert_called_once_with(True)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_mage_requires_exact_clean_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            deployment = Path(directory) / "deployment"
            deployment.mkdir()
            (deployment / "__init__.py").write_text("", encoding="utf-8")
            results = [
                types.SimpleNamespace(stdout=detectors.MAGE_REPOSITORY_REVISION),
                types.SimpleNamespace(stdout=""),
            ]
            preprocess = object()
            with patch.object(subprocess, "run", side_effect=results), patch.object(
                detectors.importlib, "import_module",
                return_value=types.SimpleNamespace(preprocess=preprocess),
            ):
                self.assertIs(detectors._mage_preprocessor(directory), preprocess)
                self.assertFalse(sys.dont_write_bytecode)
            with patch.object(
                subprocess, "run", side_effect=[
                    types.SimpleNamespace(stdout="0" * 40),
                    types.SimpleNamespace(stdout=""),
                ],
            ):
                with self.assertRaises(RuntimeError):
                    detectors._mage_preprocessor(directory)

    def test_lastde_uses_official_open_source_parameters(self):
        parameters = inspect.signature(detectors.lastde).parameters
        self.assertEqual(parameters["embed_size"].default, 3)
        self.assertEqual(parameters["tau_prime"].default, 5)
        self.assertEqual(
            detectors.LASTDE_IMPLEMENTATION_REVISION,
            "ead6939e0e9382f9ce5aa1b33b936ee6c4e0605d",
        )


if __name__ == "__main__":
    unittest.main()
