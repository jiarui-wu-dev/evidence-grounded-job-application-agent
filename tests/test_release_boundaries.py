import contextlib
import io
import unittest
from types import SimpleNamespace

from src.analysis.application_strategy import generate_strategy_with_retry
from src.rendering.latex_renderer import extract_cv


class TestReleaseBoundaries(unittest.TestCase):
    def test_wrapped_review_run_cannot_be_rendered(self):
        cv = {"profile": {"text": "Example CV"}}
        self.assertEqual(extract_cv({"final_cv": cv, "final_status": "grounded"}), cv)
        with self.assertRaisesRegex(ValueError, "grounded"):
            extract_cv({"final_cv": cv, "final_status": "needs_human_review"})

    def test_strategy_retry_does_not_echo_private_model_output(self):
        secret = "PRIVATE_RESPONSE_MARKER"
        model = SimpleNamespace(invoke=lambda _: SimpleNamespace(content=secret))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(ValueError) as failure:
            generate_strategy_with_retry(model, {}, max_attempts=1)
        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn(secret, str(failure.exception))
