"""No live Ollama or PX HTTP calls: mocked requests and local catalogue fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from filter_resolution import (
    FilterResolutionError, _choice_summary, resolve_filters,
)


TABLE = {
    "id": "SAM02002.px", "title": "Farþegar með skemmtiferðaskipum",
    "category": "Atvinnuvegir", "metadata_status": "ok",
    "url": "https://px.hagstofa.is/pxis/api/v1/is/SAM02002.px",
    "metadata": {"variables": [
        {"code": "Eining", "text": "Eining", "values": ["0", "1"],
         "valueTexts": ["Skipakomur", "Farþegar alls"]},
        {"code": "Ár", "text": "Ár", "time": True,
         "values": ["2023", "2024"], "valueTexts": ["2023", "2024"]},
    ]},
}


class FilterResolutionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "catalog.json"
        self.path.write_text(json.dumps([TABLE]), encoding="utf-8")
        self.question = "Hversu margir farþegar komu með skemmtiferðaskipum árið 2024?"
        self.normalized = {"time": {"years": [2024], "months": [], "quarters": [], "relative": None}}

    def tearDown(self):
        self.folder.cleanup()

    def ask(self, selections, unresolved=None, normalized=None, max_cells=500):
        body = json.dumps({"selections": selections,
                           "unresolved": unresolved or [], "reason": "Metadata matches."})
        response = Mock()
        response.json.return_value = {"message": {"content": body}, "done_reason": "stop"}
        response.raise_for_status.return_value = None
        session = Mock()
        session.post.return_value = response
        result = resolve_filters(
            self.question, TABLE["url"], normalized=normalized or self.normalized,
            catalog_path=self.path, session=session, max_cells=max_cells,
        )
        return result, session

    def test_valid_codes_produce_prepared_one_cell_query(self):
        result, session = self.ask([
            {"dimension": "Eining", "values": ["1"]},
            {"dimension": "Ár", "values": ["2024"]},
        ])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["selections"], {"Eining": ["1"], "Ár": ["2024"]})
        self.assertEqual(result["cells_requested"], 1)
        self.assertEqual(result["resolved_selections"]["Eining"][0]["label"], "Farþegar alls")
        api_payload = session.post.call_args.kwargs["json"]
        self.assertFalse(api_payload["stream"])
        self.assertFalse(api_payload["think"])
        self.assertEqual(api_payload["model"], "qwen3:30b-instruct")

    def test_ambiguous_year_never_makes_query_ready(self):
        result, _ = self.ask([
            {"dimension": "Eining", "values": ["1"]},
        ], [{"dimension": "Ár", "reason": "Which year?"}])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["unresolved"][0]["dimension"], "Ár")

    def test_missing_dimension_cannot_silently_default(self):
        result, _ = self.ask([{"dimension": "Ár", "values": ["2024"]}])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["unresolved"][0]["dimension"], "Eining")

    def test_wrong_explicit_year_rejected(self):
        result, _ = self.ask([
            {"dimension": "Eining", "values": ["1"]},
            {"dimension": "Ár", "values": ["2023"]},
        ])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("requested years", result["unresolved"][0]["reason"])

    def test_undefined_year_cannot_be_guessed(self):
        result, _ = self.ask([
            {"dimension": "Eining", "values": ["1"]},
            {"dimension": "Ár", "values": ["2024"]},
        ], normalized={"time": {"years": [], "relative": None}})
        self.assertEqual(result["status"], "needs_clarification")

    def test_unknown_option_rejected(self):
        with self.assertRaisesRegex(FilterResolutionError, "invented/unlisted"):
            self.ask([
                {"dimension": "Eining", "values": ["999"]},
                {"dimension": "Ár", "values": ["2024"]},
            ])

    def test_unknown_dimension_rejected(self):
        with self.assertRaisesRegex(FilterResolutionError, "Invalid or duplicate"):
            self.ask([{"dimension": "Nonexistent", "values": ["1"]}])

    def test_duplicate_dimension_rejected(self):
        with self.assertRaisesRegex(FilterResolutionError, "Invalid or duplicate"):
            self.ask([
                {"dimension": "Eining", "values": ["1"]},
                {"dimension": "Eining", "values": ["0"]},
            ])

    def test_oversized_query_fails_closed(self):
        result, _ = self.ask([
            {"dimension": "Eining", "values": ["0", "1"]},
            {"dimension": "Ár", "values": ["2023", "2024"]},
        ], max_cells=2)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("limit 2", result["unresolved"][0]["reason"])

    def test_large_dimension_only_shows_metadata_matched_options(self):
        many = {"code": "Ár", "text": "Ár", "time": True,
                "values": [str(y) for y in range(1900, 2026)],
                "valueTexts": [str(y) for y in range(1900, 2026)]}
        options, truncated = _choice_summary(many, "Fjöldi árið 2024", ["2024"])
        self.assertTrue(truncated)
        self.assertEqual(options, [{"code": "2024", "label": "2024"}])

    def test_model_length_error_is_not_accepted(self):
        response = Mock()
        response.json.return_value = {"done_reason": "length", "message": {"content": "{}"}}
        session = Mock()
        session.post.return_value = response
        with self.assertRaisesRegex(FilterResolutionError, "generation limit"):
            resolve_filters(self.question, TABLE["url"], normalized=self.normalized,
                            catalog_path=self.path, session=session)

    def test_monthly_observation_is_not_misreported_as_yearly(self):
        data = json.loads(json.dumps(TABLE))
        data["metadata"]["variables"][1].update({
            "values": ["2024M01", "2024M02"],
            "valueTexts": ["2024M01", "2024M02"],
        })
        self.path.write_text(json.dumps([data]), encoding="utf-8")
        result, _ = self.ask([
            {"dimension": "Eining", "values": ["1"]},
            {"dimension": "Ár", "values": ["2024M01"]},
        ])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("subannual", result["unresolved"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
