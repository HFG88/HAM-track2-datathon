"""Mocked, offline contract tests for llm_table_selection.py."""
import json
import unittest
from unittest.mock import Mock

import requests

from llm_table_selection import TableSelectionError, select_tables_llm


def table(id_, url=None):
    return {
        "id": id_, "url": url or f"https://example.test/{id_}",
        "title": "Farþegar með skemmtiferðaskipum 2024",
        "category": "Atvinnuvegir", "metadata": {"variables": [
            {"code": "Ár", "text": "Ár", "values": ["2023", "2024"],
             "valueTexts": ["2023", "2024"], "time": True},
            {"code": "Eining", "text": "Eining", "values": ["0", "1"],
             "valueTexts": ["Skipakomur", "Farþegar alls"]},
        ]},
    }


def fake_session(selected_numbers, *, reason="Good match", clarify=False):
    response = Mock()
    response.json.return_value = {"message": {"content": json.dumps({
        "selected_numbers": selected_numbers, "reason": reason,
        "needs_clarification": clarify,
    })}}
    response.raise_for_status.return_value = None
    session = Mock()
    session.post.return_value = response
    return session


class TableSelectionLLMTests(unittest.TestCase):
    def setUp(self):
        self.question = {"original": "Hversu margir farþegar árið 2024?",
                         "time": {"years": [2024]}, "breakdowns": [],
                         "measure": "fjoldi", "intent": "value"}
        self.first = table("SAM02002.px")
        self.second = table("SAM02003.px")

    def test_selects_exact_supplied_url(self):
        session = fake_session([2])
        result = select_tables_llm(self.question, [(self.first, 2.8), (self.second, 1.5)], session=session)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected"][0]["url"], self.second["url"])
        self.assertEqual(result["selected"][0]["retrieval_rank"], 2)
        self.assertTrue(result["requires_filter_resolution"])

    def test_uses_json_schema_and_local_model(self):
        session = fake_session([1])
        select_tables_llm(self.question, [(self.first, 2.8)], session=session)
        endpoint = session.post.call_args.args[0]
        options = session.post.call_args.kwargs["json"]
        self.assertEqual(endpoint, "http://localhost:11434/api/chat")
        self.assertEqual(options["model"], "qwen3:4b-instruct")
        self.assertFalse(options["stream"])
        self.assertEqual(options["format"]["type"], "object")
        self.assertEqual(options["options"]["temperature"], 0)
        self.assertIn("Farþegar", options["messages"][1]["content"])

    def test_model_can_abstain(self):
        result = select_tables_llm(self.question, [(self.first, 0.2)], session=fake_session([], clarify=True))
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["selected"], [])
        self.assertTrue(result["needs_clarification"])

    def test_multiple_tables(self):
        result = select_tables_llm(self.question, [(self.first, 2.8), (self.second, 1.5)],
                                   session=fake_session([1, 2]), max_tables=2)
        self.assertEqual(len(result["selected"]), 2)

    def test_rejects_out_of_range_model_choice(self):
        with self.assertRaises(TableSelectionError):
            select_tables_llm(self.question, [(self.first, 1)], session=fake_session([3]))

    def test_rejects_duplicate_model_choice(self):
        with self.assertRaises(TableSelectionError):
            select_tables_llm(self.question, [(self.first, 1)], session=fake_session([1, 1]))

    def test_rejects_boolean_as_index(self):
        with self.assertRaises(TableSelectionError):
            select_tables_llm(self.question, [(self.first, 1)], session=fake_session([True]))

    def test_http_error_is_clear(self):
        session = Mock()
        session.post.side_effect = requests.ConnectionError("server offline")
        with self.assertRaisesRegex(TableSelectionError, "server offline"):
            select_tables_llm(self.question, [(self.first, 1)], session=session)

    def test_incomplete_json_retries_once_and_recovers(self):
        broken = Mock()
        broken.json.return_value = {
            "message": {"content": '{"selected_numbers": [1], "reason": "Long explanation'},
            "done_reason": "length",
        }
        broken.raise_for_status.return_value = None
        good = Mock()
        good.json.return_value = {"message": {"content": json.dumps({
            "selected_numbers": [1], "reason": "Matched topic and dates.",
            "needs_clarification": False,
        })}}
        good.raise_for_status.return_value = None
        session = Mock()
        session.post.side_effect = [broken, good]
        result = select_tables_llm(self.question, [(self.first, 2.8)], session=session)
        self.assertEqual(result["selected"][0]["id"], "SAM02002.px")
        self.assertEqual(session.post.call_count, 2)

    def test_repeated_incomplete_json_has_clear_error(self):
        broken = Mock()
        broken.json.return_value = {
            "message": {"content": '{"selected_numbers": [1], "reason": "Long explanation'},
            "done_reason": "length",
        }
        broken.raise_for_status.return_value = None
        session = Mock()
        session.post.return_value = broken
        with self.assertRaisesRegex(TableSelectionError, "done_reason=length"):
            select_tables_llm(self.question, [(self.first, 2.8)], session=session)
        self.assertEqual(session.post.call_count, 2)

    def test_empty_candidates_skips_http(self):
        session = Mock()
        result = select_tables_llm(self.question, [], session=session)
        self.assertEqual(result["status"], "no_candidates")
        session.post.assert_not_called()

    def test_duplicate_urls_are_not_sent_twice(self):
        session = fake_session([1])
        result = select_tables_llm(self.question, [(self.first, 2), (self.first, 1)], session=session)
        self.assertEqual(result["candidates_considered"], 1)

    def test_same_id_different_url_stays_distinct(self):
        second = table("SAM02002.px", url="https://example.test/elsewhere/SAM02002.px")
        result = select_tables_llm(self.question, [(self.first, 2), (second, 1)],
                                   session=fake_session([2]))
        self.assertEqual(result["selected"][0]["url"], second["url"])


if __name__ == "__main__":
    unittest.main()
