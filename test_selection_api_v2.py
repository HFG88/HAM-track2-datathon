"""Check API orchestration without contacting Ollama or PX-API."""

import unittest
from unittest.mock import patch

import selection_api_v2 as api


class APIOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.request = api.QuestionRequest(spurning="Hversu margir farþegar komu með skemmtiferðaskipum til Reykjavíkurhafnar árið 2024?")
        self.table = {"id": "SAM02002.px", "url": "https://px.hagstofa.is/pxis/api/v1/is/SAM02002.px", "title": "Cruise passengers"}

    def test_no_tables_does_not_call_filters_or_px(self):
        with patch.object(api, "select_table", return_value={"selection": {"status": "no_match", "selected": []}}), \
             patch.object(api, "resolve_filters_endpoint") as resolver, \
             patch.object(api, "query_table") as query:
            result = api.answer(self.request)
        self.assertEqual(result["status"], "needs_clarification")
        resolver.assert_not_called()
        query.assert_not_called()

    def test_multiple_tables_does_not_execute_unplanned_comparison(self):
        with patch.object(api, "select_table", return_value={"selection": {
            "status": "selected", "selected": [self.table, self.table]}},), \
             patch.object(api, "query_table") as query:
            result = api.answer(self.request)
        self.assertEqual(result["status"], "needs_multi_table_planning")
        query.assert_not_called()

    def test_missing_filters_do_not_query_px(self):
        with patch.object(api, "select_table", return_value={"selection": {
            "status": "selected", "selected": [self.table]}}), \
             patch.object(api, "resolve_filters_endpoint", return_value={
                 "status": "needs_clarification", "unresolved": [{"dimension": "Ár", "reason": "Year missing"}]
             }), \
             patch.object(api, "query_table") as query:
            result = api.answer(self.request)
        self.assertEqual(result["status"], "needs_clarification")
        query.assert_not_called()

    def test_only_valid_filters_reach_px_api(self):
        resolved = {"status": "ready", "selections": {"Eining": ["1"], "Ár": ["2024"]}}
        px_result = {"data": [{"key": ["1", "2024"], "values": ["321966"]}],
                     "source_url": self.table["url"]}
        with patch.object(api, "select_table", return_value={"selection": {
            "status": "selected", "selected": [self.table]}}), \
             patch.object(api, "resolve_filters_endpoint", return_value=resolved), \
             patch.object(api, "query_table", return_value=px_result) as query:
            result = api.answer(self.request)
        self.assertEqual(result["status"], "data_retrieved")
        self.assertEqual(result["px_result"]["data"][0]["values"], ["321966"])
        query.assert_called_once_with(self.table["url"], resolved["selections"])


if __name__ == "__main__":
    unittest.main()
