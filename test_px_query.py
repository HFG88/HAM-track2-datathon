"""Offline tests for px_query.py (no calls to Hagstofa)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from px_query import PXQueryError, prepare_query, query_table


SAMPLE_TABLE = {
    "id": "SAM02002.px",
    "title": "Farþegar með skemmtiferðaskipum",
    "category": "Atvinnuvegir",
    "url": "https://px.hagstofa.is/pxis/api/v1/is/SAM02002.px",
    "metadata_status": "ok",
    "metadata": {
        "title": "Farþegar með skemmtiferðaskipum",
        "variables": [
            {"code": "Eining", "text": "Eining",
             "values": ["0", "1"], "valueTexts": ["Skipakomur", "Farþegar alls"]},
            {"code": "Ár", "text": "Ár", "values": ["2023", "2024"],
             "valueTexts": ["2023", "2024"], "time": True},
        ],
    },
}


class PXQueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = Path(self.temp.name) / "catalog.json"
        self.catalog.write_text(json.dumps([SAMPLE_TABLE]), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_label_to_code(self):
        prepared = prepare_query(
            "SAM02002.px", {"Eining": "Farþegar alls", "Ár": "2024"},
            catalog_path=self.catalog,
        )
        self.assertEqual(prepared["cells_requested"], 1)
        self.assertEqual(prepared["payload"]["response"], {"format": "json"})
        self.assertEqual(prepared["payload"]["query"], [
            {"code": "Eining", "selection": {"filter": "item", "values": ["1"]}},
            {"code": "Ár", "selection": {"filter": "item", "values": ["2024"]}},
        ])

    def test_case_and_accent_insensitive(self):
        prepared = prepare_query(
            "SAM02002.px", {"eining": "farþegar ALLS", "ar": 2024},
            catalog_path=self.catalog,
        )
        self.assertEqual(prepared["payload"]["query"][1]["code"], "Ár")

    def test_missing_dimension_fails(self):
        with self.assertRaisesRegex(PXQueryError, "missing: Ár"):
            prepare_query("SAM02002.px", {"Eining": "Farþegar alls"},
                          catalog_path=self.catalog)

    def test_unknown_value_fails(self):
        with self.assertRaisesRegex(PXQueryError, "Unknown value"):
            prepare_query("SAM02002.px", {"Eining": "Bifreiðar", "Ár": "2024"},
                          catalog_path=self.catalog)

    def test_large_query_fails_before_network(self):
        with self.assertRaisesRegex(PXQueryError, "limit 2"):
            prepare_query("SAM02002.px", {"Eining": ["0", "1"],
                              "Ár": ["2023", "2024"]},
                          catalog_path=self.catalog, max_cells=2)

    def test_query_result_and_source(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "columns": [{"code": "Eining", "text": "Eining", "type": "d"},
                        {"code": "Ár", "text": "Ár", "type": "t"}],
            "data": [{"key": ["1", "2024"], "values": ["12345"]}],
        }
        fake_session = Mock()
        fake_session.post.return_value = response
        result = query_table("SAM02002.px", {"Eining": "Farþegar alls", "Ár": "2024"},
                             catalog_path=self.catalog, session=fake_session)
        self.assertEqual(result["data"][0]["values"], ["12345"])
        self.assertEqual(result["source_url"], SAMPLE_TABLE["url"])
        fake_session.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
