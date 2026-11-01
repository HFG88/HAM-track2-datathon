"""Offline checks for table_selection.py; no Ollama or PX API needed."""
import copy
import unittest

from table_selection import select_tables


def variable(code, values, *, time=False, labels=None):
    return {
        "code": code, "text": code, "values": [str(v) for v in values],
        "valueTexts": list(labels) if labels is not None else [str(v) for v in values],
        **({"time": True} if time else {}),
    }


def table(id_, variables, title=None, url=None):
    return {
        "id": id_, "url": url or f"https://px.hagstofa.is/test/{id_}",
        "title": title or id_, "category": "Ibuar", "metadata_status": "ok",
        "metadata": {"variables": variables},
    }


def question(years=None, months=None, quarters=None, breakdowns=None,
             measure=None, relative=None, original=""):
    return {"original": original, "time": {
        "years": years or [], "months": months or [], "quarters": quarters or [],
        "relative": relative,
    }, "breakdowns": breakdowns or [], "measure": measure}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.annual = table("POP.px", [
            variable("Kyn", ["0", "1"], labels=["Alls", "Konur"]),
            variable("Ár", ["2023", "2024"], time=True),
        ], "Mannfjöldi eftir kyni")
        self.old = table("OLD.px", [variable("Ár", ["2000", "2001"], time=True)], "Mannfjöldi 2000-2001")
        self.monthly = table("MON.px", [
            variable("Mánuður", [f"2024M{m:02d}" for m in range(1, 13)], time=True),
            variable("Kyn", ["0", "1"], labels=["Alls", "Konur"]),
        ], "Mannfjöldi eftir mánuðum")

    def test_requested_year_from_metadata(self):
        result = select_tables(question(years=[2024]), [(self.old, 9), (self.annual, 3)])
        self.assertEqual(result["selected"][0]["id"], "POP.px")
        self.assertEqual(result["rejected"][0]["id"], "OLD.px")

    def test_monthly_year_request_requires_aggregation(self):
        result = select_tables(question(years=[2024]), [(self.monthly, 8), (self.annual, 5)])
        self.assertEqual(result["selected"][0]["id"], "POP.px")
        self.assertTrue(any("aggregation" in w for w in result["selected"][1]["warnings"]))

    def test_month_with_year_supported(self):
        result = select_tables(question(years=[2024], months=["03"]), [(self.monthly, 1), (self.annual, 2)])
        self.assertEqual([x["id"] for x in result["selected"]], ["MON.px"])

    def test_month_missing_rejected(self):
        result = select_tables(question(years=[2024], months=["12"]),
                               [(table("M.px", [variable("Mánuður", ["2024M01"], time=True)]), 1)])
        self.assertEqual(result["status"], "no_verified_match")

    def test_partial_monthly_not_annual(self):
        partial = table("PART.px", [variable("Mánuður", ["2024M01", "2024M02"], time=True)])
        result = select_tables(question(years=[2024]), [(partial, 1)])
        self.assertFalse(result["selected"])

    def test_requested_breakdown_required(self):
        result = select_tables(question(years=[2024], breakdowns=["kyn"]),
                               [(table("NO.px", [variable("Ár", ["2024"], time=True)]), 9),
                                (self.annual, 2)])
        self.assertEqual(result["selected"][0]["id"], "POP.px")
        self.assertIn("Missing breakdowns: kyn", result["rejected"][0]["reasons"])

    def test_region_alias(self):
        t = table("REG.px", [variable("Landsvæði", ["Vestfirðir"]),
                                    variable("Ár", ["2024"], time=True)])
        self.assertTrue(select_tables(question(breakdowns=["landsvaedi"]), [(t, 2)])["selected"])

    def test_measure_is_soft_signal(self):
        r = select_tables(question(measure="hlutfall"), [(self.annual, 1)])
        self.assertEqual(len(r["selected"]), 1)
        self.assertTrue(r["selected"][0]["warnings"])

    def test_missing_metadata_is_not_verified(self):
        t = table("NONE.px", [])
        r = select_tables(question(), [(t, 1)])
        self.assertEqual(r["status"], "no_verified_match")
        self.assertEqual(len(r["rejected"]), 1)

    def test_duplicate_urls_removed_but_not_duplicate_ids(self):
        same_id = copy.deepcopy(self.annual)
        same_id["url"] = "https://px.hagstofa.is/test/other/path/POP.px"
        r = select_tables(question(), [(self.annual, 1), (self.annual, 1), (same_id, 1)])
        self.assertEqual(r["checked"], 2)
        self.assertEqual(len(r["selected"]), 2)

    def test_relative_time_awaits_resolution(self):
        r = select_tables(question(relative={"last": 3, "unit": "year"}), [(self.annual, 1)])
        self.assertTrue(any("Relative" in w for w in r["selected"][0]["warnings"]))

    def test_specific_unrequested_age_subgroup_penalized(self):
        specific = table("SPEC.px", [variable("Ár", ["2024"], time=True)],
                         "Mannfjöldi 16-74 ára eftir menntun")
        general = table("GEN.px", [variable("Ár", ["2024"], time=True)],
                        "Mannfjöldi á Íslandi")
        r = select_tables(question(years=[2024], original="Mannfjöldi 2024"), [(specific, 4), (general, 3)])
        self.assertEqual(r["selected"][0]["id"], "GEN.px")

    def test_irrelevant_table_is_flagged_despite_generic_measure(self):
        cruise = table("SAM02002.px", [
            variable("Eining", ["1"], labels=["Farþegar alls"]),
            variable("Ár", ["2024"], time=True),
        ], "Farþegar með skemmtiferðaskipum til Reykjavíkurhafnar 1984-2024")
        finance = table("SJA08106.px", [
            variable("Eining", ["0"], labels=["Fjöldi fyrirtækja"]),
            variable("Ár", ["2024"], time=True),
        ], "Dreifing afkomu fyrirtækja í úrtaki Hagstofunnar 2001-2024")
        q = question(years=[2024], measure="fjoldi",
                     original="Hversu margir farþegar komu með skemmtiferðaskipum 2024?")
        q["keywords"] = ["farthegar", "komu", "skemmtiferdaskipum"]
        q["expanded"] = []
        result = select_tables(q, [(finance, 9), (cruise, 8)])
        self.assertEqual(result["selected"][0]["id"], "SAM02002.px")
        bad = next(t for t in result["selected"] if t["id"] == "SJA08106.px")
        self.assertTrue(any("semantic review" in w for w in bad["warnings"]))
        self.assertEqual(bad["topic_matches"], [])
        self.assertNotIn("Requested measure appears in the table metadata", bad["reasons"])

    def test_no_lexical_overlap_is_not_hard_rejected(self):
        semantic = table("SEM.px", [variable("Ár", ["2024"], time=True)],
                         "Ferðaþjónusta á Íslandi")
        q = question(years=[2024])
        q["keywords"] = ["gestakomur"]
        q["expanded"] = []
        result = select_tables(q, [(semantic, 0.9)])
        self.assertEqual(result["status"], "shortlist")
        self.assertTrue(result["selected"][0]["warnings"])

    def test_empty_candidates(self):
        r = select_tables(question(), [])
        self.assertEqual(r["status"], "no_verified_match")
        self.assertEqual(r["checked"], 0)

    def test_limit(self):
        r = select_tables(question(), [(self.annual, 1), (self.monthly, 1)], limit=1)
        self.assertEqual(len(r["selected"]), 1)

    def test_reject_missing_quarter(self):
        r = select_tables(question(years=[2024], quarters=[1]), [(self.annual, 1)])
        self.assertEqual(r["status"], "no_verified_match")


if __name__ == "__main__":
    unittest.main()
