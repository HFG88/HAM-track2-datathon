import unittest

from normalize import normalize_question


class NormalizeQuestionTests(unittest.TestCase):
    def test_keeps_valid_words_that_look_similar_to_catalog_terms(self):
        cases = {
            "Hve hátt hlutfall safngesta er erlent?": "hatt",
            "Hvernig hefur áfengisneysla breyst?": "breyst",
            "Hvernig er raforkuverð miðað við önnur Norðurlönd?": "onnur",
            "Hve miklu eyða erlendir ferðamenn í hverjum flokki?": "eyda",
        }

        for question, word in cases.items():
            with self.subTest(question=question):
                result = normalize_question(question)
                self.assertIn(word, result["text"].split())
                self.assertEqual(result["corrections"], {})

    def test_keeps_place_names_unchanged(self):
        result = normalize_question(
            "Hversu margir fluttu frá Austurlandi árið 2024?"
        )

        self.assertIn("austurlandi", result["text"].split())
        self.assertEqual(result["corrections"], {})


if __name__ == "__main__":
    unittest.main()
