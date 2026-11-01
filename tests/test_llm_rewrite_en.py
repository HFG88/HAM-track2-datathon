import unittest
from unittest.mock import patch

from llm_rewrite_en import rewrite


class EnglishRewriteTests(unittest.TestCase):
    @patch(
        "llm_rewrite_en.ask_llm",
        return_value={
            "ISKeywords": ["atvinnuleysi", "atvinnuleitendur", "atvinnuleysi"],
            "ENKeywords": ["unemployment", "jobseekers", "unemployment"],
        },
    )
    def test_returns_aligned_icelandic_and_english_keywords(self, ask_llm):
        question = "How many people are unemployed?"

        result = rewrite(question)

        ask_llm.assert_called_once_with(question, model="qwen3:8b")
        self.assertEqual(result["original"], question)
        self.assertEqual(
            result["ISKeywords"], ["atvinnuleysi", "atvinnuleitendur"]
        )
        self.assertEqual(result["ENKeywords"], ["unemployment", "jobseekers"])
        self.assertNotIn("llm_error", result)


if __name__ == "__main__":
    unittest.main()
