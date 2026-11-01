"""Run the questions in spurningar.txt through llm_rewrite with a chosen model.

Shows the keywords each model gives and how long it takes, so models can be
compared without editing llm_rewrite.py. Run from the project root:

    python AI_test/compare_models.py --model qwen3:4b
    python AI_test/compare_models.py --model qwen3:4b --limit 10
    python AI_test/compare_models.py --model qwen3:4b --model gemma3:4b --limit 10
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_search import load_spurningar  # noqa: E402
from llm_rewrite import rewrite  # noqa: E402
from normalize import QuestionError, normalize_question  # noqa: E402


def run(model, questions):
    seconds = []
    failures = 0

    print(f"\n===== {model} =====")
    for question in questions:
        try:
            normalized = normalize_question(question)
        except QuestionError:
            print(f"\n{question}\n  (normaliser rejected the question)")
            continue

        start = time.time()
        result = rewrite(normalized, model=model)
        seconds.append(time.time() - start)

        print(f"\n{question}")
        if "llm_error" in result:
            failures += 1
            print(f"  ERROR: {result['llm_error']}")
            continue
        print(f"  IS: {result['ISKeywords']}")
        print(f"  EN: {result['ENKeywords']}")

    if seconds:
        average = sum(seconds) / len(seconds)
        print(f"\n{model}: {len(seconds)} questions, {failures} failed, {average:.1f} s per question")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="Ollama model, can be repeated")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    args = parser.parse_args()

    questions = list(load_spurningar())[:args.limit]
    for model in args.model:
        run(model, questions)
