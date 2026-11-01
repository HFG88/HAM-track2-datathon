"""Compare table_search alone with table_search + table_selection.

Uses the same provisional ground-truth IDs as eval_search.py. In the
spurningar set these IDs come from *section headings*, not manually verified
question-level answers, so results are directional rather than definitive.

Run from the repository root:
    python eval_selection.py --set mine --mode bm25
    python eval_selection.py --set spurningar --mode bm25 --show-changes
    python eval_selection.py --set spurningar --mode cascade

BM25 needs only the local enriched catalog. Cascade/vector modes require
Ollama embeddings; table_search.py may fall back to BM25 when unavailable.
"""

import argparse
import time

from eval_search import TEST_QUESTIONS, load_spurningar
from normalize import QuestionError, normalize_question
from table_search import search_tables
from table_selection import select_tables


def first_rank(ids, acceptable):
    return next((i for i, table_id in enumerate(ids, 1) if table_id in acceptable), None)


def evaluate(questions, *, mode="bm25", pool=20, top=5, show_changes=False, max_show=12):
    """Evaluate identical retrieved pools before/after metadata selection."""
    counts = {
        "total": 0, "normalizer_rejected": 0,
        "baseline_top1": 0, "baseline_topk": 0,
        "selected_top1": 0, "selected_topk": 0,
        "expected_in_pool": 0, "expected_rejected_all": 0,
        "top1_improved": 0, "top1_worsened": 0,
        "topk_improved": 0, "topk_worsened": 0,
    }
    examples = []
    start = time.perf_counter()

    for question, acceptable_ids in questions.items():
        counts["total"] += 1
        acceptable = set(acceptable_ids)
        try:
            normalized = normalize_question(question)
        except QuestionError:
            counts["normalizer_rejected"] += 1
            continue

        # Give BOTH methods precisely the same starting pool, not different
        # retrieval calls or different candidate counts.
        candidates = search_tables(normalized, limit=pool, mode=mode)
        baseline = [table["id"] for table, _ in candidates]
        # Select with the full pool as limit so genuine metadata rejections
        # can be separated from merely falling outside a displayed top 5.
        reranked = select_tables(normalized, candidates, limit=pool)
        selected = [entry["id"] for entry in reranked["selected"]]
        rejected = [entry["id"] for entry in reranked["rejected"]]

        before = first_rank(baseline, acceptable)
        after = first_rank(selected, acceptable)
        counts["baseline_top1"] += before == 1
        counts["selected_top1"] += after == 1
        counts["baseline_topk"] += before is not None and before <= top
        counts["selected_topk"] += after is not None and after <= top
        counts["expected_in_pool"] += before is not None
        # An acceptable ID was retrieved, but *none* of the retrieved
        # acceptable tables survived metadata checks.
        rejected_all = before is not None and after is None
        counts["expected_rejected_all"] += rejected_all

        top1_improved = after == 1 and before != 1
        top1_worsened = before == 1 and after != 1
        topk_improved = (after is not None and after <= top) and (before is None or before > top)
        topk_worsened = (before is not None and before <= top) and (after is None or after > top)
        counts["top1_improved"] += top1_improved
        counts["top1_worsened"] += top1_worsened
        counts["topk_improved"] += topk_improved
        counts["topk_worsened"] += topk_worsened

        if show_changes and (top1_improved or top1_worsened or topk_improved
                             or topk_worsened or rejected_all):
            examples.append({
                "question": question,
                "expected": acceptable_ids,
                "before": before, "after": after,
                "baseline": baseline[:3], "selected": selected[:3],
                "rejected_expected": [table_id for table_id in rejected if table_id in acceptable],
            })

    duration = time.perf_counter() - start
    total = counts["total"]
    print(f"\nQuestions: {total}  | Mode: {mode}  | Candidate pool: {pool}  | Top K: {top}")
    print(f"{'Metric':29} {'Search only':>14} {'+ Selection':>14}")
    print(f"{'Correct at #1':29} {counts['baseline_top1']:>9}/{total:<4} {counts['selected_top1']:>9}/{total:<4}")
    print(f"{f'Correct in top {top}':29} {counts['baseline_topk']:>9}/{total:<4} {counts['selected_topk']:>9}/{total:<4}")
    print(f"Acceptable ID anywhere in candidate pool: {counts['expected_in_pool']}/{total}")
    print(f"Acceptable retrieved IDs all rejected:    {counts['expected_rejected_all']}")
    print(f"Rank-1 gains/losses: {counts['top1_improved']} / {counts['top1_worsened']}")
    print(f"Top-{top} gains/losses: {counts['topk_improved']} / {counts['topk_worsened']}")
    print(f"Normalizer rejected: {counts['normalizer_rejected']}")
    print(f"Elapsed: {duration:.1f} sec")

    for entry in examples[:max_show]:
        print(f"\n{entry['question']}")
        print(f"  Expected IDs: {', '.join(entry['expected'])}")
        print(f"  First acceptable rank: {entry['before']} -> {entry['after']}")
        print(f"  Search top 3:   {', '.join(entry['baseline'])}")
        print(f"  Selected top 3: {', '.join(entry['selected'])}")
        if entry["rejected_expected"]:
            print(f"  Rejected acceptable IDs: {', '.join(entry['rejected_expected'])}")
    if show_changes and len(examples) > max_show:
        print(f"\n... {len(examples) - max_show} more changed questions (use --max-show to see more)")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["mine", "spurningar", "both"], default="mine")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid", "cascade"], default="bm25")
    parser.add_argument("--pool", type=int, default=20, help="Tables retrieved before reranking")
    parser.add_argument("--top", type=int, default=5, help="Top-K success threshold")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N questions")
    parser.add_argument("--show-changes", action="store_true")
    parser.add_argument("--max-show", type=int, default=12)
    args = parser.parse_args()

    if args.pool < 1 or args.top < 1 or args.top > args.pool:
        parser.error("Require pool >= top >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    sets = {}
    if args.set in ("mine", "both"):
        sets["mine"] = TEST_QUESTIONS
    if args.set in ("spurningar", "both"):
        sets["spurningar"] = load_spurningar()
    for label, questions in sets.items():
        if args.limit:
            questions = dict(list(questions.items())[:args.limit])
        print(f"\n===== {label.upper()} =====")
        evaluate(questions, mode=args.mode, pool=args.pool, top=args.top,
                 show_changes=args.show_changes, max_show=args.max_show)


if __name__ == "__main__":
    main()
