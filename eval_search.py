"""Measure how often table_search finds the right table.

Two test sets, where each question lists the table ids that are an acceptable answer:
- "mine": 24 short questions in this file
- "spurningar": 83 questions from AI_test/spurningar.txt

Run:
    python eval_search.py                        # all modes on both sets
    python eval_search.py --cascade --bm25       # only some modes
    python eval_search.py --spurningar -v        # one set, show the top 3 for every miss
"""

import re
import sys
import time
from pathlib import Path

from normalize import QuestionError, normalize_question
from table_search import search_tables

# Question -> acceptable table ids. Add more as you find them.
TEST_QUESTIONS = {
    "Hver er mannfjöldi í Kópavogi?":
        ["MAN02005.px", "MAN10001.px", "MAN04203.px", "MAN04208.px", "MAN12104.px"],
    "Hvað búa margir á Íslandi?":
        ["MAN00000.px", "MAN00101.px"],
    "Hvað bjuggu margir á Akureyri árið 2020?":
        ["MAN02005.px", "MAN10001.px", "MAN04203.px", "MAN04208.px", "MAN12104.px"],
    "Hver var verðbólgan í mars 2024?":
        ["VIS01000.px", "VIS01002.px"],
    "Hversu margir fluttu frá Austurlandi 2023?":
        ["MAN01002.px", "MAN01201.px"],
    "Hversu margir fluttu til útlanda 2024?":
        ["MAN01400.px", "MAN01401.px", "MAN01404.px", "MAN01405.px", "MAN01406.px"],
    "Hvað eru margir atvinnulausir?":
        ["VIN00920.px", "VIN00910.px", "VIN00001.px", "VIN00002.px", "VIN01206.px", "VIN00971.px"],
    "Hvert er atvinnuleysið á Íslandi?":
        ["VIN00920.px", "VIN00910.px", "VIN00911.px", "VIN00001.px", "VIN00002.px", "VIN01002.px"],
    "Hvað fæddust mörg börn 2020?":
        ["MAN05203.px", "MAN05302.px", "MAN05101.px", "MAN05102.px"],
    "Hvað fæddust mörg börn í Reykjavík?":
        ["MAN05110.px"],
    "Hvað dóu margir árið 2023?":
        ["MAN05210.px", "MAN05221.px"],
    "Hvað voru margar gistinætur á hótelum 2024?":
        ["SAM01102.px", "SAM01103.px", "SAM01201.px", "SAM01603.px"],
    "Hvað fóru margir farþegar um Keflavíkurflugvöll?":
        ["SAM02001.px", "FER90001.px"],
    "Hvað veiddist mikill þorskur 2023?":
        ["SJA09005.px", "SJA01101.px", "SJA01102.px", "SJA09001.px", "SJA02201.px"],
    "Hver var landsframleiðslan 2024?":
        ["THJ01000.px", "THJ01104.px", "THJ01102.px", "THJ01601.px"],
    "Hvað búa margir í hverfum Reykjavíkur?":
        ["MAN03270.px"],
    "Hvað eru margir innflytjendur í hverju sveitarfélagi?":
        ["MAN43005.px", "MAN43007.px"],
    "Hver er launavísitalan?":
        ["LAU04000.px", "LAU04200.px"],
    "Hver eru laun eftir starfsstétt og kyni?":
        ["VIN02001.px", "VIN02002.px", "VIN02003.px"],
    "Hvað hefur verð á íbúðarhúsnæði hækkað?":
        ["VIS01106.px"],
    "Hvað voru byggðar margar íbúðir?":
        ["IDN03001.px"],
    "Hver er raforkunotkun eftir atvinnugreinum?":
        ["UMH40210.px", "IDN02104.px"],
    # Everyday wording that shares few words with the table titles.
    "Hvað eru margir án vinnu?":
        ["VIN00920.px", "VIN00921.px", "VIN00910.px", "VIN00001.px", "VIN00002.px", "VIN00003.px"],
    "Hvað komu margir ferðamenn til landsins?":
        ["SAM02001.px", "FER90001.px", "SAM08006.px"],
}


SPURNINGAR_PATH = Path(__file__).parent / "AI_test" / "spurningar.txt"


def load_spurningar(path=SPURNINGAR_PATH):
    """Read topic lines like "Name (MAN001, MAN002)" and the numbered questions under them."""
    questions = {}
    expected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        topic = re.match(r"^\S.*\(([A-Z0-9, ]+)\)\s*$", line)
        if topic:
            expected = [f"{i.strip()}.px" for i in topic.group(1).split(",")]
            continue
        question = re.match(r"^\s+\d+\.\s+(.*)$", line)
        if question and expected:
            # Some questions also name a table themselves, e.g. "(SAM02001)".
            extra = [f"{i}.px" for i in re.findall(r"\b([A-Z]{3}\d{4,6}[A-Za-z]?)\b", question.group(1))]
            questions[question.group(1)] = expected + extra
    return questions


def test_sets():
    sets = {"mine": TEST_QUESTIONS}
    if SPURNINGAR_PATH.exists():
        sets["spurningar"] = load_spurningar()
    return sets


def evaluate(mode="hybrid", verbose=False, k=5, questions=None, name=""):
    top1 = topk = 0
    misses = []
    start = time.time()

    questions = questions or TEST_QUESTIONS
    for question, expected in questions.items():
        try:
            results = search_tables(normalize_question(question), limit=k, mode=mode)
        except QuestionError:
            results = []
        ids = [table["id"] for table, _ in results]
        rank = next((i + 1 for i, table_id in enumerate(ids) if table_id in expected), None)

        if rank == 1:
            top1 += 1
        if rank:
            topk += 1
        if rank != 1:
            misses.append((question, rank, results))

    total = len(questions)
    seconds = (time.time() - start) / total
    print(f"{name:11} {mode:8} Top 1: {top1}/{total}   Top {k}: {topk}/{total}   ({seconds:.2f} s per question)")

    if verbose:
        for question, rank, results in misses:
            where = f"found at #{rank}" if rank else "not found"
            print(f"\n{question}  ({where})")
            for table, score in results[:3]:
                print(f"  {score:7.2f}  {table['id']:13} {table['title'][:80]}")

    return top1, topk


if __name__ == "__main__":
    all_modes = ["bm25", "vector", "hybrid", "cascade"]
    modes = [m[2:] for m in sys.argv[1:] if m[2:] in all_modes]
    sets = test_sets()
    chosen = [name for name in sets if f"--{name}" in sys.argv] or list(sets)
    for name in chosen:
        for mode in modes or all_modes:
            evaluate(mode, verbose="-v" in sys.argv, questions=sets[name], name=name)
