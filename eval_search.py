"""Measure how often table_search finds the right table.

Each question lists the table ids that are an acceptable answer. Run:
    python eval_search.py           # summary
    python eval_search.py -v        # also show the top 3 for every miss
"""

import sys
import time

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
}


def evaluate(verbose=False, k=5):
    top1 = topk = 0
    misses = []
    start = time.time()

    for question, expected in TEST_QUESTIONS.items():
        try:
            results = search_tables(normalize_question(question), limit=k)
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

    total = len(TEST_QUESTIONS)
    seconds = (time.time() - start) / total
    print(f"Top 1: {top1}/{total}   Top {k}: {topk}/{total}   ({seconds:.2f} s per question)")

    if verbose:
        for question, rank, results in misses:
            where = f"found at #{rank}" if rank else "not found"
            print(f"\n{question}  ({where})")
            for table, score in results[:3]:
                print(f"  {score:7.2f}  {table['id']:13} {table['title'][:80]}")

    return top1, topk


if __name__ == "__main__":
    evaluate(verbose="-v" in sys.argv)
