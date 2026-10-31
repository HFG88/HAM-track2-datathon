# LLM query rewriting (steps 2–3)

## What this is

When a question comes in, `normalize.py` cleans it and pulls out keywords, years, breakdowns and so on. Those keywords are used to search for the right Hagstofa table.

The problem is that people use everyday words while table titles use official terms. Someone asks *"Hvað eru margir án vinnu?"*, but the table is called *Atvinnuleysi…*. Our synonym list only covers words we thought of in advance.

`llm_rewrite.py` asks a local LLM (Llama via Ollama) for **extra search words** to close that gap. The LLM only adds words. It never picks the table and never touches the data.

## Why English?

We tested `llama3.2:3b` and it is bad at Icelandic. It made up words like *"útlagisbær"* and *"rafnemunumr"*, and got 0 of 6 questions right. Asked for **English** keywords, it got 5 of 7 right:

| Question | English keywords | |
|---|---|---|
| Hvað eru margir án vinnu? | unemployment, labour force | ✅ |
| Hvað kostar að leigja íbúð? | housing prices, rental prices | ✅ |
| Hversu mikið rafmagn er framleitt? | electricity, generation | ✅ |
| Hversu margir fluttu frá Austurlandi 2023? | migration, net migration | ✅ |
| Hvað fæddust mörg börn 2020? | births, population growth | ✅ |
| Hvað komu margir ferðamenn til landsins 2024? | migration, immigration | ❌ should be tourism |
| Hver eru meðallaun kennara? | unemployment, labour force | ❌ should be wages |

Hagstofa has an English API with the **same table ids**: `https://px.hagstofa.is/pxen/api/v1/en/`, e.g. `MAN00101.px` = "Population by sex and age". So the plan is:

1. Crawl the English titles into `local/catalog-en.json`, in the same format as `catalog-full.json` (Hermann is doing this).
2. The LLM gives English keywords, and the search also looks in the English titles.
3. A hit maps back to the same table id. Everything after that stays in Icelandic.

## How it works

```python
from normalize import normalize_question
from llm_rewrite import rewrite

result = rewrite(normalize_question("Hvað eru margir án vinnu?"))
result["llm_keywords_en"]   # ["unemployment", "labour", "force"]
```

`rewrite()` returns the normaliser's dict with one extra field:

- `llm_keywords_en`: English search words. Once `local/catalog-en.json` exists, words that appear in no English title are dropped, so the LLM cannot make up terms.
- `llm_error`: only present if Ollama failed (not running, timeout, bad JSON). The rest of the dict is unchanged, so the search still works without the LLM.

## Setup

```bash
# Install Ollama (asks for sudo)
curl -fsSL https://ollama.com/install.sh | sh

# Download a model
ollama pull llama3.2:3b     # ~2 GB, what we tested with
ollama pull llama3.1:8b     # ~4.7 GB, probably better

# Run the test questions
python llm_rewrite.py
python llm_rewrite.py "Your own question here?"
```

To change the model, edit `MODEL` at the top of `llm_rewrite.py`.

## Your tasks

1. **Compare 3b and 8b.** Run `python llm_rewrite.py` with both models. Does 8b get the tourism and teacher wages questions right? How long does each one take per question? On a laptop without a GPU, 3b took ~2 s.
2. **Improve the prompt.** Change `SYSTEM_PROMPT` and `EXAMPLES` in `llm_rewrite.py`. Adding more example question/keyword pairs usually helps the most. Use the test questions with known correct tables if we have them.
3. **Try more questions**, especially everyday wording (*"án vinnu"*, *"leigja"*, *"ferðamenn"*), since that's where the LLM should help.
4. **Write down what you find**: which model, which prompt, how many questions got good keywords, and how fast.

## Things to know

- Settings that matter: `temperature: 0` makes answers the same every time. `num_predict: 60` stops the model if it starts repeating itself; without it, 3b sometimes looped until it timed out.
- The LLM is an extra on top of the rules, not a replacement. The normaliser's keywords and synonyms still do the main work, and they're fast and give the same result every time.
- Step 7 (Hagur, Llama 3 + LoRA) is separate. This is only about search words.
