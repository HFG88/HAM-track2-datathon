# LLM query rewriting (steps 2–3)

## What this is

When a question comes in, `normalize.py` cleans it and pulls out keywords, years, breakdowns and so on. Those keywords are used to search for the right Hagstofa table.

The problem is that people use everyday words while table titles use official terms. Someone asks *"Hvað eru margir án vinnu?"*, but the table is called *Atvinnuleysi…*. Our synonym list only covers words we thought of in advance.

`llm_rewrite_is.py` asks a local LLM (via Ollama) for **extra search words** to close that gap. It accepts Icelandic questions and returns Icelandic and English keywords.
`llm_rewrite_en.py` is the English-input version: it accepts an English question and returns aligned Icelandic and English keyword pairs. Both modules only add search words; they never pick the table or touch the data.

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
from llm_rewrite_is import rewrite

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

# Download a model (see "Models to try" below)
ollama pull llama3.2:3b

# Run the test questions
python llm_rewrite_is.py
python llm_rewrite_is.py "Your own question here?"
```

To change the model, edit `MODEL` at the top of `llm_rewrite_is.py`. Nothing else needs to change: all the models below work with the same code.

For English questions, run:

```bash
python llm_rewrite_en.py "How many people are unemployed?"
```

Or call it from Python:

```python
from llm_rewrite_en import rewrite

result = rewrite("How many people are unemployed?")
result["ISKeywords"]  # e.g. ["atvinnuleysi", "atvinnuleitendur"]
result["ENKeywords"]  # e.g. ["unemployment", "jobseekers"]
```

To automatically choose the Icelandic or English pipeline, use
`question_router.py`:

```bash
python question_router.py "Hvað eru margir án vinnu?"
python question_router.py "How many people are unemployed?"
```

The router detects the language, runs Icelandic questions through
`normalize.py` and `llm_rewrite_is.py`, and sends English questions to
`llm_rewrite_en.py`. Detection is a lightweight heuristic based on common
question/topic words and Icelandic characters. In either case, the returned
JSON includes the normalizer fields (`keywords`, `time`, `intent`, etc.) plus
`ISKeywords` and `ENKeywords`.

## Models to try

We're allowed to use other models than Llama. The question is which one understands Icelandic best and is fast enough. Sizes are approximate download sizes.

| Model | Ollama name | Size | Notes |
|---|---|---|---|
| Llama 3.2 3B | `llama3.2:3b` | ~2 GB | What we tested. Bad at Icelandic, OK in English. |
| Llama 3.1 8B | `llama3.1:8b` | ~4.7 GB | Same family as Hagur in the slides. |
| Gemma 3 4B | `gemma3:4b` | ~3.3 GB | Google. Known for being good at smaller languages. |
| Gemma 3 12B | `gemma3:12b` | ~8 GB | Probably the best balance if you have the disk and RAM. |
| Qwen 3 4B | `qwen3:4b` | ~2.5 GB | Alibaba. Strong overall; Icelandic unknown. |
| Qwen 3 8B | `qwen3:8b` | ~5 GB | |

**Gemma 3** is the most interesting one. It may be good enough at Icelandic that we can ask for **Icelandic** keywords directly and skip the English catalog. Test that (task 2 below).

**Qwen 3** "thinks" (writes out its reasoning) before answering by default, which makes it much slower. The code sends `"think": False` to turn that off; other models ignore it.

For a neutral comparison of models on Icelandic, see Miðeind's leaderboard: https://mideind-icelandic-llm-leaderboard.hf.space

## Your tasks

1. **Compare models.** Run `python llm_rewrite_is.py` with each model you can fit. For each one, write down how many of the 7 questions got good keywords (tourism and teacher wages are the hard ones) and how many seconds per question. On a laptop without a GPU, Llama 3B took ~2 s.
2. **Try Icelandic keywords with Gemma.** Change `SYSTEM_PROMPT` to ask for Icelandic keywords in nominative form, e.g. *"án vinnu"* → *"atvinnuleysi"*. If Gemma gives real Icelandic words (Llama 3B made up words like *"útlagisbær"*), we may not need the English catalog at all.
3. **Improve the prompt.** Change `SYSTEM_PROMPT` and `EXAMPLES`. Adding more example question/keyword pairs usually helps the most.
4. **Try more questions**, especially everyday wording (*"án vinnu"*, *"leigja"*, *"ferðamenn"*), since that's where the LLM should help.
5. **Write down what you find** in a table like the one above: model, prompt, score out of 7, and seconds per question. That tells us which model to use for Hagur (step 7) as well.

## Things to know

- Settings that matter: `temperature: 0` makes answers the same every time. `num_predict: 60` stops the model if it starts repeating itself; without it, 3b sometimes looped until it timed out.
- The LLM is an extra on top of the rules, not a replacement. The normaliser's keywords and synonyms still do the main work, and they're fast and give the same result every time.
- Step 7 (Hagur, Llama 3 + LoRA) is separate. This is only about search words.
