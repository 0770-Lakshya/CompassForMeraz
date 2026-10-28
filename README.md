# CompassForMeraz

A question-answering bot for one website. You write the knowledge in a text
file, it answers visitors' questions from **only** that file, and it refuses
rather than guesses when the answer is not there.

Derived from Compass, with the crawler, the HTML chunker, the indexer and all
the navigation machinery removed.

## Quick start

```powershell
py -3.12 -m venv .venv      # use Python 3.12: onnxruntime has no 3.14 wheel yet
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env         # then paste your GROQ_API_KEY into it

python manage.py ingest content\meraz.txt
python manage.py runserver          # binds :8001, not Django's default :8000
```

The port is 8001 because the main Compass project also uses Django's default
8000, and running both is a confusing failure: the port is open and the server
answers, it just does not have the endpoint the caller wants. See
`ask/management/commands/runserver.py`.

Then in a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod -Uri http://127.0.0.1:8001/ask -Method Post -ContentType application/json -Body '{"question":"is there a registration fee"}'
```

To see the widget: `cd widget`, `python -m http.server 5500`, open
<http://localhost:5500/demo.html>.

The first `ingest` downloads ~90MB of model weights. After that it is offline
and free — no embedding API, no per-query cost beyond the LLM call.

## The loop

```
edit content\meraz.txt  ->  python manage.py ingest content\meraz.txt  ->  restart the server
```

The restart is needed because the index is held in memory. For a file you edit
by hand a few times, that is cheaper than building a reload endpoint and the
auth to protect it.

## How it works

```
content/meraz.txt
      |  core/text.py          headings become chunk boundaries
      v
   chunks
      |  core/retrieval.py     each chunk -> a 384-dim vector
      v
 data/chunks.json + data/embeddings.npy
      |
      |  a question arrives at POST /ask
      v
 core/retrieval.search()       hybrid: 0.65 semantic + 0.35 fuzzy + level bonus
      |
      v  top 3 candidates
 core/answer.answer()          GATE 1: below 0.30 confidence -> refuse, no LLM
      |                        GATE 2: LLM may return index 0 -> refuse
      v
   {found, heading, explanation, confidence, url}
```

### Why retrieval is hybrid rather than purely semantic

Embeddings are good at paraphrase — "is it free to enter" matching a heading
that reads "Registration fees" — and weak at exact tokens: names, acronyms,
typos. Fuzzy string matching is the reverse. Blending them covers both, and the
weights (0.65 / 0.35) are carried over from the parent project where they were
tuned against real queries.

The two halves deliberately read **different text**. Semantic sees title +
heading + heading + content; lexical sees only title + heading. Including the
body in the fuzzy half would make every long chunk match everything, because
`token_set_ratio` compares word sets and a long body contains most words.

The heading is repeated in the embedded text on purpose: an embedding is
loosely an average of its input's meaning, so a long body drowns out a short
heading. Repeating it doubles its weight.

### Why there are two refusal gates

A bot that confidently answers questions its data cannot support is worse than
no bot, especially on a public site. So refusing is checked twice:

1. **The confidence floor (0.30).** If the best retrieval score is below it, we
   refuse *without calling the LLM*. Free, instant, and incapable of
   hallucinating, because no model is involved in the decision.
2. **The model's own veto.** The prompt may return `index: 0` for "none of
   these are about the question", and a malformed index is treated as a refusal
   too — failing in the safe direction.

Raise the floor and the bot says "I don't know" to things it could have
answered; lower it and it answers from chunks that are not really on topic. The
second failure is much worse here, so it errs toward refusing.

## Layout

```
manage.py              Django entry point — run everything from this folder
compass/               project config: settings, urls, wsgi, boot
ask/                   the API: 2 endpoints, request validation, state, cache
  management/commands/ingest.py      manage.py ingest
core/                  framework-free: text parsing, retrieval, answering
content/meraz.txt      THE KNOWLEDGE. Committed — you typed it.
data/                  chunks.json + embeddings.npy. Gitignored — derived.
widget/                bot-widget.js (vanilla, no build) + demo.html
```

`core/` imports no Django at all. That is why the parent project could swap its
entire web framework in a day without touching its pipeline, and it is worth
preserving: keep framework code out of `core/`.

### content/ is committed, data/ is not

These look similar and are opposites:

| | `content/` | `data/` |
|---|---|---|
| Produced by | a person | a machine |
| Recoverable | **no** | yes, re-ingest |
| Committed | **yes** | no |

## The content format

```
title: Meraz 7.0                 optional, once, at the top

# When is Meraz?                 a heading starts a chunk (level h1)
Meraz 7.0 runs in February 2027. body text until the next heading

## Where do I register?          ## is h2, ### is h3
url: https://meraz.in/register   optional, FIRST line under the heading
Registration is on the website.
```

Two things that materially affect answer quality:

**Write headings as the questions people ask.** Retrieval fuzzy-matches against
heading text, so `# Is there a registration fee?` beats `# Fees`. Your headings
are search keys, not a table of contents.

**One fact per section, kept short.** Each chunk becomes a single vector. A
section covering dates *and* fees *and* contacts produces one blurred vector
that matches all three questions weakly instead of one strongly.

`url:` gives a "Read more →" link. A hand-typed fact has no element to scroll
to, so there is no highlighting — that is what the parent project's selectors
are for.

## Endpoints

| | |
|---|---|
| `GET /health` | chunk count and title. First thing to check when the bot says it doesn't know anything — `chunks: 0` means nothing was ingested. |
| `POST /ask` | `{"question": "..."}` → `{found, heading, explanation, confidence, url, reason}` |

The response shape is **always the same**, so the widget needs one code path:
check `found`, then read the fields. A refusal is HTTP 200 with
`found: false`, not an error status — "we don't have that information" is a
normal outcome, not an HTTP-level failure.

## Deploying

`gunicorn compass.wsgi:application --workers 1 --threads 8 --timeout 120`

`--workers 1` matters: the index and the answer cache live in one process's
memory, so a second worker would hold its own divergent copy.

Commit `data/` (or run `ingest` in the build step) if the host has an ephemeral
filesystem — otherwise the deployed server comes up with nothing indexed.

## What was deliberately left out

No crawler, no HTML chunker, no CSS selectors, no multi-site support, no
database. If you later need the bot to answer about pages you did not write by
hand, that is the parent Compass project, not this one.
