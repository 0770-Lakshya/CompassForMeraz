"""
===============================================================================
 core/answer.py — retrieve, then let an LLM answer from ONLY what was retrieved.
===============================================================================

Adapted from Compass's answer layer. One thing is gone and one thing changed:

  * GONE: the navigation prompt. Compass had two system prompts — one that
    picks a destination to scroll to, one that answers a question — and chose
    between them per query. This bot has no DOM to scroll to, so only the
    answering prompt exists and the branch disappears with it.

  * CHANGED: no `selector` in the response.

What is deliberately KEPT is the two-stage refusal, which is the single most
important behaviour in the whole file. See GATE 1 and GATE 2 below.

read more: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview
"""

import json
import logging
import os

from core.retrieval import search

# NOT print(). Under gunicorn stdout is block-buffered when it is not a
# terminal, so a print() from inside a request can sit in a buffer and never
# reach the platform's log viewer — which is precisely when you need it, since
# the visitor only ever sees "the assistant is temporarily unavailable" and the
# real reason lives in this one line. logging writes through the handler
# configured in compass/settings.py and is not subject to that buffering.
log = logging.getLogger("compass.answer")

# Overridable so you can switch models without editing code. gpt-oss-120b on
# Groq is fast and free-tier friendly; the OpenAI-compatible client means
# pointing at another provider is a base_url change, not a rewrite.
LLM_MODEL = os.environ.get("COMPASS_LLM_MODEL", "openai/gpt-oss-120b")

# How many chunks the model gets to choose between. Three is the sweet spot:
# one candidate means a near-miss has no alternative to lose to, and more than
# three bloats the prompt (so: slower, and more tokens against a free tier).
N_CANDIDATES = 3

# Characters of each candidate's content included in the prompt. Enough to
# answer from, short enough to keep three candidates cheap.
MAX_CHARS = 600

# ---- GATE 1: the confidence floor ------------------------------------------
# If the best retrieval score is below this, we refuse WITHOUT calling the LLM
# at all. Free, instant, and — the important part — incapable of hallucinating,
# because no model is involved in the decision.
#
# It is tunable because it trades the two failure modes against each other:
# raise it and the bot says "I don't know" to things it could have answered;
# lower it and it starts answering from chunks that are not really about the
# question. For a bot on a public website, the second failure is much worse,
# so this errs toward refusing.
CONFIDENCE_FLOOR = 0.30

SYSTEM = """You are a helpful assistant for one event's website. You are given a visitor's question and NUMBERED candidate sections of information.

Your job is to ANSWER the visitor's question, using only the single best candidate.

Rules:
- Use ONLY the candidates' text. Never use outside knowledge. Never state a fact that does not appear in the candidate you picked.
- If no candidate is about the requested topic, set index to 0.
- Answer the visitor directly, leading with the answer itself.
- For a yes/no question, the FIRST WORD must be the truthful answer to that question, not an acknowledgement. "Yes" means the thing asked about IS the case; "No" means it is not. Example: asked "is there a registration fee?" when entry is free, the answer begins "No" — beginning with "Yes" there would state the opposite of the truth.
- Never describe the section, the page or the website's structure. Never write "this section", "it states", "it explains", "the page says", "according to", or "confirming".
- Never mention candidates, numbers, scores, or that you were given a choice.
- Keep the explanation under 30 words, one or two sentences.
- Do not hedge. No "likely", "probably", "may".

Respond with ONLY a JSON object:
{"index": <candidate number, or 0 to reject all>, "explanation": "..."}"""

# WHY THE BANNED PHRASINGS ARE LISTED EXPLICITLY, and why the yes/no rule is
# three sentences long: both are copied from real failures.
#
# An earlier prompt said only "describe the destination", and produced:
#     "It explains OpenLake has no membership fee, confirming membership is free"
# Accurate, grounded, and nothing like how a person answers.
#
# A later version said 'if the question is yes/no, begin with "Yes" or "No"',
# and produced:
#     Q: "is there a membership fee?"
#     A: "Yes, OpenLake is free to join with no membership fee."
# Every fact correct; the first word states the opposite of all of them. An
# instruction about output FORMAT is not an instruction about output TRUTH.
#
# The general lesson: when a prompt misbehaves, feed its own bad output back as
# an explicit prohibition. Naming the specific failure beats a general
# instruction to "be direct", because the model cannot guess which phrasings
# you consider meta.

USER_TEMPLATE = """CANDIDATES:
{candidates}

VISITOR QUESTION: {query}"""


def _build_candidates(results):
    """Render (chunk, score) pairs into the numbered block the prompt expects.

    Numbered from 1, not 0, so that 0 is free to mean "reject all" — which
    makes the refusal unambiguous rather than overloading a valid index.
    Scores are NOT included: the model's job is to judge relevance from the
    text, and showing it our numbers just invites it to defer to them.
    """
    lines = []
    for i, (c, _score) in enumerate(results, start=1):
        lines.append(f"{i}. HEADING: {c['heading']}\n   TEXT: {c['content'][:MAX_CHARS]}")
    return "\n\n".join(lines)


def answer(query, chunks, vecs, llm, floor=CONFIDENCE_FLOOR):
    """Full question -> response pipeline.

    The LLM client is passed IN rather than built here — dependency injection.
    Constructing one reads env vars and sets up a connection pool, so doing it
    per request would be wasteful; and passing it in makes this function
    testable with a fake.

    Returns a dict that is always one of two shapes:
        {"found": True,  "heading", "explanation", "confidence", "url"}
        {"found": False, "reason", "score"}
    """
    if not chunks or vecs is None:
        return {"found": False,
                "reason": "nothing has been indexed yet", "score": 0.0}

    results = search(query, chunks, vecs, k=N_CANDIDATES)
    if not results:
        return {"found": False, "reason": "nothing has been indexed yet",
                "score": 0.0}

    # Gate 1 looks at the TOP score only: if even the best candidate is weak,
    # there is nothing worth showing the model.
    _, score = results[0]
    if score < floor:
        return {"found": False,
                "reason": "I don't have information about that.",
                "score": score}

    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                # The system role carries the rules; the user role carries the
                # DATA. Keeping them separate is basic prompt hygiene and the
                # first line of defence against prompt injection.
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(
                    candidates=_build_candidates(results), query=query)},
            ],
            # JSON mode constrains sampling so the output is valid JSON — no
            # ```json fences, no "Sure! Here you go:" preamble to strip.
            response_format={"type": "json_object"},
            # Deterministic. For a classification-shaped task, creativity is
            # not a feature: the same question should get the same answer.
            temperature=0,
            max_tokens=200,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as e:
        # Every one of these has actually happened on this project: a model id
        # decommissioned (404), a free-tier rate limit (429), JSON mode
        # truncated by max_tokens (400). Previously any of them propagated out
        # as an HTTP 500 the front end could not parse. A widget should degrade
        # to "temporarily unavailable", never to a broken response.
        # exc_info=True attaches the traceback. The exception TYPE is the
        # whole diagnosis here: AuthenticationError means a bad or revoked
        # key, NotFoundError means the model id is not available to this
        # account, RateLimitError means the budget is spent. All three reach
        # the visitor as the same sentence, so this log is the only place the
        # difference is recorded.
        log.error("LLM call failed: %s: %s", type(e).__name__, e, exc_info=True)
        return {"found": False,
                "reason": "the assistant is temporarily unavailable",
                "score": score}

    # ---- GATE 2: the model's own refusal --------------------------------
    # int() guards against the model returning "2" as a string; the try/except
    # guards against something that is not a number at all. A malformed index
    # is treated as a refusal, which is the safe direction to fail.
    try:
        idx = int(parsed.get("index", 0))
    except (TypeError, ValueError):
        idx = 0

    if idx < 1 or idx > len(results):
        return {"found": False,
                "reason": "I don't have information about that.",
                "score": score}

    chunk, chunk_score = results[idx - 1]
    return {
        "found": True,
        "heading": chunk["heading"],
        "explanation": parsed.get("explanation", ""),
        "confidence": chunk_score,
        # Empty string when the fact has no page of its own. The widget uses it
        # to offer a link and simply omits it when absent.
        "url": chunk.get("url", ""),
    }
