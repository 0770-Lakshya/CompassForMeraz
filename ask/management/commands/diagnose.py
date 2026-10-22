"""
 manage.py diagnose "your question"

Prints every intermediate value between a question and an answer, so a refusal
can be traced to its cause instead of guessed at.

There are exactly two reasons the bot says "I don't have information about
that", and they need opposite fixes:

  GATE 1  the best retrieval score is below CONFIDENCE_FLOOR. The LLM is never
          called. Cause is retrieval: wrong/stale embeddings, or the question
          genuinely does not match any chunk.

  GATE 2  retrieval was fine and the LLM chose index 0, meaning "none of these
          candidates answer the question". Cause is the prompt, the model, or
          candidates that look relevant by score but are not by content.

This command shows the scores, the floor, the candidates actually sent, and the
model's raw reply — so which gate fired is never in doubt.
"""

import json
import os

from django.core.management.base import BaseCommand

from core import answer as answer_mod
from core.retrieval import (CHUNKS_PATH, VECS_PATH, build_text, embed_query,
                            lexical_target, load, strip_filler)


class Command(BaseCommand):
    help = "Trace a question through retrieval and the LLM call."

    def add_arguments(self, parser):
        parser.add_argument("question", nargs="?",
                            default="how much is a pass")
        parser.add_argument("--no-llm", action="store_true",
                            help="stop after retrieval; makes no API call")

    def handle(self, *args, **opts):
        import numpy as np

        q = opts["question"]
        w = self.stdout.write

        # ---- 1. what is on disk -----------------------------------------
        w(self.style.MIGRATE_HEADING("\n1. INDEX ON DISK"))
        w(f"   chunks.json     : {CHUNKS_PATH}")
        w(f"   exists          : {CHUNKS_PATH.exists()}")
        w(f"   embeddings.npy  : {VECS_PATH.exists()}")

        chunks, vecs = load()
        w(f"   chunks loaded   : {len(chunks)}")
        if vecs is None:
            w(self.style.ERROR("   NO VECTORS — nothing was embedded."))
            return
        w(f"   vectors shape   : {vecs.shape}")

        # A matrix of zeros means the embedder produced nothing usable, which
        # kills the semantic half of the search silently — every score then
        # comes from fuzzy matching alone and most questions fall below the
        # floor. Worth checking explicitly because it looks like no error.
        w(f"   vector norm[0]  : {float(np.linalg.norm(vecs[0])):.4f} "
          f"(should be ~1.0)")
        if float(np.abs(vecs).max()) == 0.0:
            w(self.style.ERROR("   ALL VECTORS ARE ZERO — the embedder is not "
                               "working. Re-run: manage.py ingest"))

        if len(vecs) != len(chunks):
            w(self.style.ERROR(
                f"   MISMATCH: {len(vecs)} vectors vs {len(chunks)} chunks. "
                "Re-run: manage.py ingest"))

        # ---- 2. the question --------------------------------------------
        w(self.style.MIGRATE_HEADING("\n2. QUESTION"))
        w(f'   raw             : "{q}"')
        w(f'   after filler    : "{strip_filler(q)}"   (used by the FUZZY half only)')

        # ---- 3. the three signals, per chunk ----------------------------
        from core.retrieval import (LEVEL_BONUS, W_LEXICAL, W_SEMANTIC)
        from rapidfuzz import fuzz

        qv = embed_query(q)
        sem = vecs @ qv
        lean = strip_filler(q)
        lex = np.array([fuzz.token_set_ratio(lean, lexical_target(c).lower()) / 100.0
                        for c in chunks])
        bonus = np.array([LEVEL_BONUS.get(c.get("level", "h2"), 0.0) for c in chunks])
        combined = W_SEMANTIC * sem + W_LEXICAL * lex + bonus
        order = np.argsort(-combined)[:5]

        w(self.style.MIGRATE_HEADING("\n3. TOP 5 BY SCORE"))
        w(f"   {'total':>7} {'sem':>7} {'lex':>7} {'bonus':>7}   heading")
        for i in order:
            w(f"   {combined[i]:7.3f} {sem[i]:7.3f} {lex[i]:7.3f} "
              f"{bonus[i]:7.3f}   {chunks[i]['heading'][:46]}")

        # ---- 4. gate 1 ---------------------------------------------------
        floor = answer_mod.CONFIDENCE_FLOOR
        best = float(combined[order[0]])
        w(self.style.MIGRATE_HEADING("\n4. GATE 1 — confidence floor"))
        w(f"   best score      : {best:.3f}")
        w(f"   floor           : {floor:.3f}")
        if best < floor:
            w(self.style.ERROR(
                "   REFUSED HERE. The LLM was never called.\n"
                "   -> retrieval is the problem, not the prompt. Check section 1 "
                "for zero/mismatched vectors, and section 3 for whether the "
                "right heading is even in the top 5."))
            return
        w(self.style.SUCCESS("   PASSED — the LLM gets called."))

        if opts["no_llm"]:
            w("\n--no-llm: stopping before the API call.")
            return

        # ---- 5. the LLM call --------------------------------------------
        w(self.style.MIGRATE_HEADING("\n5. GATE 2 — the LLM call"))
        w(f"   model           : {answer_mod.LLM_MODEL}")
        w(f"   GROQ_API_KEY    : {'set' if os.environ.get('GROQ_API_KEY') else 'MISSING'}")
        # Printed because a token budget too small for a reasoning model is
        # invisible otherwise, and it is the failure that brought this command
        # into existence.
        w(f"   params          : {answer_mod.LLM_PARAMS}")

        from groq import Groq
        llm = Groq(api_key=os.environ.get("GROQ_API_KEY"),
                   timeout=15.0, max_retries=1)

        results = [(chunks[i], float(combined[i]))
                   for i in np.argsort(-combined)[:answer_mod.N_CANDIDATES]]
        candidates = answer_mod._build_candidates(results)
        w("\n   CANDIDATES SENT:")
        for line in candidates.splitlines():
            w(f"     {line[:110]}")

        try:
            # **answer_mod.LLM_PARAMS, NOT a second copy of those values.
            #
            # This line is the whole reason LLM_PARAMS exists. The first version
            # of this command inlined its own response_format / temperature /
            # max_tokens — so when max_tokens was fixed in answer.py, this
            # diagnostic kept making the OLD call and kept reporting the old
            # 400, which read as "the fix did not work". The tool built to find
            # the bug had a different bug.
            #
            # A diagnostic must exercise the SAME code path as production, or
            # it is measuring something else and telling you about that.
            resp = llm.chat.completions.create(
                model=answer_mod.LLM_MODEL,
                messages=[
                    {"role": "system", "content": answer_mod.SYSTEM},
                    {"role": "user", "content": answer_mod.USER_TEMPLATE.format(
                        candidates=candidates, query=q)},
                ],
                **answer_mod.LLM_PARAMS,
            )
        except Exception as e:
            # The three that actually happen: a decommissioned model id (404),
            # a free-tier rate limit (429), JSON mode truncated (400).
            w(self.style.ERROR(f"\n   LLM CALL FAILED: {type(e).__name__}: {e}"))
            w(self.style.ERROR(
                "   -> this surfaces to users as 'the assistant is temporarily "
                "unavailable', NOT as 'I don't have information about that'."))
            return

        raw = resp.choices[0].message.content
        w(f"\n   RAW REPLY       : {raw}")

        try:
            parsed = json.loads(raw)
            idx = int(parsed.get("index", 0))
        except Exception as e:
            w(self.style.ERROR(f"   UNPARSEABLE ({e}) — treated as a refusal."))
            return

        if idx < 1 or idx > len(results):
            w(self.style.ERROR(
                f"   index={idx} -> REFUSED BY THE MODEL.\n"
                "   -> retrieval was fine. The model judged none of the "
                "candidates to answer the question. Fix the prompt, the model, "
                "or the content — not the embeddings."))
            return

        w(self.style.SUCCESS(
            f"\n   ANSWERED from candidate {idx}: "
            f"{results[idx-1][0]['heading']}"))
        w(f"   explanation     : {parsed.get('explanation')}")
