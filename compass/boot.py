"""
 compass/boot.py — expensive startup work, done once.

Loads the embedder and the index so the first visitor is not the one who waits
a second for a 90MB model.

WHY NOT AppConfig.ready()? Because ready() runs for EVERY management command —
including `manage.py ingest` itself, and `check`, and `shell`. Those would each
load a model they do not need, and would each refuse to start if GROQ_API_KEY
were unset. Instead this is called from wsgi.py, which only the web server
imports, plus defensively from the view where it is a no-op after the first
call.
"""

import logging
import os
import threading

log = logging.getLogger()

_lock = threading.Lock()
_warmed = False


def warm():
    """Load the embedder, the index and the LLM client. Idempotent."""
    global _warmed

    # Fast path outside the lock: after the first call this runs on every
    # request, and taking a mutex each time would serialise everything for no
    # reason. Reading a bool is atomic in CPython.
    if _warmed:
        return

    with _lock:
        # Re-check inside: several threads may have passed the check above
        # before any of them acquired the lock.
        if _warmed:
            return

        # Imported here, not at module scope, so `manage.py check` does not
        # have to resolve numpy and fastembed just to validate settings.
        from groq import Groq
        from core.retrieval import load, model

        from ask.state import STATE

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            # Fail fast and loudly. Better to refuse to boot than to accept
            # traffic and 500 on the first real question — that turns a config
            # mistake you would catch in ten seconds into a mystery outage.
            raise RuntimeError("Set GROQ_API_KEY before starting the server.")

        log.info("warming embedder + loading index...")
        model()                       # force the lazy singleton now, on our time

        chunks, vecs = load()
        STATE["chunks"], STATE["vecs"] = chunks, vecs

        # One client, so TCP and TLS handshakes are not repeated per request.
        # The timeout and retry values are a latency decision, not boilerplate:
        # on a free tier where 429s are routine, the SDK's default
        # max_retries=2 with backoff turns one slow call into three sequential
        # attempts, which a visitor experiences as the widget hanging.
        STATE["llm"] = Groq(api_key=key, timeout=8.0, max_retries=1)

        _warmed = True
        if chunks:
            log.info("ready. %d chunk(s) indexed.", len(chunks))
        else:
            log.warning("ready, but NOTHING IS INDEXED. "
                        "Run: python manage.py ingest content/meraz.txt")
