"""
 ask/state.py — the server's whole in-memory database.

    STATE["chunks"] = list of chunk dicts
    STATE["vecs"]   = (N, 384) numpy matrix, row i corresponds to chunks[i]
    STATE["llm"]    = the shared Groq client

A module-level dict rather than globals, because a dict can be MUTATED from
inside a function without the `global` keyword (STATE["x"] = y mutates;
x = y would rebind), and it keeps all mutable server state in one greppable
place.

Also here: a small LRU cache for answers.

WHY A CACHE EARNS ITS PLACE, when caches are usually premature. An event FAQ
bot gets asked the same dozen questions over and over — when does it start,
where is it, is there a fee. That query distribution is tiny and extremely
repetitive, which is exactly the shape where a cache pays off:

  LATENCY     a hit answers in microseconds instead of ~600ms.
  RATE LIMITS Groq's free tier allows 8,000 tokens per minute and one query
              costs several hundred, so the server can serve only a handful of
              uncached questions a minute before it starts returning 429s.
              Cached answers cost zero tokens.

OrderedDict gives O(1) move-to-end and popitem(last=False), which is all an LRU
needs. The bound matters: an unbounded dict keyed on visitor-supplied strings
is a memory-exhaustion vector.
"""

import threading
from collections import OrderedDict

STATE = {"chunks": [], "vecs": None, "llm": None}

_CACHE_MAX = 500
_cache = OrderedDict()
_lock = threading.Lock()


def _key(query):
    """Normalise so trivial variations share one entry.

    "When is Meraz?" / "when is meraz" / "  WHEN IS MERAZ " collapse to one
    key. .split() with no argument splits on arbitrary runs of whitespace and
    drops empties, so re-joining with single spaces normalises internal spacing
    too — cheaper and more robust than a regex.
    """
    return " ".join(query.lower().split())


def cache_get(query):
    with _lock:
        k = _key(query)
        if k in _cache:
            _cache.move_to_end(k)      # mark most-recently-used
            return _cache[k]
    return None


def cache_put(query, value):
    with _lock:
        _cache[_key(query)] = value
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)  # pop the FRONT = least recently used


def cache_clear():
    """Called after a re-ingest: the index changed, so old answers may cite
    text that no longer exists."""
    with _lock:
        _cache.clear()
