"""
===============================================================================
 core/retrieval.py — embed the knowledge, then find the right piece of it.
===============================================================================

Adapted from the Compass retrieval layer, with two things removed:

  * MULTI-SITE. Compass holds a separate index per customer website, keyed by
    site_id, with a folder each. This bot serves ONE body of knowledge, so the
    index is a single pair of files in data/ and every function that took a
    site_id lost it.

  * THE URL SLUG SIGNAL. Compass embedded each chunk's URL path ("/programs"
    -> "programs") because a crawled chunk's address carries real meaning. Here
    the knowledge is typed by hand and mostly has no URL, so that signal would
    be the same constant for every chunk — adding tokens while adding no
    ability to tell chunks apart.

Everything that does the actual work is unchanged, including the tuned
constants. Those weights were arrived at by testing against real queries, and
re-guessing them in a new project would be throwing away the only part of this
that cannot be re-derived from first principles.
"""

import json
import os
from pathlib import Path

import numpy as np

# rapidfuzz: fast fuzzy string matching in C++ (the faster successor to
# fuzzywuzzy).  read more: https://rapidfuzz.github.io/RapidFuzz/
from rapidfuzz import fuzz

# fastembed runs sentence-transformer models through ONNX Runtime instead of
# PyTorch — no torch dependency, ~90MB instead of ~2GB, and it fits in a small
# container.  read more: https://qdrant.github.io/fastembed/
from fastembed import TextEmbedding

# ---- where things live -----------------------------------------------------
# Anchored to __file__, never to the working directory. A bare Path("data")
# would resolve against wherever you happened to run the command from, and the
# failure mode is the worst kind: Python silently creates a second, empty data/
# folder, the bot answers "I don't know" to everything, and nothing errors.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
VECS_PATH = DATA_DIR / "embeddings.npy"

MODEL_CACHE = Path(os.environ.get("FASTEMBED_CACHE") or ROOT / ".model_cache")

# ---- the embedding model ---------------------------------------------------
# all-MiniLM-L6-v2: 6 layers, ~22M parameters, ~90MB, 384 dimensions. The
# standard "small and good enough" sentence embedder. Larger models score a few
# percent better on benchmarks and cost several times the memory and latency.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- the blend weights -----------------------------------------------------
# These sum to 1.0, which is what keeps the combined score on a roughly 0..1
# scale — and that is what makes CONFIDENCE_FLOOR in answer.py a meaningful,
# tunable number rather than a magic constant.
#
# Semantic gets the larger share because most real questions are paraphrases
# ("is it free to join" vs a heading reading "Membership and fees"), which is
# exactly what embeddings are good at. Lexical earns its 0.35 by catching the
# cases embeddings fumble: exact names, acronyms and typos.
W_SEMANTIC = 0.65
W_LEXICAL = 0.35

# A small thumb on the scale for higher-level headings. An h1 is a broader
# answer than an h3 buried inside it, so when two chunks score alike the
# broader one is usually the better reply.
LEVEL_BONUS = {"h1": 0.08, "h2": 0.0, "h3": -0.02}

# ---- stopwords, tuned for how people actually ask --------------------------
# Alongside ordinary grammar words this includes asking verbs — "tell", "show",
# "find", "know" — so that "can you tell me what the registration fee is"
# reduces to "registration fee", which is the string we want to fuzzy-match
# against heading text.
#
# A set, not a list, so `w not in FILLER` is an O(1) hash lookup.
FILLER = {"where", "is", "are", "the", "all", "of", "a", "an", "to",
          "me", "take", "show", "find", "i", "want", "can", "how",
          "do", "on", "in", "at", "page", "please", "what", "tell",
          "know", "you", "there", "any", "about", "does", "did", "it"}

_model = None


def model():
    """Load the ONNX embedder once, lazily.

    The singleton / lazy-initialisation pattern. Loading means reading ~90MB
    from disk and starting an ONNX session — about a second, plus permanent
    RAM. It must happen exactly once per process:
      * not at import time, or merely importing this module in a test pays for
        a model nobody is going to use;
      * not per request, or every query costs a second.

    `global _model` is required: without it, `_model = ...` would create a new
    function-local variable and the cache would never persist.
    """
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(MODEL_CACHE))
    return _model


def _encode(texts):
    """List of strings -> (N, 384) matrix of unit-length vectors.

    The normalisation is the part that matters. What we want is cosine
    similarity:

        cos(a, b) = (a . b) / (|a| * |b|)

    That division is expensive to repeat for every chunk on every query. But
    pre-scale every vector so |a| = 1 and the denominator becomes 1, and the
    formula collapses to a plain dot product — so the whole search becomes one
    matrix multiply. We pay once at index time and get exact cosine similarity
    free on every query thereafter.

    float32 rather than NumPy's default float64 halves the memory; at 384
    dimensions the precision difference is irrelevant to ranking.
    """
    vecs = np.array(list(model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # np.clip guards against a zero-length vector producing a divide-by-zero.
    return vecs / np.clip(norms, 1e-8, None)


def embed_query(query):
    """Embed one query and return a flat (384,) vector.

    `_encode` always returns 2-D, so a single string gives shape (1, 384). The
    [0] unwraps it to the shape that `vecs @ q` expects.
    """
    return _encode([query])[0]


def strip_filler(query):
    """Reduce a question to its content words.

        "can you tell me if there is a registration fee"  ->  "registration fee"

    THIS ONLY AFFECTS THE LEXICAL HALF of the search. Fuzzy similarity is
    computed over the whole string, so filler dilutes it: matching that full
    sentence against the heading "Registration fees" scores poorly because most
    of the words are noise. Matching "registration fee" against it scores ~100.

    The embedding model, trained on natural sentences, handles filler perfectly
    well — so it deliberately receives the ORIGINAL untouched query.

    The `or query.lower()` fallback catches a question made entirely of filler
    ("where is it"), where the comprehension yields nothing and "".join gives
    an empty string. Fuzzy-matching an empty string returns garbage.
    """
    words = [w for w in query.lower().split() if w not in FILLER]
    return " ".join(words) or query.lower()


def build_text(c):
    """The text that actually gets EMBEDDED for a chunk.

        <page title> | <heading> | <heading> | <content>

    THE HEADING APPEARS TWICE, deliberately. An embedding is loosely a weighted
    average of the meaning of its input, so a 1500-character body drowns out a
    20-character heading. Repeating the heading doubles its influence — and the
    heading is the most question-shaped part of a chunk, so that is what we
    want. The trick is known as field weighting; BM25 implements the same idea
    with explicit per-field weights.
    """
    return (f"{c['page_title']} | {c['heading']} | "
            f"{c['heading']} | {c['content']}")


def lexical_target(c):
    """The text that fuzzy matching runs against.

    NOTICE WHAT IS MISSING: c['content'].

    `token_set_ratio` compares sets of words, and a long body contains so many
    words that it partially matches almost any query by coincidence. Including
    content would make every long chunk score high on everything — the lexical
    signal would stop being a signal.

    So the two halves of the search look at deliberately different text:
        semantic -> title + heading + heading + content   (rich, nuanced)
        lexical  -> title + heading                       (short, precise)
    """
    return f"{c['page_title']} {c['heading']}"


# ---------------------------------------------------------------------------
#  loading and embedding the single index
# ---------------------------------------------------------------------------

def embed_all(chunks):
    """Embed every chunk and save the matrix. The expensive operation.

    The list comprehension preserves input order, which preserves the invariant
    everything downstream relies on: ROW i OF THE MATRIX IS chunks[i]. Break
    that and search returns the right row number attached to the wrong text —
    wrong answers with no error anywhere.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"embedding {len(chunks)} chunk(s)...")
    vecs = _encode([build_text(c) for c in chunks])

    CHUNKS_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    # .npy is NumPy's own binary format: raw float32 bytes, far smaller than
    # JSON and near-instant to load.
    np.save(VECS_PATH, vecs)
    return chunks, vecs


def load():
    """Load chunks + vectors from data/. Returns ([], None) if nothing is indexed.

    THE CACHE-VALIDITY CHECK is the interesting line. A stale embeddings.npy is
    poison, because search assumes row i corresponds to chunks[i]. If someone
    edits chunks.json and we load an old matrix, every result silently points
    at the wrong text.

    `len(vecs) == len(chunks)` is cheap and imperfect: it catches any change in
    the number of chunks, which is nearly every real edit, and would miss an
    edit that altered wording while keeping the count identical. Since the only
    way to change this index is to re-run `manage.py ingest`, which always
    rewrites both files together, that residual risk is acceptable here.
    """
    if not CHUNKS_PATH.exists():
        return [], None

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if VECS_PATH.exists():
        vecs = np.load(VECS_PATH)
        if len(vecs) == len(chunks):
            return chunks, vecs
        print("embeddings.npy is stale (count mismatch) — re-embedding")
    return embed_all(chunks)


def search(query, chunks, vecs, k=3, debug=False):
    """Score every chunk against the query, return the top k as (chunk, score).

    Stateless on purpose: chunks and vecs are passed in rather than read from a
    global, which keeps this testable from a plain script with no server.
    """
    # ---- SIGNAL 1: semantic similarity ----------------------------------
    q = embed_query(query)
    # `@` is matrix multiplication. (N, 384) @ (384,) -> (N,): one score per
    # chunk in a single optimised BLAS call rather than an N-iteration Python
    # loop. Both sides are unit-length, so each result IS cosine similarity.
    sem = vecs @ q

    # ---- SIGNAL 2: fuzzy lexical similarity -----------------------------
    lean = strip_filler(query)
    # token_set_ratio treats each side as a SET of words and compares the
    # common words against the leftovers — the most forgiving of rapidfuzz's
    # scorers, and forgiveness is what we need: word order in a heading is
    # arbitrary and query length never matches target length. It still catches
    # typos, since the underlying comparison is edit-distance based.
    #
    # rapidfuzz returns 0..100, so divide by 100 to share a scale with cosine.
    # Two signals MUST share a scale before weighting them, or the weights lie.
    lex = np.array([
        fuzz.token_set_ratio(lean, lexical_target(c).lower()) / 100.0
        for c in chunks
    ])

    # ---- SIGNAL 3: the structural bonus ---------------------------------
    # .get(key, 0.0) rather than [key], so an unexpected level yields a neutral
    # 0.0 instead of raising KeyError.
    bonus = np.array([LEVEL_BONUS.get(c.get("level", "h2"), 0.0) for c in chunks])

    combined = W_SEMANTIC * sem + W_LEXICAL * lex + bonus

    # np.argsort returns indices that would sort ascending; NumPy has no
    # descending flag, so the idiom is to sort the NEGATED array.
    top = np.argsort(-combined)[:k]
    out = []
    for i in top:
        out.append((chunks[i], float(combined[i])))
        if debug:
            print(f"    [{combined[i]:.3f}] sem={sem[i]:.3f} lex={lex[i]:.3f}  "
                  f"{chunks[i]['heading'][:40]}")
    return out
