"""
===============================================================================
 core/text.py — turn a plain text file into chunks.
===============================================================================

This replaces the crawler, the HTML chunker and the indexer with about forty
lines, because the hard part of those was never the chunking — it was working
out where one meaningful section of a stranger's HTML ends and the next begins,
and generating a CSS selector that would still resolve months later.

When a human writes the knowledge, they mark the boundaries themselves with
headings. That is the whole problem, solved by asking.

THE FORMAT
----------
    title: Meraz 7.0                 optional, once, at the top

    # When is Meraz?                 a heading starts a chunk (level h1)
    Meraz 7.0 runs from 5 to 7       body text until the next heading
    February 2027.

    ## Where do I register?          ## is h2, ### is h3
    url: https://meraz.in/register   optional, FIRST line under the heading
    Registration is on the website.

Rules, kept few on purpose:
  * A heading starts a chunk. Text before the first heading is ignored, so
    notes at the top of the file are fine.
  * `url:` must be the first line under its heading or it is treated as body.
  * A heading with no body is dropped. An empty chunk can still win a
    similarity match and then have nothing to say — a confident empty answer,
    which is worse than no answer.
"""

import re
import sys

# ^\s* tolerates accidental leading whitespace, which is the most common way a
# hand-written file silently fails to parse.
HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*\S)\s*$")
URL_RE = re.compile(r"^\s*url\s*:\s*(\S+)\s*$", re.IGNORECASE)
TITLE_RE = re.compile(r"^\s*title\s*:\s*(.*\S)\s*$", re.IGNORECASE)


def parse(text, default_title="Knowledge base", warn=None):
    """Text in, list of chunk dicts out. No I/O, so it is trivial to test.

    `warn` is an optional callable for messages, so a Django management command
    can route them through self.stderr instead of print.
    """
    warn = warn or (lambda m: print(m, file=sys.stderr))

    page_title = default_title
    raw_chunks = []
    current = None          # the chunk being built; None before the first heading

    for line in text.splitlines():
        if current is None:
            m = TITLE_RE.match(line)
            if m:
                page_title = m.group(1)
                continue

        m = HEADING_RE.match(line)
        if m:
            if current is not None:
                raw_chunks.append(current)
            hashes, heading = m.group(1), m.group(2)
            current = {
                "heading": heading,
                # "h1".."h6", matching what LEVEL_BONUS in retrieval.py expects.
                "level": f"h{len(hashes)}",
                "url": "",
                "body": [],
            }
            continue

        if current is None:
            continue        # a note above the first heading

        # `not current["body"]` is what makes "first line" true. A later line
        # starting with "url:" is body text — which is what you would want if a
        # fact is literally about a URL.
        if not current["body"]:
            m = URL_RE.match(line)
            if m:
                current["url"] = m.group(1)
                continue

        current["body"].append(line)

    if current is not None:
        raw_chunks.append(current)

    out = []
    for c in raw_chunks:
        # join-then-strip collapses blank lines at the ends while keeping
        # paragraph breaks inside the body.
        content = "\n".join(c["body"]).strip()
        if not content:
            warn(f"  skipped (no body): {c['heading']}")
            continue
        out.append({
            # `id` MUST equal this chunk's row index in the embeddings matrix,
            # so it is assigned here, AFTER filtering.
            "id": len(out),
            "heading": c["heading"],
            "level": c["level"],
            "url": c["url"],
            "content": content,
            "page_title": page_title,
        })
    return out
