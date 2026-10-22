"""
 manage.py ingest content/meraz.txt

A Django MANAGEMENT COMMAND rather than a standalone script, which is worth
understanding because it is the idiomatic way to add a CLI task to a Django
project.

Putting a file at <app>/management/commands/<name>.py makes `manage.py <name>`
work automatically — no registration, no entry point. Django finds it by path
convention. In return it hands you settings already loaded, the app registry
ready, and argparse wired to --help.

WHY THIS IS A COMMAND AND NOT AN HTTP ENDPOINT:
the parent project uploads chunks over POST /register, because its crawler runs
on a laptop with Playwright and the server must never run a browser. Here the
source of truth is a text file that lives in this repo, so there is no network
hop to make — and no admin token to manage, no endpoint to secure.

THE TRADE-OFF, STATED PLAINLY: the running server holds the index in memory, so
it will not see a fresh ingest until it restarts. For a file you edit by hand a
handful of times, a restart is cheaper than building a reload endpoint and the
auth to protect it. If that stops being true, add POST /reload.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core.retrieval import embed_all
from core.text import parse


class Command(BaseCommand):
    help = "Index a plain text file so the bot can answer from it."

    def add_arguments(self, parser):
        parser.add_argument("file", help="the .txt file to ingest")
        parser.add_argument("--dry-run", action="store_true",
                            help="parse and print, write nothing")

    def handle(self, *args, **opts):
        path = Path(opts["file"])
        if not path.exists():
            # CommandError prints a clean message and exits non-zero, rather
            # than dumping a traceback at someone who mistyped a filename.
            raise CommandError(f"no such file: {path}")

        chunks = parse(
            path.read_text(encoding="utf-8"),
            default_title=path.stem,
            warn=lambda m: self.stderr.write(m),
        )

        if not chunks:
            raise CommandError(
                "no chunks produced — does the file have '# Heading' lines "
                "with text underneath them?")

        self.stdout.write(f"\nparsed {len(chunks)} chunk(s) from {path.name}:\n")
        for c in chunks:
            nav = f"  -> {c['url']}" if c["url"] else ""
            self.stdout.write(
                f"  [{c['id']:3}] {c['level']}  {c['heading'][:50]:50}"
                f"  {len(c['content']):5} chars{nav}")

        if opts["dry_run"]:
            self.stdout.write(self.style.WARNING("\n--dry-run: nothing written."))
            return

        # embed_all writes BOTH chunks.json and embeddings.npy together, which
        # is what keeps them consistent — the "row i is chunks[i]" invariant
        # cannot drift if the only way to change either file writes both.
        embed_all(chunks)

        self.stdout.write(self.style.SUCCESS(
            f"\nindexed {len(chunks)} chunk(s) into data/"))
        self.stdout.write(
            "RESTART the server to pick this up — the index is held in memory.")
