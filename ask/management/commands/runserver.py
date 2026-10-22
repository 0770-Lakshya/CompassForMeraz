"""
 A `runserver` that defaults to port 8001 instead of 8000.

WHY THIS FILE EXISTS
--------------------
This project and the main Compass project are both Django, so both default to
port 8000. Running one while the other is up produces a genuinely nasty failure:
the port IS open and the server DOES answer, it just does not have the endpoint
the caller wants. Nothing says "wrong project", so you go looking for CORS bugs
and missing API keys.

The fix is to make the port a fact about the project rather than something you
have to remember to type. `python manage.py runserver` now binds 8001 here, and
an explicit `runserver 8000` still works when you want it.

HOW IT WORKS
------------
Django builds its command list by scanning every installed app for a
`management/commands/` folder. A command whose filename matches a built-in one
REPLACES it — so a file called runserver.py here shadows Django's.

TWO SUBTLETIES, both of which will bite if you copy this pattern:

 1. WE SUBCLASS staticfiles' RUNSERVER, NOT django.core's. django.contrib
    .staticfiles ships its own runserver override, the one that serves static
    files in development. Subclassing django.core's would silently throw that
    away, and in DEBUG the DRF browsable API would lose its CSS.

 2. "ask" MUST COME FIRST IN INSTALLED_APPS. Django resolves competing commands
    in `get_commands()` by iterating the apps in REVERSE order and updating a
    dict, so the app listed FIRST is applied last and wins. With "ask" last,
    staticfiles' runserver would win and this file would be dead code that
    looks like it should work — see the note in compass/settings.py.
"""

from django.contrib.staticfiles.management.commands.runserver import (
    Command as StaticfilesRunserver,
)


class Command(StaticfilesRunserver):
    default_port = "8001"
