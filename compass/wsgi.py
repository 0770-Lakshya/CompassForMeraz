"""compass/wsgi.py — what the production server imports.

WSGI rather than ASGI because every view here is a plain `def` that awaits
nothing. Running sync views under WSGI with a thread pool gives the same
concurrency with a simpler stack.
"""
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "compass.settings")

from django.core.wsgi import get_wsgi_application  # noqa: E402

application = get_wsgi_application()

# AFTER get_wsgi_application(), because that call is what initialises Django's
# settings and app registry, and warm() imports modules that expect both.
from compass.boot import warm  # noqa: E402

warm()
