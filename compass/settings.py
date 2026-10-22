"""
 compass/settings.py — configuration.

NO DATABASE, and that is not a shortcut. The entire index is two files in
data/, written by `manage.py ingest` and loaded into memory at startup. Adding
Postgres here would buy nothing: there is one body of knowledge, it is small,
and the only thing that changes it is you re-running the ingest command.

(The parent Compass project DOES need a database, for a different reason: it
holds a separate index per customer site and loads all of them into RAM at
once, so memory grows with every customer. That pressure does not exist here.)

Because there is no database, django.contrib.auth / admin / sessions are all
absent — they would fail at startup without one.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "dev-only-insecure-key-change-in-production")

# Environment variables are ALWAYS strings, so bool(os.environ.get("DEBUG"))
# would be True even for "False". Compare the string explicitly.
DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    # "ask" IS FIRST DELIBERATELY, and moving it would break something subtle.
    # ask/management/commands/runserver.py overrides Django's runserver to
    # default to port 8001. Django resolves competing management commands by
    # iterating the apps in REVERSE order and updating a dict, so the app
    # listed FIRST is applied LAST and wins. With "ask" at the bottom,
    # django.contrib.staticfiles' own runserver would win instead and the
    # override would be dead code that looks like it should work.
    "ask",
    "django.contrib.staticfiles",   # also gives DRF's browsable API its CSS
    "corsheaders",
    "rest_framework",
]

# Order matters. CorsMiddleware must come before CommonMiddleware so it can
# answer a browser's preflight OPTIONS request and return immediately.
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.security.SecurityMiddleware",
]

# The widget is embedded on a site served from a different origin than this
# API, so the browser blocks the response unless the server opts in. Without
# this, every fetch() fails with a CORS error — the number one thing that trips
# people up when building an embeddable widget.
#
# IN DEVELOPMENT: allow everything, so localhost:3000, localhost:5500 and a
# file:// page all work without thinking about it.
#
# IN PRODUCTION: an allowlist, and the reason is money rather than secrecy.
# There is nothing confidential in the answers — it is published fest
# information. But every uncached question spends from a free tier capped at
# 200,000 tokens a day, which is roughly 160 questions. Leave the API open and
# anyone who finds the URL can exhaust that budget in an afternoon, after which
# the bot stops answering FOR MERAZ. The allowlist is a cost control.
#
# Honest limit: CORS is enforced by BROWSERS. It stops a page on another
# domain from using this API; it does nothing against curl, which ignores the
# whole mechanism. Stopping that needs per-site keys or rate limiting by IP,
# neither of which exists yet. This raises the bar, it does not close the door.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    CORS_ALLOWED_ORIGINS = _origins
else:
    # No allowlist configured: fall back to open. Deliberate, so a missing env
    # var in development is not a mysterious CORS failure — but it does mean
    # production must set CORS_ORIGINS (render.yaml does).
    CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS"]

ROOT_URLCONF = "compass.urls"

# Django's default would 301 "/ask" to "/ask/". Browsers follow a 301 on a POST
# by re-issuing it as a GET WITHOUT the body — so the widget's POST would
# arrive as an empty GET and fail in a way that is genuinely hard to diagnose.
APPEND_SLASH = False

WSGI_APPLICATION = "compass.wsgi.application"
DATABASES = {}

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [], "APP_DIRS": True, "OPTIONS": {"context_processors": []},
}]
STATIC_URL = "static/"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    # REQUIRED with no django.contrib.auth installed. DRF populates
    # request.user on every request and falls back to AnonymousUser, and
    # importing that module pulls in contenttypes' models — which raises on
    # EVERY request, as a 500, with a traceback that mentions nothing about
    # authentication. Setting these to None says "there is no user concept
    # here", which is the truth.
    "UNAUTHENTICATED_USER": None,
    "UNAUTHENTICATED_TOKEN": None,
    "DEFAULT_RENDERER_CLASSES": (
        ["rest_framework.renderers.JSONRenderer",
         "rest_framework.renderers.BrowsableAPIRenderer"]
        if DEBUG else ["rest_framework.renderers.JSONRenderer"]),
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "[{levelname}] {message}", "style": "{"}},
    "handlers": {"console": {"class": "logging.StreamHandler",
                             "formatter": "simple"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
