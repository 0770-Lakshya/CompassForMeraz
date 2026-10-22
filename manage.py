#!/usr/bin/env python
"""Django's command-line entry point. Run everything from this folder."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "compass.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Could not import Django. Is your virtual environment active? "
            "Try: pip install -r requirements.txt"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
