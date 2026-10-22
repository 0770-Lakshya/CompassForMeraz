from django.apps import AppConfig


class AskConfig(AppConfig):
    """Deliberately empty. The startup work lives in compass/boot.py — see the
    long note there on why AppConfig.ready() is the wrong home for it."""
    default_auto_field = "django.db.models.BigAutoField"
    name = "ask"
