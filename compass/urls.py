"""compass/urls.py — the root URL map."""
from django.urls import include, path

urlpatterns = [
    path("", include("ask.urls")),
]
