"""ask/urls.py — no trailing slashes, matching APPEND_SLASH = False."""
from django.urls import path

from ask import views

urlpatterns = [
    path("health", views.health, name="health"),
    path("ask", views.ask, name="ask"),
]
