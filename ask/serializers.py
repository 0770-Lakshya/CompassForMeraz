"""
 ask/serializers.py — request validation.

DRF needs an explicit field object per attribute, where Pydantic would infer
it from a type hint. Two rules worth remembering, because both bite:

  * CharField REJECTS "" by default, so allow_blank=True is needed wherever an
    empty string should be accepted rather than 400.
  * CharField TRIMS whitespace by default. Harmless for a question; it would be
    corruption for anything content-shaped.
"""

from rest_framework import serializers


class AskInSerializer(serializers.Serializer):
    # allow_blank so an empty question reaches the view, which returns a clean
    # 400 with its own message rather than DRF's field-error shape.
    question = serializers.CharField(allow_blank=True, max_length=500)


class AskOutSerializer(serializers.Serializer):
    """Guarantees a STABLE response shape.

    Without this, a refusal would omit `heading` and `explanation` entirely and
    a success would omit `reason`, so the widget's JavaScript would have to
    check which keys exist. One shape means one code path: check `found`, then
    read the fields.
    """
    found = serializers.BooleanField()
    heading = serializers.CharField(required=False, allow_null=True, default=None)
    explanation = serializers.CharField(required=False, allow_null=True, default=None)
    confidence = serializers.FloatField(required=False, allow_null=True, default=None)
    url = serializers.CharField(required=False, allow_null=True, default=None)
    reason = serializers.CharField(required=False, allow_null=True, default=None)
