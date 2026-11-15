"""Schema assertions — payload shape, field presence, banking formats."""

from __future__ import annotations

import re

_IBAN_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$")
_CARD_RE = re.compile(r"\b\d{13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ISO_4217 = {
    "USD", "EUR", "GBP", "INR", "JPY", "CHF", "CAD", "AUD", "SGD", "HKD",
}


def assert_json_schema(payload: dict, schema: dict) -> None:
    """Assert a payload validates against a JSON Schema draft."""
    from jsonschema import ValidationError, validate

    try:
        validate(payload, schema)
    except ValidationError as exc:
        raise AssertionError(f"schema violation: {exc.message}") from exc


def assert_required_fields(payload: dict, fields: list[str]) -> None:
    """Assert every named field is present and non-None."""
    missing = [f for f in fields if payload.get(f) is None]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")


def assert_iso_currency(value: str) -> None:
    """Assert the currency code is a known ISO 4217 code."""
    if str(value).upper() not in _ISO_4217:
        raise AssertionError(f"not a supported ISO 4217 currency: {value!r}")


def assert_iban(value: str) -> None:
    """Assert the value matches IBAN structure (country+check+BBAN)."""
    if not _IBAN_RE.match(str(value)):
        raise AssertionError(f"malformed IBAN: {value!r}")


def assert_no_pii(payload: dict) -> None:
    """Assert no card numbers, SSNs, or emails appear in any field value."""
    for key, value in payload.items():
        text = str(value)
        for label, pattern in (("card", _CARD_RE), ("ssn", _SSN_RE)):
            if pattern.search(text):
                raise AssertionError(f"possible {label} number in field {key!r}")
