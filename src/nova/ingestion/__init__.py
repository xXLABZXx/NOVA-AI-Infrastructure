"""NOVA ingestion: append-only raw storage plus text cleaning."""

from nova.ingestion.clean import (
    clean_document,
    detect_language,
    filter_by_language,
    normalize_text,
    strip_boilerplate,
)
from nova.ingestion.store import RawStore

__all__ = [
    "RawStore",
    "normalize_text",
    "strip_boilerplate",
    "detect_language",
    "filter_by_language",
    "clean_document",
]
