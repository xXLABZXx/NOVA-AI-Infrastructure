"""Hermetic tests for nova.ingestion and nova.deduplication.

Uses the ``synthetic_docs`` fixture from tests/conftest.py:
doc-1/doc-3 are exact duplicates, doc-4 is a near-duplicate of doc-2.
"""

import json

import pytest

from nova.deduplication import ExactDeduper, NearDeduper, SimHash
from nova.ingestion import (
    RawStore,
    clean_document,
    detect_language,
    filter_by_language,
    normalize_text,
    strip_boilerplate,
)


# ---------------------------------------------------------------------------
# (a) RawStore round-trip + append-only immutability of the raw file
# ---------------------------------------------------------------------------
def test_raw_store_append_load_round_trip(tmp_path, synthetic_docs):
    store = RawStore(tmp_path / "raw.jsonl")
    returned_ids = [store.append(doc) for doc in synthetic_docs]

    assert returned_ids == ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]
    assert store.count() == 5

    loaded = store.load_all()
    assert len(loaded) == 5
    for original, round_tripped in zip(synthetic_docs, loaded):
        assert round_tripped["id"] == original["id"]
        assert round_tripped["text"] == original["text"]
        assert round_tripped["url"] == original["url"]

    # iter_docs streams the same content lazily.
    assert [d["id"] for d in store.iter_docs()] == returned_ids


def test_raw_store_assigns_id_without_mutating_input(tmp_path):
    store = RawStore(tmp_path / "raw.jsonl")
    doc = {"url": "http://example.com/n", "text": "hello world"}
    new_id = store.append(doc)
    assert len(new_id) == 32
    assert "id" not in doc  # caller's dict untouched
    assert store.load_all()[0]["id"] == new_id


def test_raw_file_bytes_unchanged_by_cleaning(tmp_path, synthetic_docs):
    path = tmp_path / "raw.jsonl"
    store = RawStore(path)
    for doc in synthetic_docs:
        store.append(doc)
    before = path.read_bytes()

    for doc in store.load_all():
        clean_document(doc)  # must not touch the raw file or the input

    assert path.read_bytes() == before

    # RawStore is append-only: no update/delete API exists.
    for forbidden in ("update", "delete", "remove", "upsert", "replace"):
        assert not hasattr(store, forbidden)


# ---------------------------------------------------------------------------
# (b) exact dedupe
# ---------------------------------------------------------------------------
def test_exact_dedupe_flags_doc3_not_doc2(synthetic_docs):
    deduper = ExactDeduper()
    unique, duplicates = deduper.dedupe(synthetic_docs)

    assert {d["id"] for d in duplicates} == {"doc-3"}
    assert {d["id"] for d in unique} == {"doc-1", "doc-2", "doc-4", "doc-5"}

    # Fingerprint is stable and normalization-insensitive.
    assert (ExactDeduper.fingerprint("  Hello   World ")
            == ExactDeduper.fingerprint("hello world"))

    # Incremental API agrees.
    inc = ExactDeduper()
    assert not inc.is_duplicate("some text")
    inc.add("some text")
    assert inc.is_duplicate("some text")
    assert not inc.is_duplicate("some other text")


# ---------------------------------------------------------------------------
# (c) near dedupe (SimHash)
# ---------------------------------------------------------------------------
def test_simhash_distances_behave():
    d1 = "Machine learning is a field of artificial intelligence. " * 20
    d2 = "Deep learning uses neural networks with many layers. " * 20
    d4 = (("Deep learning uses neural networks with many layers. " * 19)
          + "Deep learning uses neural networks with several layers. ")

    h1, h2, h4 = SimHash(d1), SimHash(d2), SimHash(d4)
    dist_near = h2.hamming(h4)
    dist_far = h1.hamming(h2)

    # Core tuning assumption: near-dups are much closer than unrelated docs.
    assert dist_near < dist_far
    # The default threshold (8) sits comfortably between the two.
    assert dist_near <= 8 < dist_far
    assert h2.similarity(h4) > h1.similarity(h2)


def test_near_dedupe_flags_doc4_not_doc1_or_doc2(synthetic_docs):
    docs = [synthetic_docs[0], synthetic_docs[1], synthetic_docs[3]]  # 1, 2, 4
    h1 = SimHash(docs[0]["text"])
    h2 = SimHash(docs[1]["text"])
    h4 = SimHash(docs[2]["text"])
    dist_near = h2.hamming(h4)
    dist_far = h1.hamming(h2)
    assert dist_near < dist_far

    # Adaptive threshold: strictly between the near and far distances.
    threshold = (dist_near + dist_far) // 2
    assert dist_near <= threshold < dist_far

    deduper = NearDeduper(threshold_bits=threshold)
    unique, duplicates = deduper.dedupe(docs)
    assert [d["id"] for d in duplicates] == ["doc-4"]
    assert {d["id"] for d in unique} == {"doc-1", "doc-2"}

    # The default threshold also separates this corpus correctly.
    default_deduper = NearDeduper()
    unique_d, duplicates_d = default_deduper.dedupe(docs)
    assert [d["id"] for d in duplicates_d] == ["doc-4"]
    assert {d["id"] for d in unique_d} == {"doc-1", "doc-2"}

    # Incremental API.
    inc = NearDeduper(threshold_bits=threshold)
    inc.add(docs[1]["text"])
    assert inc.is_near_duplicate(docs[2]["text"])
    assert not inc.is_near_duplicate(docs[0]["text"])


def test_near_deduper_from_config():
    deduper = NearDeduper.from_config({"threshold_bits": 3, "hashbits": 64})
    assert deduper.threshold_bits == 3
    assert deduper.hashbits == 64


# ---------------------------------------------------------------------------
# (d) normalize / strip_boilerplate / clean_document
# ---------------------------------------------------------------------------
def test_normalize_text():
    assert normalize_text("  Hello\t\tworld! \n foo  ") == "Hello world! foo"
    assert normalize_text("") == ""
    # NFKC: full-width characters fold to ASCII.
    assert normalize_text("Ｈｅｌｌｏ") == "Hello"


def test_strip_boilerplate():
    messy = "\n".join([
        "Welcome to the Example documentation portal.",
        "Copyright 2026 Example Corp. All rights reserved.",
        "Home | About | Contact | Privacy",
        "We use cookies to improve your experience on this site.",
        "Machine learning is a fascinating field of study with many applications.",
        "It continues to grow as more researchers join the field every year.",
        "hi",
    ])
    stripped = strip_boilerplate(messy)

    assert "Welcome to the Example documentation portal." in stripped
    assert "Machine learning is a fascinating field" in stripped
    assert "researchers join the field" in stripped
    assert "Copyright" not in stripped
    assert "Home | About" not in stripped
    assert "cookies" not in stripped
    assert "\nhi\n" not in f"\n{stripped}\n"  # very short line dropped

    # Mostly-short documents keep their short lines (poetry/code safety).
    terse = "red\nblue\ngreen\na much longer line here now"
    assert "red" in strip_boilerplate(terse).splitlines()


def test_clean_document_never_mutates_input(synthetic_docs):
    doc = dict(synthetic_docs[0])
    original_text = doc["text"]
    original_keys = set(doc.keys())

    cleaned = clean_document(doc)

    assert cleaned is not doc
    assert set(doc.keys()) == original_keys
    assert doc["text"] == original_text
    assert cleaned["metadata"] == {"cleaned": True}
    assert "metadata" not in doc
    # Cleaning is idempotent on already-clean synthetic text.
    assert cleaned["text"] == normalize_text(original_text)


def test_clean_document_merges_existing_metadata():
    doc = {"id": "x", "text": "some   text here",
           "metadata": {"source": "crawl"}}
    cleaned = clean_document(doc)
    assert cleaned["metadata"] == {"source": "crawl", "cleaned": True}
    assert doc["metadata"] == {"source": "crawl"}


# ---------------------------------------------------------------------------
# (e) language detection + filtering
# ---------------------------------------------------------------------------
def test_detect_language():
    english = ("This is a sample document about machine learning. The field "
               "of artificial intelligence is growing and it is used for "
               "many different kinds of work in the modern world.")
    assert detect_language(english) == "en"

    french = ("Le chat est sur la table dans la maison bleue. "
              "Les enfants jouent dans le jardin avec leurs amis.") * 3
    assert detect_language(french) == "fr"

    assert detect_language("x") == "unknown"
    assert detect_language("") == "unknown"


def test_filter_by_language_keeps_english(synthetic_docs):
    kept = filter_by_language(synthetic_docs, ["en"])
    assert {d["id"] for d in kept} == {"doc-1", "doc-2", "doc-3", "doc-4",
                                      "doc-5"}

    french = {"id": "fr-1", "lang": "fr",
              "text": "Le chat est sur la table. " * 20}
    kept = filter_by_language(synthetic_docs + [french], ["en"])
    assert "fr-1" not in {d["id"] for d in kept}

    # Falls back to detection when the doc carries no lang field.
    no_lang_french = {"id": "fr-2",
                      "text": ("Le chat est sur la table dans la maison. "
                               "Les enfants jouent dehors.") * 5}
    assert filter_by_language([no_lang_french], ["en"]) == []
    assert filter_by_language([no_lang_french], ["fr"]) == [no_lang_french]
