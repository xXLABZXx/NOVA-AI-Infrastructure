"""Hermetic tests for the curation package (no network, no torch)."""

import json
import shutil

import pytest

from nova.curation import (
    ChangeLog,
    quality_score,
    run_pipeline,
    source_reliability,
    verify_manifest,
    write_manifest,
)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- scoring -----------------------------------------------------------------


def test_quality_score_deterministic_and_bounded(synthetic_docs):
    for doc in synthetic_docs:
        first = quality_score(doc)
        assert 0.0 <= first <= 1.0
        assert quality_score(doc) == first  # deterministic


def test_quality_score_rejects_tiny_text(synthetic_docs):
    tiny = next(d for d in synthetic_docs if d["id"] == "doc-5")
    good = next(d for d in synthetic_docs if d["id"] == "doc-1")
    assert quality_score(tiny) < 0.35
    assert quality_score(good) >= 0.35


def test_quality_score_empty_doc():
    assert quality_score({}) == 0.0
    assert quality_score({"id": "x", "text": ""}) == 0.0


def test_source_reliability_tiers():
    assert source_reliability("https://en.wikipedia.org/wiki/X") == 0.9
    assert source_reliability("https://arxiv.org/abs/1234") == 0.9
    assert source_reliability("https://docs.python.org/3/") == 0.9
    assert source_reliability("https://mit.edu/page") == 0.8
    assert source_reliability("https://www.nih.gov/x") == 0.8
    assert source_reliability("https://github.com/org/repo") == 0.7
    assert source_reliability("https://example.com/a") == 0.5
    assert source_reliability("") == 0.3
    assert source_reliability("some-bare-source") == 0.3


# -- changelog -----------------------------------------------------------------


def test_changelog_propose_approve_reject(tmp_path):
    log = ChangeLog(tmp_path / "cl.jsonl")
    entry_id = log.propose("doc-1", {"action": "clean"}, "looks good")
    assert len(log.pending()) == 1
    assert log.entries_for("doc-1")[0]["approval_status"] == "pending"

    log.approve(entry_id, approver="cleo")
    assert log.pending() == []
    assert log.entries_for("doc-1")[0]["approval_status"] == "approved"
    assert log.entries_for("doc-1")[0]["approver"] == "cleo"

    entry2 = log.propose("doc-2", {"action": "clean"}, "meh")
    log.reject(entry2, approver="cleo", reason="too short")
    entry = log.entries_for("doc-2")[0]
    assert entry["approval_status"] == "rejected"
    assert entry["rejection_reason"] == "too short"

    with pytest.raises(KeyError):
        log.approve("nope", approver="cleo")
    with pytest.raises(ValueError):
        log.approve(entry_id, approver="cleo")  # already approved


# -- versions --------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "a.jsonl").write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    manifest = write_manifest(d, "v1", extra={"note": "test"})
    ok, problems = verify_manifest(manifest)
    assert ok and problems == []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["version"] == "v1"
    assert data["note"] == "test"
    assert data["files"][0]["records"] == 2


def test_verify_manifest_detects_tamper(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "a.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    manifest = write_manifest(d, "v1")
    with open(d / "a.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"id": 2}\n')
    ok, problems = verify_manifest(manifest)
    assert not ok
    assert any("mismatch" in p for p in problems)


def test_verify_manifest_missing_file(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "a.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    manifest = write_manifest(d, "v1")
    (d / "a.jsonl").unlink()
    ok, problems = verify_manifest(manifest)
    assert not ok
    assert any("missing" in p for p in problems)


# -- pipeline ----------------------------------------------------------------------


def _run(raw_src, tmp_path, **kwargs):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(raw_src, raw_dir / "synthetic.jsonl")
    before = (raw_dir / "synthetic.jsonl").read_bytes()
    stats = run_pipeline(
        raw_dir=raw_dir,
        cleaned_dir=tmp_path / "cleaned",
        approved_dir=tmp_path / "approved",
        rejected_dir=tmp_path / "rejected",
        changelog_path=tmp_path / "changelog.jsonl",
        **kwargs,
    )
    after = (raw_dir / "synthetic.jsonl").read_bytes()
    assert before == after, "raw input file was modified"
    return stats, tmp_path


def test_pipeline_auto_approve(synthetic_jsonl, tmp_path):
    stats, root = _run(synthetic_jsonl, tmp_path, require_human_approval=False,
                       version="v1")
    assert stats["total"] == 5
    assert stats["cleaned"] == 5
    assert stats["approved"] == 4
    assert stats["rejected"] == 1
    assert 0.0 <= stats["avg_quality"] <= 1.0

    cleaned = _read_jsonl(root / "cleaned" / "all.jsonl")
    assert len(cleaned) == 5
    for doc in cleaned:
        assert "quality_score" in doc and "source_reliability" in doc

    approved = _read_jsonl(root / "approved" / "dataset-v1.jsonl")
    assert {d["id"] for d in approved} == {"doc-1", "doc-2", "doc-3", "doc-4"}

    rejected = _read_jsonl(root / "rejected" / "rejected.jsonl")
    assert len(rejected) == 1
    assert rejected[0]["id"] == "doc-5"
    assert rejected[0]["rejection_reason"]  # non-empty reason

    # One changelog entry per processed doc.
    log = ChangeLog(root / "changelog.jsonl")
    for doc_id in ("doc-1", "doc-2", "doc-3", "doc-4", "doc-5"):
        assert log.entries_for(doc_id), f"no changelog entry for {doc_id}"
    assert log.pending() == []  # pipeline auto-decided everything

    # Manifest verifies...
    manifest = root / "approved" / "manifest.json"
    ok, problems = verify_manifest(manifest)
    assert ok, problems

    # ...and detects tampering of the approved dataset.
    with open(root / "approved" / "dataset-v1.jsonl", "a",
              encoding="utf-8") as fh:
        fh.write('{"id": "evil", "text": "tampered"}\n')
    ok, problems = verify_manifest(manifest)
    assert not ok and problems


def test_pipeline_human_approval_gates_approved(synthetic_jsonl, tmp_path):
    stats, root = _run(synthetic_jsonl, tmp_path, require_human_approval=True,
                       version="v1")
    assert stats["total"] == 5
    assert stats["approved"] == 0  # nothing auto-approved
    assert stats["rejected"] == 1
    assert not (root / "approved" / "dataset-v1.jsonl").exists()
    log = ChangeLog(root / "changelog.jsonl")
    pending = log.pending()
    assert len(pending) == 4  # the four passing docs await a human
    assert all(e["approval_status"] == "pending" for e in pending)
    # Manifest of an empty approved dir still verifies.
    ok, _ = verify_manifest(root / "approved" / "manifest.json")
    assert ok


def test_pipeline_missing_raw_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            raw_dir=tmp_path / "nope",
            cleaned_dir=tmp_path / "cleaned",
            approved_dir=tmp_path / "approved",
            rejected_dir=tmp_path / "rejected",
            changelog_path=tmp_path / "changelog.jsonl",
        )
