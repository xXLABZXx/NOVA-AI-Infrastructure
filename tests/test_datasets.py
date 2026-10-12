"""Tests for the dataset factory (hermetic, synthetic fixtures only)."""

import json
import sqlite3

import pytest

from nova.datasets.base import (
    PartStats,
    Record,
    fingerprint,
    iter_jsonl_parts,
    write_jsonl,
)
from nova.datasets.profile import profile_parts
from nova.datasets.diversify import (
    discover_templates,
    diversify,
    validate_rewrite,
)
from nova.datasets.mix import (
    BaseHashStore,
    build_base_hash_store,
    mix,
    normalize_source_record,
)


def _write(tmp_path, name, records):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def _sft(i, category="general_chat", prompt=None, answer="canned answer"):
    return {
        "id": f"NOVA8-{category}-{i:07d}",
        "category": category,
        "prompt": prompt or f"Question number {i} about topic {i % 7}?",
        "answer": answer,
        "synthetic": True,
    }


# ---------------------------------------------------------------- base

def test_reader_skips_malformed_lines(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text(
        '{"id": "a", "prompt": "q", "answer": "x"}\n'
        "NOT JSON AT ALL\n"
        '{"id": "b", "prompt": "q2", "answer": "y"}\n'
        '{"broken": \n',
        encoding="utf-8",
    )
    stats = PartStats()
    recs = list(iter_jsonl_parts(str(p), stats=stats))
    assert [r["id"] for r in recs] == ["a", "b"]
    assert stats.malformed == 2
    assert stats.records == 2


def test_reader_supports_zip(tmp_path):
    import zipfile
    zp = tmp_path / "parts.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("part-00.txt", '{"id": "z1"}\n{"id": "z2"}\n')
    recs = list(iter_jsonl_parts(f"{zp}:part-*.txt"))
    assert [r["id"] for r in recs] == ["z1", "z2"]


def test_record_round_trip():
    r = Record(id="x", kind="sft", category="c", prompt="p", answer="a",
               source="s", synthetic=True, provenance={"k": "v"})
    r2 = Record.from_dict(r.to_dict())
    assert r2.id == "x" and r2.answer == "a" and r2.provenance == {"k": "v"}


def test_fingerprint_stable_and_normalized():
    a = {"prompt": "Hello  World", "answer": "Yes."}
    b = {"prompt": "hello world", "answer": "yes."}
    assert fingerprint(a) == fingerprint(b)
    c = {"prompt": "Hello World", "answer": "No."}
    assert fingerprint(a) != fingerprint(c)


# ---------------------------------------------------------------- profile

def test_profiler_detects_template_collapse(tmp_path):
    recs = [_sft(i, answer="the one true answer") for i in range(100)]
    p = _write(tmp_path, "p.txt", recs)
    rep = profile_parts(p)
    assert rep["total_records"] == 100
    assert rep["unique_answers"] == 1
    assert rep["answer_collapse_ratio"] == pytest.approx(0.99)
    assert rep["per_category"]["general_chat"] == 100
    assert rep["synthetic_ratio"] == 1.0


def test_profiler_counts_categories(tmp_path):
    recs = ([_sft(i, "a") for i in range(10)]
            + [_sft(100 + i, "b", answer="other") for i in range(5)])
    p = _write(tmp_path, "p.txt", recs)
    rep = profile_parts(p)
    assert rep["per_category"] == {"a": 10, "b": 5}
    assert rep["unique_answers"] == 2


# ---------------------------------------------------------------- diversify

def test_discover_templates(tmp_path):
    recs = [_sft(i, answer="tpl") for i in range(10)] + \
           [_sft(100 + i, answer=f"unique {i}") for i in range(3)]
    p = _write(tmp_path, "p.txt", recs)
    tpl = discover_templates(p, min_count=5)
    assert len(tpl) == 1
    assert next(iter(tpl.values()))["count"] == 10


def test_validate_rewrite():
    tpl = "this is the old canned template answer text"
    ok, _ = validate_rewrite("A brand new explanation here.", tpl,
                             "some prompt?")
    assert ok
    ok, reason = validate_rewrite(tpl, tpl, "q")
    assert not ok and reason == "identical_to_template"
    ok, reason = validate_rewrite("   ", tpl, "q")
    assert not ok and reason == "empty"
    ok, reason = validate_rewrite("short", tpl, "q")
    assert not ok and reason == "too_short"


def test_diversify_dry_run_no_api_key(tmp_path):
    recs = [_sft(i, answer="tpl") for i in range(20)]
    p = _write(tmp_path, "p.txt", recs)
    out = str(tmp_path / "out.jsonl")
    stats = diversify(p, out, teacher=None, min_template_count=5,
                      dry_run=True)
    assert stats["dry_run"] is True
    assert stats["collapsed_records"] == 20
    assert stats["rewritten"] == 0
    assert stats["est_cost_usd"] >= 0
    # dry run writes nothing
    assert not open(out).read() if __import__("os").path.exists(out) else True


class _FakeTeacher:
    model = "fake-teacher"
    enabled = True

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def complete(self, prompt, max_tokens=None):
        self.calls += 1
        return {"content": self.answers[(self.calls - 1) % len(self.answers)],
                "model": self.model, "usage": {}, "calls_made": self.calls}


def test_diversify_rewrites_with_fake_teacher(tmp_path):
    recs = [_sft(i, answer="tpl") for i in range(6)]
    p = _write(tmp_path, "p.txt", recs)
    out = str(tmp_path / "out.jsonl")
    teacher = _FakeTeacher([
        "Completely rewritten answer number one with new words.",
        "Another distinct rewritten answer with different structure.",
    ])
    stats = diversify(p, out, teacher=teacher, min_template_count=5,
                      limit=4)
    assert stats["rewritten"] == 4
    assert teacher.calls == 4
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 4
    # ids preserved, provenance stamped
    assert rows[0]["id"] == recs[0]["id"]
    assert rows[0]["provenance"]["rewritten_by"] == "diversify"
    assert rows[0]["provenance"]["rewrite_model"] == "fake-teacher"
    assert "original_answer_sha256" in rows[0]["provenance"]
    # answers actually differ from the template
    assert all(r["answer"] != "tpl" for r in rows)


def test_diversify_rejects_bad_teacher_output(tmp_path):
    recs = [_sft(i, answer="tpl") for i in range(3)]
    p = _write(tmp_path, "p.txt", recs)
    out = str(tmp_path / "out.jsonl")
    teacher = _FakeTeacher(["tpl"])  # echoes the template -> invalid
    stats = diversify(p, out, teacher=teacher, min_template_count=2)
    assert stats["rewritten"] == 0
    assert stats["failed_or_rejected"] == 3


def test_diversify_resume_from(tmp_path):
    recs = [_sft(i, answer="tpl") for i in range(6)]
    p = _write(tmp_path, "p.txt", recs)
    out = str(tmp_path / "out.jsonl")
    teacher = _FakeTeacher(["A sufficiently long rewritten answer here."])
    stats = diversify(p, out, teacher=teacher, min_template_count=2,
                      resume_from=recs[2]["id"])
    assert stats["rewritten"] == 3  # records 3,4,5


# ---------------------------------------------------------------- mix

def _base_parts(tmp_path, n=30):
    recs = [_sft(i, answer=f"base answer {i % 3}") for i in range(n)]
    return _write(tmp_path, "base.txt", recs), recs


def test_build_base_hash_store(tmp_path):
    p, recs = _base_parts(tmp_path)
    db = str(tmp_path / "hashes.sqlite")
    stats = build_base_hash_store(p, db, batch=7)
    assert stats["base_records"] == 30
    store = BaseHashStore(db)
    assert store.contains(fingerprint(recs[0]))
    assert not store.contains(fingerprint({"prompt": "nope", "answer": "nah"}))
    store.close()
    # unique fingerprints only: 30 records but only 3 distinct answers... but
    # prompts differ, so all 30 fingerprints are distinct
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM base_hashes").fetchone()[0]
    con.close()
    assert n == 30


def test_mixer_dedups_against_base(tmp_path):
    base_p, base_recs = _base_parts(tmp_path)
    db = str(tmp_path / "hashes.sqlite")
    build_base_hash_store(base_p, db)
    # new source: 5 exact copies of base records + 5 genuinely new
    new_recs = [dict(r) for r in base_recs[:5]]
    new_recs += [_sft(1000 + i, answer=f"brand new answer {i}") for i in range(5)]
    new_p = _write(tmp_path, "new.txt", new_recs)
    out_dir = str(tmp_path / "v8")
    stats = mix(
        sources=[{"name": "new", "pattern": new_p}],
        base_db=db,
        out_dir=out_dir,
        version="8-test",
        enable_near_dedup=False,
        split_ratios={"train": 100, "val": 0, "test": 0, "challenge": 0},
        records_per_part=1000,
        min_text_chars=1,
    )
    assert stats["rejected"].get("exact_dup_of_base") == 5
    assert stats["accepted"] == 5
    assert stats["splits"] == {"train": 5}


def test_mixer_splits_leak_free(tmp_path):
    base_p, _ = _base_parts(tmp_path, n=10)
    db = str(tmp_path / "hashes.sqlite")
    build_base_hash_store(base_p, db)
    new_recs = [_sft(5000 + i, answer=f"fresh answer {i}") for i in range(200)]
    new_p = _write(tmp_path, "new.txt", new_recs)
    out_dir = str(tmp_path / "v8")
    stats = mix(
        sources=[{"name": "new", "pattern": new_p}],
        base_db=db,
        out_dir=out_dir,
        version="8-test",
        enable_near_dedup=False,
        split_ratios={"train": 80, "val": 10, "test": 5, "challenge": 5},
        records_per_part=1000,
        min_text_chars=1,
    )
    # every record appears in exactly one split
    seen: dict[str, str] = {}
    from pathlib import Path
    for part in Path(out_dir).glob("dataset-v8-test-*-part-*.jsonl"):
        split = part.name.split("-")[3]
        for line in open(part, encoding="utf-8"):
            rec = json.loads(line)
            assert rec["split"] == split
            assert rec["id"] not in seen, f"{rec['id']} in two splits!"
            seen[rec["id"]] = split
    assert len(seen) == 200
    # SplitEnforcer semantics: held-out never in training loader
    from nova.training.datasets import SplitEnforcer, HeldOutLeakError
    enforcer = SplitEnforcer()
    train_recs = [json.loads(l) for l in
                  open(next(Path(out_dir).glob("dataset-v8-test-train-part-*.jsonl")),
                       encoding="utf-8")]
    assert enforcer.load_for_training(train_recs) == train_recs
    val_recs = [json.loads(l) for l in
                open(next(Path(out_dir).glob("dataset-v8-test-val-part-*.jsonl")),
                     encoding="utf-8")]
    with pytest.raises(HeldOutLeakError):
        enforcer.load_for_training(train_recs + val_recs)


def test_mixer_manifest_verifies(tmp_path):
    from nova.curation.versions import verify_manifest
    base_p, _ = _base_parts(tmp_path, n=10)
    db = str(tmp_path / "hashes.sqlite")
    build_base_hash_store(base_p, db)
    new_p = _write(tmp_path, "new.txt",
                   [_sft(9000 + i, answer=f"m{i}") for i in range(10)])
    out_dir = str(tmp_path / "v8")
    stats = mix(
        sources=[{"name": "new", "pattern": new_p}],
        base_db=db, out_dir=out_dir, version="8-test",
        enable_near_dedup=False,
        min_text_chars=1,
        split_ratios={"train": 100, "val": 0, "test": 0, "challenge": 0},
    )
    ok, problems = verify_manifest(stats["manifest"])
    assert ok, problems
    manifest = json.loads(open(stats["manifest"], encoding="utf-8").read())
    assert manifest["version"] == "8-test"
    assert manifest["splits"]["train"] == 10


def test_mixer_rejected_has_reasons(tmp_path):
    base_p, base_recs = _base_parts(tmp_path, n=5)
    db = str(tmp_path / "hashes.sqlite")
    build_base_hash_store(base_p, db)
    dup = dict(base_recs[0])
    new_p = _write(tmp_path, "new.txt", [dup])
    out_dir = str(tmp_path / "v8")
    mix(sources=[{"name": "new", "pattern": new_p}], base_db=db,
        out_dir=out_dir, version="8-test", enable_near_dedup=False,
        min_text_chars=1,
        split_ratios={"train": 100, "val": 0, "test": 0, "challenge": 0})
    rej = [json.loads(l) for l in
           open(f"{out_dir}/rejected.jsonl", encoding="utf-8")]
    assert len(rej) == 1
    assert rej[0]["reason"] == "exact_dup_of_base"
    assert rej[0]["id"] == base_recs[0]["id"]


def test_normalize_crawler_record():
    rec = normalize_source_record(
        {"url": "http://x", "title": "T", "text": "body text here"}, "crawler")
    assert rec["kind"] == "text"
    assert rec["source"] == "crawler"
