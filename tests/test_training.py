"""Hermetic tests for the training package (no torch required)."""

import json
import sys
from pathlib import Path

import pytest

from nova.training import (
    SPLITS,
    CheckpointManager,
    ExperimentSnapshot,
    HeldOutLeakError,
    IncompatibleCheckpointError,
    LossTracker,
    SplitEnforcer,
    TrainingConfig,
    describe_device,
)
from nova.training.datasets import check_dataset_compatibility
from nova.training.preflight import main as preflight_main

META = {"architecture": "nova-1", "vocab_size": 32000, "tokenizer": "nova-bpe"}


# -- config ------------------------------------------------------------------------


def test_training_config_defaults_and_validate():
    cfg = TrainingConfig().validate()
    assert cfg.batch_size == 8
    assert cfg.sequence_length == 2048


def test_training_config_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0).validate()
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=-0.1).validate()
    with pytest.raises(ValueError):
        TrainingConfig(sequence_length=0).validate()
    with pytest.raises(ValueError):
        TrainingConfig(checkpoint_dir="").validate()


def test_training_config_from_dict_ignores_unknown():
    cfg = TrainingConfig.from_dict({"batch_size": 16, "nope": 1})
    assert cfg.batch_size == 16
    assert cfg.seed == 42


# -- hardware ------------------------------------------------------------------------


def test_describe_device_shape():
    d = describe_device()
    assert d["torch_available"] in (True, False)
    assert d["device"] in ("cuda", "cpu")
    assert isinstance(d["device_count"], int)
    assert ("torch_version" in d) and ("cuda_available" in d)


def test_describe_device_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # force ImportError
    d = describe_device()
    assert d["torch_available"] is False
    assert d["torch_version"] is None
    assert d["device"] == "cpu"


# -- checkpoints -----------------------------------------------------------------------


def test_checkpoint_save_load_roundtrip(tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts")
    meta_path = mgr.save("step-100", META, weights=b"\x00\x01\x02")
    assert meta_path.name == "checkpoint.json"
    loaded = mgr.load("step-100", META)
    assert loaded["metadata"]["vocab_size"] == 32000
    assert loaded["metadata"]["tag"] == "step-100"
    assert loaded["weights_path"] is not None
    assert loaded["weights_path"].read_bytes() == b"\x00\x01\x02"


def test_checkpoint_load_without_weights(tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts")
    mgr.save("step-1", META)
    loaded = mgr.load("step-1", META)
    assert loaded["weights_path"] is None


def test_checkpoint_incompatible_vocab(tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts")
    mgr.save("step-100", META)
    with pytest.raises(IncompatibleCheckpointError) as excinfo:
        mgr.load("step-100", {**META, "vocab_size": 999})
    assert "vocab_size" in str(excinfo.value)


def test_checkpoint_incompatible_architecture(tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts")
    mgr.save("step-100", META)
    with pytest.raises(IncompatibleCheckpointError):
        mgr.load("step-100", {**META, "architecture": "other"})


def test_checkpoint_latest_tag_and_resume(tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts")
    assert mgr.latest_tag() is None
    assert mgr.resume_latest(META) is None
    mgr.save("step-1", META)
    mgr.save("step-2", META)
    assert mgr.latest_tag() == "step-2"
    resumed = mgr.resume_latest(META)
    assert resumed["metadata"]["tag"] == "step-2"


def test_checkpoint_missing_tag(tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts")
    with pytest.raises(FileNotFoundError):
        mgr.load("nope", META)


# -- splits ----------------------------------------------------------------------------


def test_splits_constant():
    assert SPLITS == ("train", "val", "test", "challenge")


def test_assign_deterministic_and_covering():
    enforcer = SplitEnforcer()
    seen = set()
    for i in range(500):
        doc_id = f"doc-{i}"
        first = enforcer.assign(doc_id)
        assert enforcer.assign(doc_id) == first  # deterministic
        assert first in SPLITS
        seen.add(first)
    assert seen == set(SPLITS)  # all splits get used


def test_split_map_override():
    enforcer = SplitEnforcer(split_map={"special": "challenge"})
    assert enforcer.assign("special") == "challenge"


def test_custom_ratios():
    enforcer = SplitEnforcer(ratios={"train": 50, "val": 50})
    with pytest.raises(ValueError):
        SplitEnforcer(ratios={"train": 50, "val": 49})  # must sum to 100


def test_load_for_training_rejects_heldout():
    enforcer = SplitEnforcer()
    records = [
        {"id": "a", "split": "train"},
        {"id": "b", "split": "test"},
        {"id": "c"},  # no split -> train
    ]
    with pytest.raises(HeldOutLeakError) as excinfo:
        enforcer.load_for_training(records)
    assert "b" in str(excinfo.value)


def test_load_for_training_ok():
    enforcer = SplitEnforcer()
    records = [{"id": "a", "split": "train"}, {"id": "c"}]
    assert enforcer.load_for_training(records) == records


def test_assert_no_leak_raises_on_normalized_match():
    enforcer = SplitEnforcer()
    with pytest.raises(HeldOutLeakError) as excinfo:
        enforcer.assert_no_leak(["Hello   World"], ["hello world"])
    assert "1 held-out" in str(excinfo.value)


def test_assert_no_leak_passes_when_clean():
    SplitEnforcer().assert_no_leak(["alpha"], ["beta"])  # no raise


def test_check_dataset_compatibility():
    info = {"tokenizer": "nova-bpe", "vocab_size": 32000, "format": "jsonl"}
    check_dataset_compatibility(info, info)  # no raise
    with pytest.raises(ValueError) as excinfo:
        check_dataset_compatibility(info, {**info, "vocab_size": 1})
    assert "vocab_size" in str(excinfo.value)


# -- tracking ----------------------------------------------------------------------------


def test_loss_tracker_roundtrip(tmp_path):
    tracker = LossTracker(tmp_path / "losses.jsonl")
    tracker.log_step(1, 2.5, val_loss=2.7, extra={"lr": 1e-5})
    tracker.log_step(2, 2.3)
    rows = tracker.read_all()
    assert len(rows) == 2
    assert rows[0]["step"] == 1 and rows[0]["val_loss"] == 2.7
    assert rows[0]["lr"] == 1e-5
    assert rows[1]["val_loss"] is None


def test_experiment_snapshot_roundtrip(tmp_path):
    cfg = TrainingConfig(batch_size=4)
    path = ExperimentSnapshot.save(cfg, tmp_path / "snap.json")
    data = ExperimentSnapshot.load(path)
    assert data["config"]["batch_size"] == 4
    assert "saved_at" in data


# -- preflight -----------------------------------------------------------------------------


def test_preflight_ok_without_torch(synthetic_jsonl, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # guarantee no torch
    monkeypatch.chdir(tmp_path)  # checkpoint dir lands in tmp_path
    rc = preflight_main([str(synthetic_jsonl)])
    assert rc == 0


def test_preflight_with_repo_config(synthetic_jsonl, tmp_path, monkeypatch):
    repo_config = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    monkeypatch.chdir(tmp_path)
    rc = preflight_main(["--config", str(repo_config), str(synthetic_jsonl)])
    assert rc == 0


def test_preflight_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = preflight_main([str(tmp_path / "missing.jsonl")])
    assert rc == 1


def test_preflight_bad_jsonl(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "text": "fine"}\nnot json\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = preflight_main([str(bad)])
    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert any(not c["ok"] for c in report["checks"])
