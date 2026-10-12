"""Preflight checks before a training run.

Verifies the environment and dataset *without* requiring torch:
device description, dataset file validity, split integrity, and
checkpoint-directory writability. Prints a JSON report and exits 0 when
every check passes, 1 otherwise.

Backs ``python -m nova.training.preflight`` (see ``scripts/train.sh``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nova.training.checkpoints import CheckpointManager
from nova.training.config import TrainingConfig
from nova.training.datasets import HeldOutLeakError, SplitEnforcer
from nova.training.hardware import describe_device

REQUIRED_RECORD_FIELDS = ("id", "text")


def _read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Return (records, problems) for a JSONL file."""
    records: list[dict] = []
    problems: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {lineno}: invalid JSON ({exc})")
                continue
            if not isinstance(obj, dict):
                problems.append(f"line {lineno}: expected a JSON object")
                continue
            records.append(obj)
    return records, problems


def run_preflight(dataset: str | Path, config_path: str | Path | None = None) -> dict:
    """Run all preflight checks; return the report dict (with ``"ok"``)."""
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # -- config (lazy import; optional) ------------------------------------
    training_section: dict = {}
    if config_path:
        try:
            from nova.config import load_config  # lazy

            training_section = load_config(config_path).section("training")
            check("config", True, f"loaded {config_path}")
        except Exception as exc:  # config errors are reported, not fatal here
            check("config", False, f"{type(exc).__name__}: {exc}")
    else:
        check("config", True, "no config file requested; using defaults")

    try:
        tconfig = TrainingConfig.from_dict(training_section).validate()
        check("training-config", True, f"batch_size={tconfig.batch_size}")
    except ValueError as exc:
        tconfig = TrainingConfig()
        check("training-config", False, str(exc))

    # -- device --------------------------------------------------------------
    device = describe_device()
    check(
        "device",
        True,
        f"torch_available={device['torch_available']} device={device['device']}",
    )

    # -- dataset ---------------------------------------------------------------
    ds_path = Path(dataset)
    records: list[dict] = []
    dataset_info: dict = {"path": str(ds_path), "records": 0}
    if not ds_path.is_file():
        check("dataset", False, f"not found: {ds_path}")
    else:
        records, problems = _read_jsonl(ds_path)
        missing = [
            str(r.get("id", f"record#{i}"))
            for i, r in enumerate(records)
            if not r.get("id") or not r.get("text")
        ]
        if missing:
            problems.append(
                f"{len(missing)} record(s) missing required fields "
                f"{REQUIRED_RECORD_FIELDS} (e.g. {', '.join(missing[:3])})"
            )
        dataset_info["records"] = len(records)
        check(
            "dataset",
            not problems,
            f"{len(records)} records" + ("" if not problems else f"; {problems[0]}"),
        )

    # -- split integrity ---------------------------------------------------------
    enforcer = SplitEnforcer()
    tally: dict[str, int] = {}
    for record in records:
        split = enforcer.assign(str(record.get("id")))
        tally[split] = tally.get(split, 0) + 1
    dataset_info["simulated_splits"] = tally
    try:
        train_only = [
            {"id": r.get("id"), "split": "train"} for r in records
        ]
        enforcer.load_for_training(train_only)
        check("split-enforcement", True,
              "train-only view accepted; held-out splits excluded")
    except HeldOutLeakError as exc:
        check("split-enforcement", False, str(exc))
    # Negative control: the enforcer must reject a held-out record.
    try:
        enforcer.load_for_training([{"id": "__probe__", "split": "val"}])
        check("split-enforcement-negative", False,
              "held-out probe record was NOT rejected")
    except HeldOutLeakError:
        check("split-enforcement-negative", True,
              "held-out probe record correctly rejected")

    # -- checkpoint directory writable ---------------------------------------------
    ckpt_dir = Path(tconfig.checkpoint_dir)
    try:
        manager = CheckpointManager(ckpt_dir)
        probe = ckpt_dir / ".preflight-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check("checkpoint-dir", True,
              f"writable: {manager.checkpoint_dir} "
              f"(latest_tag={manager.latest_tag()})")
    except OSError as exc:
        check("checkpoint-dir", False, f"not writable: {exc}")

    ok = all(c["ok"] for c in checks)
    return {
        "ok": ok,
        "device": device,
        "dataset": dataset_info,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when all checks pass, 1 otherwise."""
    parser = argparse.ArgumentParser(
        prog="nova.training.preflight",
        description="Preflight checks for a training run (no torch required).",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("dataset", help="JSONL dataset file to validate")
    args = parser.parse_args(argv)

    report = run_preflight(args.dataset, config_path=args.config)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
