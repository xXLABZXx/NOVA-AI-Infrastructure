# Google Colab integration

Training runs where GPUs live. This repo is designed so the **data pipeline runs
locally (or in Docker)** and **training runs in Colab** against approved,
versioned datasets.

## 1. Prepare an approved dataset locally

```bash
./scripts/crawl.sh https://en.wikipedia.org/wiki/Artificial_intelligence
./scripts/curate.sh
ls data/approved/   # pick a versioned dataset, e.g. dataset-v1.jsonl + manifest.json
```

## 2. Get the data into Colab

Option A — upload the approved JSONL + manifest to Google Drive, then mount it:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Option B — `git clone` this repo in Colab and copy the approved dataset in.

Verify integrity before training:

```python
from nova.curation.versions import verify_manifest
ok, problems = verify_manifest("data/approved/manifest.json")
assert ok, problems
```

## 3. Install and preflight

```python
!pip install -e ".[torch]"   # or: pip install torch pyyaml
!python -m nova.training.preflight --config configs/config.yaml data/approved/dataset-v1.jsonl
```

`preflight` checks: GPU availability, torch version, dataset/tokenizer
compatibility, split integrity (val/test/challenge never leak into training),
and checkpoint architecture compatibility — all without starting training.

## 4. Train

```python
from nova.training import TrainingConfig, CheckpointManager, SplitEnforcer

cfg = TrainingConfig(batch_size=8, gradient_accumulation_steps=4,
                     sequence_length=2048, learning_rate=2e-5,
                     mixed_precision=True)
# ... your existing NOVA training loop plugs in here ...
```

See `src/nova/training/` docstrings for the checkpoint save/resume helpers and
the JSONL loss tracker. Resume after interruption with
`CheckpointManager.resume_latest(checkpoint_dir)`.
