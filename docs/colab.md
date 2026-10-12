# Google Colab integration

## 0. Run the crawler in Colab (no computer needed)

Colab works from a phone browser and has normal internet access, so the
sandbox's SSRF protection is happy. Steps:

**1. Create a GitHub access token** (the repo is private): on github.com go to
Settings → Developer settings → Personal access tokens → Fine-grained tokens →
Generate new token. Give it access to `NOVA-AI-Infrastructure` with
**Contents: Read-only**. Copy the token (it starts with `github_pat_...`).

**2. New notebook** at colab.research.google.com, then:

```python
# Clone (paste your token where indicated)
!git clone https://<PASTE_TOKEN_HERE>@github.com/xXLABZXx/NOVA-AI-Infrastructure.git
%cd NOVA-AI-Infrastructure
!pip install -e . -q
```

**3. Crawl:**

```python
!python -m nova.crawlers.runner --out data/raw https://en.wikipedia.org/wiki/Artificial_intelligence
```

The default config already allowlists `en.wikipedia.org`, `arxiv.org`, and
`docs.python.org`, and caps the crawl at 5,000 pages / 200 per host. Tune
`configs/config.yaml` (`crawlers:` section) for bigger or smaller runs.

**4. Inspect what you got:**

```python
!ls -la data/raw
!head -c 1500 data/raw/crawl.jsonl
```

**5. Curate it** (raw → cleaned → approved/rejected):

```python
!./scripts/curate.sh
!ls data/approved/
```

**6. Save your data:** Colab VMs are ephemeral — download `data/approved/` (or
copy it to Google Drive with `drive.mount('/content/drive')`) before closing
the notebook, or your crawl is gone.

> **Security note:** your token grants repo access. Don't share the notebook
> with the token in it — revoke the token at
> Settings → Developer settings → Personal access tokens when you're done, or
> use a fresh one per session.

---

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
