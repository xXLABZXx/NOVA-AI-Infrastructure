# Roadmap

## Near term (0.1.x)
- [ ] Browser-rendered crawling (headless) for JS-heavy docs, still behind the sandbox.
- [ ] PDF text extraction for papers (currently HTML/text only).
- [ ] Semantic dedup via embeddings (pluggable; off by default, no vendor lock-in).
- [ ] Teacher support for contradiction graphs across approved examples.
- [ ] Prometheus metrics export from `nova.monitoring`.
- [ ] DVC remote configuration example for large datasets.

## Medium term (0.2.x)
- [ ] Distributed crawl workers with a shared frontier (SQLite → Postgres).
- [ ] Human review UI for the approval queue (`data/cleaned/` → `data/approved/`).
- [ ] Automated eval regression: block dataset promotion if eval scores drop.
- [ ] Checkpoint registry with lineage (which dataset version trained which checkpoint).
- [ ] Signed dataset manifests (sigstore) for provenance.

## Longer term
- [ ] Fine-tuning recipes for NOVA on Colab (QLoRA).
- [ ] RAG index builder over approved data.
- [ ] Multi-teacher ensembles with disagreement metrics.
