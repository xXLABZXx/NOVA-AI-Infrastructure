# ============================================================
# NOVA 750M on T4 — OPTIMIZED CELLS 1/3, 2/3, 3/3
# Paste each CELL block into its own Colab cell, in order.
# Same model, same data, same math — just faster.
#
# What changed vs your original:
#  CELL 1: token cache builds on LOCAL SSD (/content), not Drive;
#          tokenizer.encode_batch() instead of one encode() per record;
#          Drive only gets a backup copy at the end (background).
#  CELL 2: chunked_loss uses 256-token chunks (was 32 -> 8x fewer
#          Python loop iterations); micro-batch is probed (2 then 1);
#          torch.compile() wraps the model after the smoke test.
#  CELL 3: (your loop was missing) full training loop with cosine
#          schedule + warmup, grad clipping, fixed-size eval,
#          local checkpoints + background Drive copies, resume,
#          and tokens/sec + ETA logging.
# ============================================================


# ================= CELL 1/3 =================
# NOVA setup, tokenizer, dataset preparation (FAST version)
%pip -q install tokenizers bitsandbytes tqdm

import os, json, zipfile, hashlib, random, gc, time, shutil, threading
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "Enable a GPU runtime in Colab first."
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))

DRIVE = Path("/content/drive/MyDrive")
if not (DRIVE / "NOVA_DATASET_#0007_20parts.zip").exists():
    from google.colab import drive
    drive.mount("/content/drive")

ARCHIVE_PATH = DRIVE / "NOVA_DATASET_#0007_20parts.zip"
assert ARCHIVE_PATH.is_file(), f"Dataset ZIP not found: {ARCHIVE_PATH}"

TOKENIZER_PATH = DRIVE / "NOVA_256K" / "tokenizer.json"
if not TOKENIZER_PATH.exists():
    from google.colab import files
    print("Upload your NOVA 256K tokenizer.json.")
    uploaded = files.upload()
    assert "tokenizer.json" in uploaded, "Upload a file named tokenizer.json."
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("tokenizer.json").replace(TOKENIZER_PATH)

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB_SIZE = tokenizer.get_vocab_size(with_added_tokens=True)
print("Tokenizer vocabulary:", VOCAB_SIZE)
assert VOCAB_SIZE == 256000, (
    f"Expected exactly 256,000 tokenizer IDs, got {VOCAB_SIZE}. "
    "Do not silently change the model vocabulary."
)

eos_id = None
for token in ["</s>", "<|endoftext|>", "<eos>"]:
    found = tokenizer.token_to_id(token)
    if found is not None:
        eos_id = found
        break
print("EOS token ID:", eos_id)

RUN_DIR = DRIVE / "NOVA_750M_T4_RUNS" / "run_001"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ---- FAST PATH: cache lives on local SSD; Drive only keeps a backup ----
CACHE_DIR = Path("/content/nova_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_CACHE = RUN_DIR / "token_cache_backup"
DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RUN_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CKPT = Path("/content/nova_ckpts")
LOCAL_CKPT.mkdir(parents=True, exist_ok=True)

# Architecture and training targets
D_MODEL = 1536
N_LAYERS = 12
N_HEADS = 24
FFN_DIM = 4352
REQUESTED_CONTEXT = 2048
CONTEXT_CANDIDATES = [2048, 1024, 512, 256, 128]
MICRO_BATCH = 1
GRAD_ACCUM = 16
MAX_STEPS = 100000
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 500
SAVE_EVERY = 500
EVAL_EVERY = 100
LOG_EVERY = 10
SEED = 20261012

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

archive_size = ARCHIVE_PATH.stat().st_size
tokenizer_hash = hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest()
cache_key = hashlib.sha256(
    f"{archive_size}:{ARCHIVE_PATH.stat().st_mtime_ns}:{tokenizer_hash}".encode()
).hexdigest()[:16]

TRAIN_BIN = CACHE_DIR / f"train_{cache_key}.bin"
VAL_BIN = CACHE_DIR / f"val_{cache_key}.bin"
CACHE_META = CACHE_DIR / f"cache_{cache_key}.json"

def _drive_backup_path(name):
    return DRIVE_CACHE / name

# If local cache is missing but a Drive backup exists, restore it (fast download,
# no re-tokenizing).
if not (TRAIN_BIN.exists() and VAL_BIN.exists()):
    for name in [TRAIN_BIN.name, VAL_BIN.name, CACHE_META.name]:
        src = _drive_backup_path(name)
        if src.exists():
            print("Restoring from Drive backup:", name)
            shutil.copy2(src, CACHE_DIR / name)

def record_text(line):
    line = line.strip()
    if not line:
        return "", None
    try:
        obj = json.loads(line)
    except Exception:
        return line, None
    if isinstance(obj, str):
        return obj, None
    if not isinstance(obj, dict):
        return line, None
    split = str(obj.get("split", "")).strip().lower()
    prompt_keys = ["task", "prompt", "question", "instruction", "input"]
    answer_keys = ["answer", "output", "response", "completion", "target"]
    prompt = next((str(obj[k]) for k in prompt_keys
                   if obj.get(k) not in (None, "")), "")
    answer = next((str(obj[k]) for k in answer_keys
                   if obj.get(k) not in (None, "")), "")
    if prompt and answer:
        text = f"User: {prompt}\nAssistant: {answer}"
    elif prompt or answer:
        text = prompt or answer
    else:
        text = json.dumps(obj, ensure_ascii=False)
    return text, split

ENCODE_BATCH = 4096

def flush_batch(handle, texts):
    """Tokenize a batch at once (multithreaded Rust) and append ids."""
    if not texts:
        return 0
    enc = tokenizer.encode_batch(texts)
    n = 0
    for e in enc:
        ids = e.ids
        if eos_id is not None:
            ids = ids + [eos_id]
        if ids:
            np.asarray(ids, dtype=np.uint32).tofile(handle)
            n += len(ids)
    return n

if TRAIN_BIN.exists() and VAL_BIN.exists() and CACHE_META.exists():
    meta = json.loads(CACHE_META.read_text())
    assert meta.get("cache_key") == cache_key
    assert TRAIN_BIN.stat().st_size > 0 and VAL_BIN.stat().st_size > 0
    print("Reusing existing token caches:", cache_key)
else:
    print("Preparing token caches on LOCAL SSD. This runs once, then is reused.")
    tmp_train = CACHE_DIR / "train_building.bin"
    tmp_val = CACHE_DIR / "val_building.bin"
    for p in [tmp_train, tmp_val]:
        if p.exists():
            p.unlink()

    counts = Counter()
    seen_hashes = set()
    start = time.time()

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf, \
         open(tmp_train, "wb") as train_f, \
         open(tmp_val, "wb") as val_f:

        members = [
            n for n in zf.namelist()
            if not n.endswith("/")
            and Path(n).suffix.lower() in {".txt", ".jsonl", ".json"}
            and not any(tag in Path(n).name.lower()
                        for tag in ["test", "challenge", "evaluation", "eval"])
        ]
        print("Dataset files:", len(members))

        for member in members:
            print("\nProcessing:", member)
            batch_texts, batch_dests = [], []
            with zf.open(member, "r") as src:
                for raw in tqdm(src, desc=Path(member).name, unit="lines"):
                    try:
                        line = raw.decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    text, split = record_text(line)
                    if not text:
                        continue
                    if split and any(x in split for x in
                                     ["test", "challenge", "evaluation", "eval"]):
                        counts["heldout_skipped"] += 1
                        continue
                    digest = hashlib.sha1(text.encode("utf-8")).digest()
                    if digest in seen_hashes:
                        counts["duplicates_skipped"] += 1
                        continue
                    seen_hashes.add(digest)
                    if split in {"val", "valid", "validation", "dev"}:
                        dest = "val"
                        counts["validation_records"] += 1
                    elif split in {"train", "training"}:
                        dest = "train"
                        counts["training_records"] += 1
                    else:
                        bucket = int.from_bytes(digest[:4], "little") % 100
                        if bucket == 0:
                            dest = "val"
                            counts["validation_records"] += 1
                        else:
                            dest = "train"
                            counts["training_records"] += 1
                    batch_texts.append(text)
                    batch_dests.append(dest)
                    if len(batch_texts) >= ENCODE_BATCH:
                        # encode in order; split batches by destination
                        enc = tokenizer.encode_batch(batch_texts)
                        for e, d in zip(enc, batch_dests):
                            ids = e.ids + ([eos_id] if eos_id is not None else [])
                            if ids:
                                h = train_f if d == "train" else val_f
                                np.asarray(ids, dtype=np.uint32).tofile(h)
                                counts["tokens_written"] += len(ids)
                        batch_texts, batch_dests = [], []
            if batch_texts:
                enc = tokenizer.encode_batch(batch_texts)
                for e, d in zip(enc, batch_dests):
                    ids = e.ids + ([eos_id] if eos_id is not None else [])
                    if ids:
                        h = train_f if d == "train" else val_f
                        np.asarray(ids, dtype=np.uint32).tofile(h)
                        counts["tokens_written"] += len(ids)

    del seen_hashes
    gc.collect()

    assert tmp_train.stat().st_size > 0, "No training tokens were produced."
    assert tmp_val.stat().st_size > 0, "No validation tokens were produced."

    tmp_train.replace(TRAIN_BIN)
    tmp_val.replace(VAL_BIN)
    CACHE_META.write_text(json.dumps({
        "cache_key": cache_key,
        "archive": str(ARCHIVE_PATH),
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": VOCAB_SIZE,
        "counts": dict(counts),
        "created_unix": time.time()
    }, indent=2))

    # Backup to Drive in the background; training does not wait for it.
    def _backup():
        for p in [TRAIN_BIN, VAL_BIN, CACHE_META]:
            shutil.copy2(p, DRIVE_CACHE / p.name)
        print("Token cache backed up to Drive.")
    threading.Thread(target=_backup, daemon=True).start()

    print("Preparation seconds:", round(time.time() - start))
    print("Record counts:", dict(counts))

train_tokens = np.memmap(TRAIN_BIN, dtype=np.uint32, mode="r")
val_tokens = np.memmap(VAL_BIN, dtype=np.uint32, mode="r")
print("\nTrain tokens:", f"{len(train_tokens):,}")
print("Validation tokens:", f"{len(val_tokens):,}")
print("Train cache GiB:", round(TRAIN_BIN.stat().st_size / 2**30, 3))
print("Validation cache GiB:", round(VAL_BIN.stat().st_size / 2**30, 3))
print("Cell 1 complete. Cache is on local SSD (fast); Drive holds a backup.")


# ================= CELL 2/3 =================
# NOVA model, optimizer, and context smoke test (FAST version)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import bitsandbytes as bnb
import gc, math, json, time
from pathlib import Path

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

USE_BF16 = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_SCALER = not USE_BF16
scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)

class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert D_MODEL % N_HEADS == 0
        self.head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        k = k.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        v = v.view(b, t, N_HEADS, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, FFN_DIM, bias=False)
        self.up = nn.Linear(D_MODEL, FFN_DIM, bias=False)
        self.down = nn.Linear(FFN_DIM, D_MODEL, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)
        self.attn = CausalAttention()
        self.norm2 = RMSNorm(D_MODEL)
        self.mlp = SwiGLU()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class NOVA(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(N_LAYERS)]
        )
        self.norm = RMSNorm(D_MODEL)
        self.gradient_checkpointing = True
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def logits_for(self, hidden):
        return F.linear(hidden, self.embed.weight)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

model = NOVA().to(device)
param_count = model.count_parameters()
print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
print("Precision:", str(AMP_DTYPE))

try:
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    print("Optimizer: bitsandbytes AdamW8bit")
except Exception as e:
    print("8-bit optimizer initialization failed:", repr(e))
    raise RuntimeError("The 8-bit optimizer could not initialize. Stopping.")

def get_batch(seq_len, split="train"):
    source = train_tokens if split == "train" else val_tokens
    max_start = len(source) - seq_len - 1
    assert max_start > 0, f"Not enough {split} tokens for context {seq_len}."
    starts = torch.randint(0, max_start, (MICRO_BATCH,)).tolist()
    xs, ys = [], []
    for s in starts:
        chunk = np.asarray(source[s:s + seq_len + 1], dtype=np.int64)
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.tensor(np.stack(xs), device=device, dtype=torch.long)
    y = torch.tensor(np.stack(ys), device=device, dtype=torch.long)
    return x, y

LOSS_CHUNK_TOKENS = 256  # was 32: 8x fewer Python loop iterations per step

def chunked_loss(hidden, targets, chunk_tokens=LOSS_CHUNK_TOKENS):
    b, t, d = hidden.shape
    flat_h = hidden.reshape(b * t, d)
    flat_y = targets.reshape(b * t)
    total = flat_y.numel()
    loss_sum = 0.0
    for start in range(0, total, chunk_tokens):
        end = min(start + chunk_tokens, total)
        logits = model.logits_for(flat_h[start:end])
        loss_sum = loss_sum + F.cross_entropy(
            logits.float(), flat_y[start:end], reduction="sum"
        )
    return loss_sum / total

def test_context(seq_len, micro_batch=1):
    global MICRO_BATCH
    old_mb = MICRO_BATCH
    MICRO_BATCH = micro_batch
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch(seq_len)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            hidden = model.forward_hidden(x)
            loss = chunked_loss(hidden, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
        if USE_SCALER:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        optimizer.zero_grad(set_to_none=True)
        del x, y, hidden, loss
        torch.cuda.synchronize()
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    finally:
        MICRO_BATCH = old_mb

SEQ_LEN = None
for candidate in CONTEXT_CANDIDATES:
    if candidate >= len(train_tokens) or candidate >= len(val_tokens):
        continue
    print(f"\nTesting context={candidate}...")
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak_gib = test_context(candidate)
        SEQ_LEN = candidate
        print(f"PASS: context={candidate}; peak allocated VRAM={peak_gib} GiB")
        break
    except torch.cuda.OutOfMemoryError:
        print(f"OOM at context={candidate}; trying a smaller context.")
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Memory failure at context={candidate}: {e}")
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            raise

assert SEQ_LEN is not None, "Even context 128 failed. Model does not fit the T4."

# Probe a bigger micro-batch at the chosen context (2x throughput if it fits).
try:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    peak = test_context(SEQ_LEN, micro_batch=2)
    MICRO_BATCH = 2
    print(f"Micro-batch 2 fits (peak {peak} GiB). MICRO_BATCH=2")
except Exception:
    MICRO_BATCH = 1
    gc.collect(); torch.cuda.empty_cache()
    print("Micro-batch 2 did not fit. MICRO_BATCH=1")

# Compile after the smoke tests: ~20-30% faster steps on the transformer body.
# (First real step takes a few minutes to compile; everything after is faster.)
model = torch.compile(model)
print("Model compiled with torch.compile.")

model.train()
optimizer.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()

config = {
    "model": "NOVA",
    "vocab_size": VOCAB_SIZE,
    "d_model": D_MODEL,
    "layers": N_LAYERS,
    "heads": N_HEADS,
    "ffn_dim": FFN_DIM,
    "context_length": SEQ_LEN,
    "requested_context": REQUESTED_CONTEXT,
    "micro_batch": MICRO_BATCH,
    "gradient_accumulation": GRAD_ACCUM,
    "max_steps": MAX_STEPS,
    "parameter_count": param_count,
    "precision": str(AMP_DTYPE),
    "random_initialization": True,
    "tokenizer_sha256": tokenizer_hash,
    "cache_key": cache_key
}
(RUN_DIR / "model_config.json").write_text(json.dumps(config, indent=2))
print("\nSelected context:", SEQ_LEN, "| Micro-batch:", MICRO_BATCH)
print("Tokens per optimizer step:", MICRO_BATCH * GRAD_ACCUM * SEQ_LEN)
print("Cell 2 complete.")


# ================= CELL 3/3 =================
# NOVA training loop: cosine schedule, eval, checkpoints, resume
import math, time, shutil, threading
from pathlib import Path

tokens_per_step = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN

def lr_at(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

for group in optimizer.param_groups:
    group["lr"] = lr_at(0)

def drive_copy_bg(local_path, drive_path):
    def _copy():
        shutil.copy2(local_path, drive_path)
    threading.Thread(target=_copy, daemon=True).start()

def save_checkpoint(step, is_best=False):
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if USE_SCALER else None,
        "config": config,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    local = LOCAL_CKPT / f"nova_step_{step}.pt"
    torch.save(state, local)  # local SSD: seconds, not minutes
    drive_copy_bg(local, CKPT_DIR / f"nova_step_{step}.pt")
    latest_local = LOCAL_CKPT / "nova_latest.pt"
    shutil.copy2(local, latest_local)
    drive_copy_bg(latest_local, CKPT_DIR / "nova_latest.pt")
    # keep only the last 3 local checkpoints to protect local disk
    for old in sorted(LOCAL_CKPT.glob("nova_step_*.pt"))[:-3]:
        old.unlink()
    print(f"[ckpt] step {step} saved locally; Drive copy in background.")

def load_checkpoint():
    cand = LOCAL_CKPT / "nova_latest.pt"
    if not cand.exists():
        d = CKPT_DIR / "nova_latest.pt"
        if d.exists():
            print("Restoring checkpoint from Drive...")
            shutil.copy2(d, cand)
    if cand.exists():
        print("Loading checkpoint:", cand)
        ck = torch.load(cand, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if USE_SCALER and ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        torch.set_rng_state(ck["rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        return ck["step"]
    return 0

@torch.inference_mode()
def evaluate(n_batches=50):
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch(SEQ_LEN, split="val")
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            hidden = model.forward_hidden(x)
            loss = chunked_loss(hidden, y)
        total += loss.item(); n += 1
    if was_training:
        model.train()
    return total / n

start_step = load_checkpoint()
print(f"Resuming from step {start_step} -> target {MAX_STEPS}")
print(f"Tokens/step: {tokens_per_step:,}")

model.train()
t0 = time.time()
for step in range(start_step + 1, MAX_STEPS + 1):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(GRAD_ACCUM):
        x, y = get_batch(SEQ_LEN, split="train")
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            hidden = model.forward_hidden(x)
            loss = chunked_loss(hidden, y) / GRAD_ACCUM
        if USE_SCALER:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_loss += loss.item()
    if USE_SCALER:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if USE_SCALER:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    if step % LOG_EVERY == 0:
        dt = time.time() - t0
        tps = tokens_per_step * LOG_EVERY / dt
        remaining = (MAX_STEPS - step) * (dt / LOG_EVERY)
        print(f"step {step}/{MAX_STEPS} loss {accum_loss:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"{tps:,.0f} tok/s | ETA {remaining/3600:.1f}h")
        t0 = time.time()

    if step % EVAL_EVERY == 0:
        val_loss = evaluate()
        print(f"[eval] step {step} val_loss {val_loss:.4f}")

    if step % SAVE_EVERY == 0:
        save_checkpoint(step)

print("Training complete.")
save_checkpoint(MAX_STEPS)
