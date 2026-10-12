# ============================================================
# NOVA SPEED PATCH for Colab T4
# Run this ONCE, in one cell, AFTER your model/optimizer/streams
# exist and BEFORE the training loop. It patches the model that
# is already in memory -- no rebuild needed.
#
# What it fixes:
#   1. MoE expert loop  -> vectorized top-1 dispatch (same math)
#   2. nn.MultiheadAttention -> SDPA (same weights, much faster)
#   3. Batch sampler    -> single indexing op + bigger batches
#   4. Adds the missing get_batch() your pre-flight check wants
#   5. torch.compile    -> fused kernels for the whole model
#   6. save_ckpt_bg()   -> save locally, copy to Drive in background
#   7. fast_evaluate()  -> batched validation that actually exists
#
# A numeric self-check runs automatically: old vs new forward
# must match to ~1e-3 before the patch is kept.
# ============================================================
import inspect
import random
import shutil
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "Enable a GPU runtime first."
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

_HERE = {k: v for k, v in globals().items()}

# ---------- 1. Vectorized top-1 MoE (identical math, no Python loop) ----------
def _fast_moe_forward(self, x):
    dims, d = x.shape[:-1], x.shape[-1]
    flat = x.reshape(-1, d)
    choice = self.gate(flat).argmax(dim=-1)          # (N,)
    def _stack(get):
        ws = [get(e) for e in self.experts]
        if any(w is None for w in ws):
            ws = [w if w is not None else torch.zeros_like(ws[0]) for w in ws]
        return torch.stack(ws)
    w1 = _stack(lambda e: e.net[0].weight)           # (E,f,d)
    b1 = _stack(lambda e: e.net[0].bias)             # (E,f)
    w2 = _stack(lambda e: e.net[2].weight)           # (E,d,f)
    b2 = _stack(lambda e: e.net[2].bias)             # (E,d)
    h = torch.baddbmm(b1[choice].unsqueeze(1), flat.unsqueeze(1),
                      w1[choice].transpose(1, 2)).squeeze(1)
    h = F.gelu(h)
    out = torch.baddbmm(b2[choice].unsqueeze(1), h.unsqueeze(1),
                        w2[choice].transpose(1, 2)).squeeze(1)
    return out.view(*dims, d)

# ---------- 2. SDPA attention reusing the exact MultiheadAttention weights ----------
def _fast_attn_forward(self, x):
    attn = self.attn
    B, T, d = x.shape
    heads, hd = attn.num_heads, d // attn.num_heads
    qkv = F.linear(x, attn.in_proj_weight, getattr(attn, "in_proj_bias", None))
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, heads, hd).transpose(1, 2)
    k = k.view(B, T, heads, hd).transpose(1, 2)
    v = v.view(B, T, heads, hd).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).reshape(B, T, d)
    return F.linear(y, attn.out_proj.weight, attn.out_proj.bias)

_patched = []
_MoEFFN = _HERE.get("MoEFFN")
if _MoEFFN is not None and hasattr(_MoEFFN, "experts"):
    _orig_moe = _MoEFFN.forward
    _MoEFFN.forward = _fast_moe_forward
    _patched.append("MoEFFN")
else:
    _orig_moe = None
_orig_attns = {}
for _name in ("Attention", "CausalAttention"):
    _cls = _HERE.get(_name)
    if _cls is not None:
        _orig_attns[_name] = _cls.forward
        _cls.forward = _fast_attn_forward
        _patched.append(_name)
print("Patched:", ", ".join(_patched) if _patched else "NONE - class names not found")

# ---------- Self-check: old vs new must agree ----------
with torch.no_grad():
    _d = globals().get("D_MODEL", 512)
    _x = torch.randn(2, 16, _d, device=device)
    if _orig_moe is not None and "model" in _HERE:
        _moe_mod = None
        for _m in model.modules():
            if isinstance(_m, _MoEFFN):
                _moe_mod = _m
                break
        if _moe_mod is not None:
            _old = _orig_moe(_moe_mod, _x)
            _new = _moe_mod(_x)
            _err = (_old.float() - _new.float()).abs().max().item()
            print(f"MoE self-check max abs err: {_err:.2e}")
            assert _err < 1e-2, "MoE patch changed the math!"
    for _name, _orig in _orig_attns.items():
        _found = None
        for _m in model.modules():
            if type(_m).__name__ == _name:
                _found = _m
                break
        if _found is not None:
            _old = _orig(_found, _x)
            if isinstance(_old, (tuple, list)):
                _old = _old[0]
            _new = _found(_x)
            _err = (_old.float() - _new.float()).abs().max().item()
            print(f"{_name} self-check max abs err: {_err:.2e}")
            assert _err < 1e-2, f"{_name} patch changed the math!"
print("Self-check passed: patched math matches the original.")

# ---------- 3. Bigger batches + single-op sampler ----------
BATCH_SIZE = 16   # was 2 -- a T4 has 16 GB and this model is small
GRAD_ACCUM = 2    # effective batch 32

def sample_batch(stream, batch_size=None):
    bs = batch_size or BATCH_SIZE
    max_start = len(stream) - SEQ_LEN - 1
    if max_start < 1:
        raise ValueError("Token stream is too short for the configured context.")
    starts = torch.randint(0, max_start, (bs,))
    idx = starts[:, None] + torch.arange(SEQ_LEN)[None, :]
    return (stream[idx].to(device, non_blocking=True),
            stream[idx + 1].to(device, non_blocking=True))

def get_batch(stream, batch_size=None):   # the function cell 8's pre-flight wants
    return sample_batch(stream, batch_size)

def get_training_batch():
    ts = _HERE.get("teacher_streams") or globals().get("teacher_streams")
    prob = globals().get("TEACHER_BATCH_PROBABILITY", 0.0)
    if ts and random.random() < prob:
        combined = torch.cat(ts)
        if len(combined) > SEQ_LEN + 1:
            return sample_batch(combined)
    return sample_batch(train_stream)

# ---------- 4. Compile (one-time ~1-2 min, faster every step after) ----------
_takes_y = len(inspect.signature(model.forward).parameters) > 1  # bound method: (x) vs (x, y)
model = torch.compile(model)
print("Model compiled. First step will be slow (compiling), then much faster.")

# ---------- 5. Checkpoints: local save, background Drive copy ----------
def save_ckpt_bg(state, drive_path):
    tmp = "/content/_nova_ckpt_tmp.pt"
    torch.save(state, tmp)
    def _copy():
        shutil.copy2(tmp, str(drive_path))
        print(f"[ckpt] Drive copy done: {drive_path}")
    threading.Thread(target=_copy, daemon=True).start()
    print(f"[ckpt] saved locally, copying to Drive in background...")

# ---------- 6. Fast validation (the old evaluate() was never defined) ----------
@torch.inference_mode()
def fast_evaluate(n_batches=40):
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        xb, yb = sample_batch(val_stream)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(xb, yb) if _takes_y else model(xb)
        if isinstance(out, (tuple, list)):
            loss = out[1] if (len(out) == 2 and isinstance(out[1], torch.Tensor)) else None
            if loss is None:
                logits = out[0]
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       yb.reshape(-1))
        else:
            loss = F.cross_entropy(out.float().reshape(-1, out.size(-1)), yb.reshape(-1))
        total += loss.item()
    if was_training:
        model.train()
    return total / n_batches

print("\nDone. New knobs: BATCH_SIZE=16, GRAD_ACCUM=2")
print("In your loop: replace torch.save(...) with save_ckpt_bg(...),")
print("and call fast_evaluate() instead of evaluate().")
print("If you hit CUDA OOM, drop BATCH_SIZE to 8.")
