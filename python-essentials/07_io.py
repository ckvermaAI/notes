"""
Section 7: Input and Output
=============================
FOCUS: file I/O, serialization, pathlib, in-memory streams — the practical
patterns for data pipelines and model artifact management.

Skipping: print() formatting (covered in §3 f-strings), input().

FILE I/O
--------
open(path, mode, encoding, buffering)
  modes : 'r' read (default), 'w' write (truncate), 'a' append,
          'x' exclusive create (fails if exists), 'b' binary, '+' read+write
  Always use a context manager (with open(...)) — guarantees close() on exception.
  Always specify encoding='utf-8' for text mode — default is platform-dependent.

SERIALIZATION QUICK REFERENCE
-------------------------------
  json    : human-readable, interoperable, strings/numbers/lists/dicts only
  pickle  : Python-native, arbitrary objects, NOT safe to unpickle untrusted data
  msgpack : binary JSON — faster, smaller, cross-language (pip install msgpack)
  numpy   : np.save/.load for arrays, np.savez for multiple arrays
  torch   : torch.save/load (wraps pickle; use weights_only=True for safety)
  h5py    : HDF5 — hierarchical, lazy-loaded, good for large datasets
  parquet : columnar, compressed, fast for tabular ML data (via pyarrow/pandas)

PATHLIB vs os.path
------------------
  pathlib.Path is the modern API (3.4+). Prefer it over os.path string concat.
  Path('/data') / 'models' / 'v1.pt'   ← operator overloading for join
  p.stem, p.suffix, p.parent, p.name, p.exists(), p.glob('**/*.pt')
"""

import json
import pickle
import io
from pathlib import Path
import tempfile
import os

# ── PATHLIB ───────────────────────────────────────────────────────────────────
print("=== pathlib ===")

base = Path("/tmp/ml_demo")
base.mkdir(exist_ok=True)

# Path composition with /
checkpoint_dir = base / "checkpoints" / "v1"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

model_path = checkpoint_dir / "model.pt"
config_path = checkpoint_dir / "config.json"

print(f"path     : {model_path}")
print(f"stem     : {model_path.stem}")       # 'model'
print(f"suffix   : {model_path.suffix}")     # '.pt'
print(f"parent   : {model_path.parent}")
print(f"parts    : {model_path.parts}")

# glob — find all checkpoints
(base / "checkpoints" / "v1").mkdir(parents=True, exist_ok=True)
(base / "checkpoints" / "v2").mkdir(parents=True, exist_ok=True)
for subdir in ["v1", "v2"]:
    (base / "checkpoints" / subdir / "model.pt").touch()

checkpoints = sorted(base.glob("checkpoints/**/*.pt"))
print(f"found checkpoints: {[str(p) for p in checkpoints]}")

# ── TEXT FILE I/O ─────────────────────────────────────────────────────────────
print("\n=== Text file I/O ===")

log_path = base / "training.log"

# Write
with open(log_path, 'w', encoding='utf-8') as f:
    for epoch, loss in enumerate([0.92, 0.71, 0.54, 0.43], 1):
        f.write(f"epoch={epoch} loss={loss:.4f}\n")

# Read all at once
with open(log_path, encoding='utf-8') as f:
    content = f.read()
print(content.strip())

# Read line by line — memory efficient for large logs
with open(log_path, encoding='utf-8') as f:
    last_line = None
    for line in f:          # file object is an iterator — no readlines() needed
        last_line = line.strip()
print(f"last line: {last_line}")

# ── JSON ──────────────────────────────────────────────────────────────────────
print("\n=== JSON ===")

config = {
    "model": "transformer",
    "hidden_dim": 512,
    "num_layers": 6,
    "optimizer": {"type": "adam", "lr": 3e-4, "betas": [0.9, 0.999]},
    "tags": ["nlp", "classification"],
}

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

with open(config_path, encoding='utf-8') as f:
    loaded = json.load(f)

print(f"round-trip: {loaded == config}")
print(f"lr: {loaded['optimizer']['lr']}")

# json.dumps / json.loads for strings (HTTP payloads, Redis, etc.)
payload = json.dumps(config, separators=(',', ':'))   # compact, no whitespace
print(f"compact size: {len(payload)} bytes")

# Custom serialization for non-standard types
import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

event = {"ts": datetime.datetime(2026, 6, 13, 10, 0), "event": "train_start"}
print(json.dumps(event, cls=DateTimeEncoder))

# ── PICKLE ────────────────────────────────────────────────────────────────────
print("\n=== Pickle ===")
# Use for: sklearn models, tokenizers, arbitrary Python objects
# NEVER unpickle untrusted data — it executes arbitrary code on load

data = {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "labels": [0, 1]}
pkl_path = base / "cache.pkl"

with open(pkl_path, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(pkl_path, 'rb') as f:
    loaded = pickle.load(f)

print(f"pickle round-trip: {loaded == data}")
print(f"pickle protocols: highest={pickle.HIGHEST_PROTOCOL}, default={pickle.DEFAULT_PROTOCOL}")

# pickle.dumps/loads for in-memory (e.g. multiprocessing IPC)
blob = pickle.dumps(config)
print(f"pickled config: {len(blob)} bytes")

# ── IN-MEMORY STREAMS (StringIO / BytesIO) ────────────────────────────────────
print("\n=== In-memory streams ===")
# Same API as file objects but backed by memory — great for testing I/O code
# without touching the filesystem

buf = io.StringIO()
json.dump({"status": "ok", "score": 0.97}, buf)
buf.seek(0)
print(f"StringIO contents: {buf.read()}")

# BytesIO — common for image data, model bytes, HTTP responses
import struct
byte_buf = io.BytesIO()
byte_buf.write(struct.pack('>4f', 1.0, 2.0, 3.0, 4.0))   # 4 big-endian floats
byte_buf.seek(0)
values = struct.unpack('>4f', byte_buf.read())
print(f"BytesIO float round-trip: {values}")

# ── TEMPFILE ──────────────────────────────────────────────────────────────────
print("\n=== tempfile ===")
# Use in tests and pipelines — auto-cleaned up

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump({"tmp": True}, f)
    tmp_path = f.name

print(f"wrote to temp: {tmp_path}")
os.unlink(tmp_path)   # manual cleanup since delete=False

# TemporaryDirectory — entire dir auto-cleaned
with tempfile.TemporaryDirectory() as tmpdir:
    p = Path(tmpdir) / "artifact.bin"
    p.write_bytes(b"\x00\x01\x02")
    print(f"temp dir: {tmpdir}, exists during: {p.exists()}")
print(f"temp dir cleaned up: {not Path(tmpdir).exists()}")

# Cleanup demo dir
import shutil
shutil.rmtree(base)
print(f"\ncleaned up {base}")
