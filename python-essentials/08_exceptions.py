"""
Section 8: Errors and Exceptions
==================================
FOCUS: exception hierarchy, custom exceptions, chaining, ExceptionGroup,
context managers — patterns for robust ML pipelines.

EXCEPTION HIERARCHY (relevant subset)
---------------------------------------
BaseException
 ├── SystemExit            ← sys.exit() — don't catch with `except Exception`
 ├── KeyboardInterrupt     ← Ctrl-C — same, don't swallow
 ├── GeneratorExit         ← generator.close() called
 └── Exception             ← everything you normally catch
      ├── ValueError        ← bad value (wrong dtype, out-of-range index)
      ├── TypeError         ← wrong type
      ├── RuntimeError      ← general (torch raises these heavily)
      ├── OSError           ← file/IO errors (FileNotFoundError, PermissionError)
      ├── KeyError          ← missing dict key
      ├── IndexError        ← list out of range
      ├── AttributeError    ← missing attribute
      ├── ImportError       ← module not found (ModuleNotFoundError is subclass)
      ├── StopIteration     ← iterator exhausted (don't raise manually in coroutines)
      └── AssertionError    ← assert statement failed (stripped with -O flag!)

KEY MECHANICS
-------------
try / except / else / finally:
  else   : runs if NO exception was raised in try — underused, good for
           "success path" logic that shouldn't be in the except handler
  finally: always runs — use for cleanup (though context managers are cleaner)

raise X from Y:
  Chains exceptions. Sets __cause__ (explicit chain) or __context__ (implicit).
  raise RuntimeError("training failed") from original_exc
  → traceback shows both, clearly linked. Use this in wrappers/adapters.

raise X from None:
  Suppresses the chained context — use when re-raising with a cleaner message
  and the original is irrelevant to the caller.

ExceptionGroup (3.11+):
  Groups multiple simultaneous exceptions — designed for async/concurrent code
  where several tasks can fail at once. Caught with `except*` syntax.
"""

# ── TRY / EXCEPT / ELSE / FINALLY ────────────────────────────────────────────
print("=== try/except/else/finally ===")

def load_config(path: str) -> dict:
    import json
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config not found: {path}") from None  # clean re-raise
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e         # chain it
    else:
        print(f"  loaded {len(data)} keys from {path}")   # only on success
        return data
    finally:
        print(f"  load_config({path!r}) completed")       # always runs

try:
    load_config("/nonexistent/config.json")
except FileNotFoundError as e:
    print(f"  caught: {e}")

# ── CUSTOM EXCEPTIONS ─────────────────────────────────────────────────────────
print("\n=== Custom exceptions ===")

# Pattern: exception hierarchy mirrors your domain
class MLError(Exception):
    """Base for all ML pipeline errors."""

class DataError(MLError):
    """Bad input data — caller should fix data, not retry."""
    def __init__(self, msg: str, offending_sample=None):
        super().__init__(msg)
        self.offending_sample = offending_sample

class TrainingError(MLError):
    """Training diverged or failed — may be retryable with different config."""
    def __init__(self, msg: str, epoch: int, loss: float):
        super().__init__(msg)
        self.epoch = epoch
        self.loss  = loss

class CheckpointError(MLError):
    """Checkpoint save/load failure."""

# Usage
def validate_batch(batch: list):
    for i, sample in enumerate(batch):
        if not isinstance(sample, dict) or 'input_ids' not in sample:
            raise DataError(
                f"Sample {i} missing 'input_ids'",
                offending_sample=sample
            )

try:
    validate_batch([{"input_ids": [1,2,3]}, {"label": 0}])
except DataError as e:
    print(f"  DataError: {e}")
    print(f"  offending_sample: {e.offending_sample}")
    print(f"  is MLError: {isinstance(e, MLError)}")

# ── EXCEPTION CHAINING ────────────────────────────────────────────────────────
print("\n=== Exception chaining (raise from) ===")

def save_checkpoint(path: str, state: dict):
    import json
    try:
        with open(path, 'w') as f:
            json.dump(state, f)
    except OSError as e:
        raise CheckpointError(f"Failed to save checkpoint to {path}") from e

try:
    save_checkpoint("/root/no_permission/ckpt.json", {"epoch": 1})
except CheckpointError as e:
    print(f"  caught: {e}")
    print(f"  caused by: {e.__cause__}")   # the original OSError
    print(f"  chain: {type(e.__cause__).__name__}")

# ── SUPPRESS — contextlib ─────────────────────────────────────────────────────
print("\n=== contextlib.suppress ===")
from contextlib import suppress
import os

# Cleaner than try/except pass for expected, ignorable errors
with suppress(FileNotFoundError):
    os.remove("/tmp/file_that_may_not_exist.tmp")
print("  suppress: no crash if file missing")

# ── EXCEPTIONGROUP (3.11+) ────────────────────────────────────────────────────
print("\n=== ExceptionGroup (3.11+) ===")

# Raised when multiple concurrent tasks fail — asyncio.gather raises this
# when return_exceptions=False and multiple tasks fail

def run_parallel_validation(samples: list) -> list:
    """Simulate catching multiple errors from parallel workers."""
    errors = []
    for i, s in enumerate(samples):
        try:
            if not isinstance(s.get('text', None), str):
                raise DataError(f"sample {i}: 'text' must be str", offending_sample=s)
            if len(s['text']) == 0:
                raise DataError(f"sample {i}: empty text", offending_sample=s)
        except DataError as e:
            errors.append(e)

    if errors:
        raise ExceptionGroup("batch validation failed", errors)
    return samples

try:
    run_parallel_validation([
        {"text": "hello"},
        {"text": 123},           # wrong type
        {"text": ""},            # empty
        {"text": "world"},
    ])
except* DataError as eg:          # except* — catches from ExceptionGroup
    print(f"  {len(eg.exceptions)} DataError(s):")
    for e in eg.exceptions:
        print(f"    - {e}")

# ── ASSERTION BEST PRACTICES ──────────────────────────────────────────────────
print("\n=== assert vs explicit raise ===")
# assert is for invariants YOU control (internal logic, unit tests)
# Raise explicitly for user/API-facing validation — asserts are stripped with -O

def set_learning_rate(lr: float):
    # BAD: assert lr > 0   ← stripped with python -O
    # GOOD:
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    return lr

try:
    set_learning_rate(-1e-3)
except ValueError as e:
    print(f"  caught: {e}")

# ── CONTEXT MANAGER VIA __enter__/__exit__ ────────────────────────────────────
print("\n=== Custom context manager ===")
import time

class Timer:
    """Context manager for timing code blocks — common in profiling."""
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self                         # bound to 'as' target

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._start
        print(f"  [{self.label}] {self.elapsed*1000:.2f}ms")
        return False   # False = don't suppress exceptions

with Timer("list comprehension") as t:
    _ = [x**2 for x in range(100_000)]
print(f"  elapsed accessible after: {t.elapsed*1000:.2f}ms")

# contextlib.contextmanager — generator-based, less boilerplate
from contextlib import contextmanager

@contextmanager
def managed_resource(name: str):
    print(f"  acquiring {name}")
    try:
        yield name.upper()       # value bound to 'as' target
    except Exception as e:
        print(f"  error during {name}: {e}")
        raise
    finally:
        print(f"  releasing {name}")

with managed_resource("gpu_memory") as res:
    print(f"  using: {res}")
