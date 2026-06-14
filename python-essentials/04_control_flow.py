"""
Section 4: More Control Flow Tools
====================================
Skipping: if/elif/else, basic for, break/continue (you know these).

FOCUS AREAS:
1. for loop semantics  — iterates over any iterable, not index-based
2. range()             — lazy sequence, not a list
3. Loop else clause    — underused, genuinely useful
4. Function signatures — positional-only (/), keyword-only (*), *args/**kwargs
5. Annotations         — type hints, __annotations__
"""

# ── FOR LOOP SEMANTICS ────────────────────────────────────────────────────────
print("=== FOR / RANGE ===")

# Modifying a list while iterating → iterate over a copy
items = [1, 2, 3, 4, 5]
for x in items[:]:          # iterate copy, mutate original safely
    if x % 2 == 0:
        items.remove(x)
print(items)                # [1, 3, 5]

# range() is lazy — O(1) memory regardless of size
r = range(0, 10**9, 3)
print(f"range object: {r}, len={len(r)}, last={r[-1]}")  # all O(1)

# ── LOOP ELSE ─────────────────────────────────────────────────────────────────
print("\n=== LOOP ELSE ===")
# else executes if the loop completes WITHOUT hitting break
# Classic use: search-with-fallback (avoids a flag variable)

def find_prime_factor(n):
    for i in range(2, n):
        if n % i == 0:
            print(f"{n} is divisible by {i}")
            break
    else:
        print(f"{n} is prime")   # only runs if no break

find_prime_factor(17)
find_prime_factor(18)

# ── FUNCTION SIGNATURES ───────────────────────────────────────────────────────
print("\n=== FUNCTION SIGNATURES ===")

# Full syntax: def f(pos_only, /, normal, *, kw_only, **kwargs)
# /  → everything before is positional-only (cannot be passed by name)
# *  → everything after is keyword-only (must be passed by name)

def process(input_data, /, *, batch_size=32, device="cpu"):
    """
    input_data  : positional-only (internal name hidden from callers)
    batch_size  : keyword-only (must say batch_size=64, not just 64)
    device      : keyword-only
    """
    return f"processing {len(input_data)} items, bs={batch_size}, dev={device}"

print(process([1,2,3], batch_size=16, device="cuda"))
# process(input_data=[1,2,3])  ← TypeError: positional-only argument

# *args and **kwargs — and forwarding them
def wrapper(*args, **kwargs):
    print(f"  args={args}, kwargs={kwargs}")
    return inner(*args, **kwargs)

def inner(a, b, c=0):
    return a + b + c

# Unpacking at call site
params = [1, 2]
opts   = {"c": 10}
print(inner(*params, **opts))    # 13

# ── ANNOTATIONS / TYPE HINTS ──────────────────────────────────────────────────
print("\n=== ANNOTATIONS ===")
# Type hints are NOT enforced at runtime — purely for tooling (mypy, pyright)
# Stored in __annotations__; evaluated lazily with 'from __future__ import annotations'

from typing import Sequence

def embed(texts: Sequence[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Type hints communicate intent; runtime ignores them."""
    ...

print(embed.__annotations__)

# Python 3.12+: type aliases with 'type' keyword
type Vector = list[float]      # PEP 695 — cleaner than TypeAlias
type Matrix = list[Vector]
print(f"Vector alias: {Vector}")
