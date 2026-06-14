"""
Section 5: Data Structures
============================
Python's built-in collection types — deeper than the intro.

INTERNAL IMPLEMENTATIONS
------------------------
list    : C array of PyObject* pointers. Over-allocates capacity (growth factor
          ~1.125x) to amortize append to O(1). insert/delete at index 0 is O(n)
          — shifts all pointers. Random access is O(1).

tuple   : Fixed-length C array of PyObject* pointers allocated in one block.
          No over-allocation. Immutable → CPython can intern small tuples and
          use them as dict/set keys directly.

dict    : Open-addressing hash table (compact layout since 3.6). Stores a
          separate array of (hash, key, value) entries in insertion order, with
          a sparse indices array for lookups. Average O(1) get/set; worst-case
          O(n) on hash collisions. Load factor ~2/3 before resize.

set /
frozenset: Same open-addressing hash table as dict but stores only keys (no
           values). frozenset is immutable → hashable → usable as a dict key.

deque   : Doubly-linked list of fixed-size blocks (not individual nodes).
          O(1) append/pop at both ends. O(n) random access — not a replacement
          for list when you need indexing.

FOCUS AREAS
-----------
1. List methods & comprehensions  — including nested, conditional
2. Tuples                          — immutability, packing/unpacking, named tuples
3. Sets                            — operations, frozenset
4. Dicts                           — power features, ordering guarantee, merge ops
5. Looping techniques              — enumerate, zip, zip(strict=), items(), sorted()
6. del statement                   — what it actually does
"""

# ── LIST COMPREHENSIONS ───────────────────────────────────────────────────────
print("=== LIST COMPREHENSIONS ===")

# Basic form: [expr for var in iterable if condition]
squares = [x**2 for x in range(10) if x % 2 == 0]
print(squares)

# Nested: outer loop first, inner loop second (reads like nested for)
matrix = [[1,2,3],[4,5,6],[7,8,9]]
flat = [x for row in matrix for x in row]
print(f"flat: {flat}")

# Transpose without zip (comprehension form)
transposed = [[row[i] for row in matrix] for i in range(3)]
print(f"transposed: {transposed}")

# Generator expression — same syntax, lazy (use for large data)
total = sum(x**2 for x in range(10**6))   # no intermediate list
print(f"sum of squares: {total}")

# Dict and set comprehensions
word_len = {w: len(w) for w in ["apple", "banana", "cherry"]}
print(f"word lengths: {word_len}")

unique_lens = {len(w) for w in ["apple", "banana", "cherry", "date"]}
print(f"unique lengths: {unique_lens}")

# ── DEL STATEMENT ─────────────────────────────────────────────────────────────
print("\n=== DEL ===")
# del removes a NAME binding, not necessarily the object (GC handles that)
# Can also delete slice of list (mutates in-place), or dict key

lst = list(range(10))
del lst[2:5]          # removes indices 2,3,4
print(f"after del slice: {lst}")

d = {"a": 1, "b": 2, "c": 3}
del d["b"]
print(f"after del key: {d}")

# del x just unbinds the name — the object lives on if other refs exist
a = [1, 2, 3]
b = a
del a            # 'a' is gone but [1,2,3] still lives via 'b'
print(f"b still alive: {b}")

# ── TUPLES ───────────────────────────────────────────────────────────────────
print("\n=== TUPLES ===")

# Immutable sequences — hashable if contents are hashable (usable as dict keys)
point = (3, 4)
x, y = point        # unpacking
print(f"x={x}, y={y}")

# Extended unpacking (3.x)
first, *middle, last = range(10)
print(f"first={first}, middle={middle}, last={last}")

# Single-element tuple needs trailing comma
not_tuple = (1)     # just int
is_tuple  = (1,)    # actual tuple
print(f"types: {type(not_tuple)}, {type(is_tuple)}")

# Named tuples — lightweight, no overhead vs tuple, readable fields
from collections import namedtuple
from typing import NamedTuple

# Modern typed NamedTuple (preferred)
class ModelConfig(NamedTuple):
    name: str
    hidden_dim: int
    num_layers: int
    dropout: float = 0.1

cfg = ModelConfig("transformer", 512, 6)
print(f"config: {cfg}")
print(f"name={cfg.name}, as dict={cfg._asdict()}")
print(f"replace: {cfg._replace(dropout=0.2)}")

# ── SETS ──────────────────────────────────────────────────────────────────────
print("\n=== SETS ===")

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}

print(f"union        a|b : {a | b}")
print(f"intersection a&b : {a & b}")
print(f"difference   a-b : {a - b}")
print(f"sym diff     a^b : {a ^ b}")  # in one but not both

# Set methods vs operators: methods accept any iterable, operators need sets
a.update([6, 7, 8])    # in-place union with any iterable
print(f"after update: {a}")

# frozenset — immutable set, hashable, usable as dict key
seen = frozenset(["gpt-4", "claude", "gemini"])
cache = {seen: "already processed"}   # frozenset as dict key

# Practical: fast membership test, deduplication
tokens = ["the", "cat", "sat", "on", "the", "mat", "the"]
vocab = set(tokens)
print(f"vocab size: {len(vocab)}, unique: {sorted(vocab)}")

# ── DICTIONARIES ─────────────────────────────────────────────────────────────
print("\n=== DICTS ===")

# Guaranteed insertion-order since 3.7 (CPython 3.6+)
# dict is O(1) average for get/set/delete

# Multiple construction forms
d1 = dict(a=1, b=2, c=3)
d2 = dict([("x", 10), ("y", 20)])
d3 = {k: v for k, v in zip("abcde", range(5))}

# Merge operator (3.9+)
merged = d1 | d2              # new dict
d1 |= {"d": 4, "e": 5}       # in-place merge
print(f"merged: {d1}")

# Useful dict methods
config = {"lr": 1e-3, "epochs": 10, "batch_size": 32}
print(f"get with default: {config.get('weight_decay', 1e-5)}")

# setdefault — insert only if key missing
config.setdefault("weight_decay", 1e-5)
print(f"after setdefault: {config['weight_decay']}")

# pop with default
removed = config.pop("nonexistent", None)

# dict views are live — they reflect mutations
keys_view = config.keys()
config["new_key"] = 99
print(f"live view sees new key: {'new_key' in keys_view}")  # True

# ── LOOPING TECHNIQUES ────────────────────────────────────────────────────────
print("\n=== LOOPING TECHNIQUES ===")

# enumerate — index + value, cleaner than range(len(...))
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits, start=1):
    print(f"  {i}. {fruit}")

# zip — parallel iteration; stops at shortest
names  = ["alice", "bob", "carol"]
scores = [95, 87, 92, 100]   # longer — extra item silently dropped by zip

for name, score in zip(names, scores):
    print(f"  {name}: {score}")

# zip(strict=True) — 3.10+, raises ValueError on length mismatch (safer)
try:
    list(zip(names, scores, strict=True))
except ValueError as e:
    print(f"  strict zip caught mismatch: {e}")

# items(), sorted(), reversed()
metrics = {"loss": 0.34, "accuracy": 0.91, "f1": 0.88}
for key, val in sorted(metrics.items(), key=lambda kv: kv[1], reverse=True):
    print(f"  {key}: {val:.3f}")

# itertools for power looping
import itertools

# chain — flatten iterables without materializing
combined = list(itertools.chain([1,2], [3,4], [5,6]))
print(f"chain: {combined}")

# islice — lazy slice of any iterable
import itertools
first_5_evens = list(itertools.islice((x for x in itertools.count() if x%2==0), 5))
print(f"first 5 evens: {first_5_evens}")

# groupby — consecutive grouping (sort first!)
data = sorted([("model","gpt"), ("type","llm"), ("model","claude"), ("type","vlm")], key=lambda x: x[0])
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(f"  {key}: {[v for _, v in group]}")
