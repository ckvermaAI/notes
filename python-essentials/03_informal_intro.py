"""
Section 3: An Informal Introduction to Python
===============================================
Covers: numbers, strings, lists — Python's three workhorses.
Skipping absolute basics; focusing on Python-specific gotchas and power features.

NUMBERS
-------
- int  : arbitrary precision (no overflow — ever)
- float: IEEE 754 double (64-bit). 0.1 + 0.2 != 0.3 — classic.
- complex: built-in! 3+4j, no import needed
- Useful: int('0xff', 16), int('0b1010', 2) — base-aware parsing
- // is floor division (truncates toward -inf, not zero): -7 // 2 == -4
- ** is exponentiation: 2**32 is fine, 2**1000 is fine (arbitrary precision int)
- Numeric tower: int -> float -> complex (implicit widening)

ARBITRARY PRECISION INTEGERS (how it works)
--------------------------------------------
CPython's int is a variable-length array of 30-bit limbs (on 64-bit systems),
NOT a fixed-width register value like C's int64.

Internal struct layout:
  ob_digit[]  — array of 30-bit chunks stored in base 2^30, little-endian
  ob_size     — number of limbs in use (negative value = negative number)

  value: 9      → ob_digit=[9],       ob_size=1   (fits in one limb)
  value: 2^30   → ob_digit=[0, 1],    ob_size=2   (overflows one limb)
  value: 2^128  → ob_digit=[0,0,0,0,1], ob_size=5

Why 30-bit limbs (not 32 or 64)?
  Multiplying two 30-bit limbs → 60-bit result, fits in a 64-bit register with
  room for carry. 32-bit limbs would overflow on some platforms.

Arithmetic:
  Addition     : schoolbook column add across limbs, propagate carry — O(n)
  Multiplication: schoolbook O(n²) for small; Karatsuba O(n^1.585) for large
  Every op allocates a new ob_digit[] array — no in-place mutation.

Small int cache:
  CPython pre-allocates singletons for -5 to 256 at startup.
    a = 100; b = 100; a is b   → True  (same cached object)
    a = 1000; b = 1000; a is b → False (freshly allocated)
  Never use `is` for integer equality — cache range is an implementation detail.

Memory cost (sys.getsizeof):
  0       → 24 bytes  (base object overhead)
  2^30    → 28 bytes  (1 limb, +4 bytes)
  2^60    → 32 bytes  (2 limbs)
  2^90    → 36 bytes  (3 limbs)   — +4 bytes per additional 30-bit limb

AI/research note:
  Once inside NumPy/PyTorch you're using fixed-width int32/int64 dtype arrays
  — hardware speed, no arbitrary precision overhead. Python ints matter for
  indexing, pure-Python loops, and crypto/hashing work.

STRINGS
-------
- Immutable sequences of Unicode code points (not bytes!)
- Raw strings: r"C:\new\file" — backslashes are literal
- Byte strings: b"data" — different type, different methods
- f-strings (3.6+, preferred): f"{value!r:.2f}" — format spec + conversion
- Multiline: triple-quote '''...''' or  \"\"\"...\"\"\"
- str is NOT bytes. Encode/decode explicitly: s.encode('utf-8'), b.decode()

LISTS
-----
- Mutable, heterogeneous sequences
- Negative indexing: lst[-1] is last element
- Slicing: lst[start:stop:step] — all optional, all Pythonic
- Slices return NEW lists (shallow copy)
- lst[:] — full shallow copy idiom
- Mutability: lst[1:3] = [10, 20, 30] — slice assignment (can change length!)
"""

# ── NUMBERS ──────────────────────────────────────────────────────────────────
print("=== NUMBERS ===")

# Arbitrary precision int
huge = 2 ** 128
print(f"2^128 = {huge}")  # No overflow

# Floor division vs truncation
print(f"-7 // 2 = {-7 // 2}")   # -4  (floor toward -inf)
print(f"-7 / 2  = {-7 / 2}")    # -3.5

# Complex — no import
z = 3 + 4j
print(f"complex: {z}, magnitude: {abs(z)}, conjugate: {z.conjugate()}")

# Float gotcha — always relevant in ML loss comparisons
print(f"0.1 + 0.2 == 0.3 → {0.1 + 0.2 == 0.3}")   # False
print(f"round(0.1+0.2, 10) == 0.3 → {round(0.1+0.2, 10) == 0.3}")  # True

# ── STRINGS ──────────────────────────────────────────────────────────────────
print("\n=== STRINGS ===")

# f-string power: format spec + repr conversion
pi = 3.14159265
print(f"pi = {pi:.4f}")          # 3.1416
print(f"pi repr = {pi!r}")       # repr() applied
print(f"pi as 10-wide: {pi:>10.3f}")  # right-aligned

# Raw strings — essential for regex and Windows paths
import re
pattern = r"\bword\b"   # no need to double-escape
print(re.findall(pattern, "a word here"))

# Strings are sequences — slicing works
s = "Hello, Python!"
print(s[:5])        # Hello
print(s[-7:])       # Python!
print(s[::2])       # every other char

# encode/decode — critical for network/file I/O
raw = "café"
encoded = raw.encode("utf-8")
print(f"'{raw}' → {encoded} → '{encoded.decode('utf-8')}'")

# ── LISTS ────────────────────────────────────────────────────────────────────
print("\n=== LISTS ===")

lst = [1, 2, 3, 4, 5]

# Slice assignment — mutates in place, can resize
lst[1:3] = [20, 30, 40]   # replaces 2 elements with 3
print(f"after slice assign: {lst}")  # [1, 20, 30, 40, 4, 5]

# Shallow copy idiom
copy = lst[:]
copy[0] = 999
print(f"original unchanged: {lst[0]}")  # 1

# Nested list — gotcha: don't use [[]] * n for mutable rows
wrong = [[0] * 3] * 3   # all rows are THE SAME object
wrong[0][0] = 1
print(f"wrong 2D init: {wrong}")   # all rows affected!

correct = [[0] * 3 for _ in range(3)]
correct[0][0] = 1
print(f"correct 2D init: {correct}")  # only row 0 affected

# List as stack (append/pop) and queue (use collections.deque for O(1) popleft)
from collections import deque
q = deque([1, 2, 3])
q.append(4)
q.appendleft(0)
print(f"deque: {q}, popleft: {q.popleft()}, remaining: {q}")
