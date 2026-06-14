"""
Section 2: Using the Python Interpreter
========================================

Invocation:
  python3                        # REPL
  python3 script.py              # run file
  python3 -c "print('hello')"    # one-liner
  python3 -m module_name         # run module as script (e.g. -m pytest, -m http.server)
  python3 -i script.py           # run then drop into REPL (great for debugging)

Key flags for engineers:
  -O   : strip assert statements and __debug__ blocks (minor optimization)
  -OO  : also strip docstrings
  -u   : unbuffered stdout/stderr (critical in Docker/subprocess pipelines)
  -X   : dev-mode flags, e.g. -X tracemalloc, -X dev (enables extra checks)
  -W   : warning filters, e.g. -W error::DeprecationWarning

Source encoding:
  Default UTF-8. Override with: # -*- coding: utf-8 -*- at line 1 (rarely needed now)

REPL — Read-Eval-Print Loop
===========================
The REPL is a stateful session of the interpreter: each line is compiled to
bytecode, executed, and the result printed (if not None). State accumulates
— variables, imports, function definitions persist across lines.

HOW IT WORKS INTERNALLY:
  1. Read   : reads a line (or block) — detects incomplete input (open parens,
              colons at end) and prompts with '...' for continuation
  2. Eval   : compiles to bytecode, executes in the __main__ module's namespace
  3. Print  : if the result is not None, calls repr() on it and prints
  4. Loop   : repeat

The namespace is just a dict: globals() in the REPL == {'__name__': '__main__', ...}
Every name you bind goes into that dict. Exit clears it entirely.

REPL VARIANTS:
  python3          : standard CPython REPL
  python3 -i foo.py: run script, then drop into REPL with its state loaded
                     (great for post-mortem inspection of a script's objects)
  ipython          : enhanced REPL — syntax highlighting, magic commands,
                     shell integration, better tracebacks, history search
  jupyter notebook : REPL cells in a browser — persistent, shareable, visual
  ptpython         : drop-in replacement with autocomplete and multiline editing

KEY BUILT-INS FOR EXPLORATION:
  _           : last non-None result (CPython REPL only, not in scripts)
  __          : second-to-last result
  ___         : third-to-last result

  help(obj)   : rendered docstring — help(list.sort), help('for')
  dir(obj)    : sorted list of attributes/methods — filters dunder by default
  vars(obj)   : obj.__dict__ — the actual instance/module namespace as a dict
  type(obj)   : exact type — type([])==list, type(type)==type (metaclass)
  id(obj)     : CPython: memory address — useful for identity debugging
  repr(obj)   : what the REPL prints — should be eval()-able ideally
  callable(obj): True if obj has __call__
  hasattr/getattr/setattr/delattr: runtime attribute access by name string

INSPECTION PATTERN (the "what is this object" workflow):
  1. type(x)              — what is it?
  2. dir(x)               — what can it do?
  3. vars(x) or x.__dict__— what data does it hold? (for instances/modules)
  4. help(x.method)       — how does this method work?
  5. import inspect; inspect.getsource(x) — read the actual source

REPL-SPECIFIC TRICKS:
  import readline          : enables history, Ctrl-R reverse search (auto on most systems)
  import rlcompleter; readline.parse_and_bind("tab: complete")  : tab completion

  # Suppress output: assign to _ explicitly
  _ = some_noisy_call()   # result stored but not printed

  # Multiline in REPL: open a block (def/for/if/with), hit Enter, indent,
  # then hit Enter twice (blank line) to close and execute

  # Quick timing (no import needed in IPython: just %timeit)
  import timeit
  timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)

USEFUL INSPECT PATTERNS:
  import inspect
  inspect.signature(fn)       # full signature including defaults
  inspect.getsource(fn)       # source code as string
  inspect.getmembers(obj, predicate=inspect.ismethod)  # filtered member list
  inspect.isclass / isfunction / isbuiltin / iscoroutinefunction

  # Check if something is a coroutine function (async def)
  import asyncio
  asyncio.iscoroutinefunction(fn)
"""

# -m is the most underused flag. Examples of what it unlocks:
import subprocess

examples = {
    "run tests":         "python3 -m pytest",
    "serve local files": "python3 -m http.server 8080",
    "profile script":    "python3 -m cProfile -s cumulative script.py",
    "check JSON":        "echo '{\"a\":1}' | python3 -m json.tool",
    "time import":       "python3 -X importtime -c 'import torch'",
    "pdb debugger":      "python3 -m pdb script.py",
    "pip":               "python3 -m pip install package",
}

for use_case, cmd in examples.items():
    print(f"  {use_case:<20} →  {cmd}")

# ── INSPECTION DEMO ───────────────────────────────────────────────────────────
print("\n=== OBJECT INSPECTION WORKFLOW ===")

import inspect

# Subject: a realistic object to explore
class EmbeddingModel:
    """Wraps a text embedding model with caching."""
    def __init__(self, model_name: str, dim: int = 512):
        self.model_name = model_name
        self.dim = dim
        self._cache: dict = {}

    def encode(self, text: str) -> list[float]:
        """Encode text to a fixed-size vector."""
        ...

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Async batch encoding."""
        ...

model = EmbeddingModel("text-embedding-3-small", dim=1536)

print(f"type(model)        : {type(model)}")
print(f"type(model).__mro__ : {[c.__name__ for c in type(model).__mro__]}")
print(f"vars(model)         : {vars(model)}")        # instance __dict__
print(f"id(model)           : {id(model):#x}")       # memory address (hex)

print("\n--- dir() filtered (no dunders) ---")
public = [a for a in dir(model) if not a.startswith('_')]
print(public)

print("\n--- inspect.signature ---")
print(inspect.signature(model.encode))

print("\n--- inspect.getsource ---")
print(inspect.getsource(EmbeddingModel.encode))

print("\n--- callable / coroutine checks ---")
print(f"callable(model.encode)       : {callable(model.encode)}")
print(f"iscoroutinefunction(encode)  : {inspect.iscoroutinefunction(model.encode)}")
print(f"iscoroutinefunction(enc_batch): {inspect.iscoroutinefunction(model.encode_batch)}")

# ── _ (LAST RESULT) SIMULATION ────────────────────────────────────────────────
print("\n=== _ IN SCRIPTS (simulating REPL behavior) ===")
# In the REPL, _ is set automatically. In scripts you manage it yourself.
# This shows what the REPL does under the hood:

import sys
namespace = {}
statements = [
    "x = [1, 2, 3]",       # assignment — no output (_  unchanged)
    "[i**2 for i in x]",    # expression — result printed, _ = result
    "len(x)",               # expression
    "None",                 # None — not printed, _ unchanged
]
for stmt in statements:
    try:
        result = eval(stmt, namespace)
        if result is not None:
            namespace['_'] = result
            print(f"  >>> {stmt}")
            print(f"  {repr(result)}")
        else:
            print(f"  >>> {stmt}  # None — not printed")
    except SyntaxError:
        exec(stmt, namespace)
        print(f"  >>> {stmt}  # statement — no return value")

print(f"\n  _ = {namespace.get('_')!r}")

# ── TIMEIT QUICK BENCHMARK ────────────────────────────────────────────────────
print("\n=== QUICK TIMING (timeit) ===")
import timeit

# Compare two approaches for the same operation
join_gen  = timeit.timeit('"-".join(str(n) for n in range(100))', number=50_000)
join_list = timeit.timeit('"-".join([str(n) for n in range(100)])', number=50_000)
print(f"  generator expression : {join_gen:.3f}s")
print(f"  list comprehension   : {join_list:.3f}s")
print(f"  list is {join_gen/join_list:.2f}x faster (list comp avoids generator overhead for small N)")
