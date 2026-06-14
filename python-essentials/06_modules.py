"""
Section 6: Modules
===================
FOCUS: import system internals, package structure, dynamic imports — the parts
that bite you in large ML codebases.

IMPORT MECHANICS
----------------
import foo:
  1. Check sys.modules (cache) — return cached if found
  2. Find module: sys.meta_path finders (PathFinder, FrozenImporter, etc.)
  3. Load: compile source → bytecode (.pyc in __pycache__), execute in new namespace
  4. Bind name in current namespace

Implication: importing the same module twice is cheap (step 1 short-circuits).
Mutating a module's namespace is visible to all importers — modules are singletons.

PACKAGE STRUCTURE
-----------------
  mypackage/
    __init__.py       ← executed on 'import mypackage'; controls public API
    utils.py
    models/
      __init__.py
      transformer.py

  __init__.py controls what 'from mypackage import *' exports via __all__.
  Empty __init__.py is valid — marks directory as a package.
  Namespace packages (3.3+): directory WITHOUT __init__.py — useful for
  splitting a package across multiple directories/repos.

__all__  = ['PublicClass', 'public_fn']   # whitelist for 'import *'
           # also signals intent to readers — the package's public contract

RELATIVE IMPORTS (inside a package only)
-----------------------------------------
  from . import sibling_module        # same package
  from .. import parent_module        # parent package
  from ..utils import helper          # parent's sibling

  Relative imports only work inside packages — not in top-level scripts.
  Running a module directly (python3 pkg/module.py) breaks relative imports;
  use python3 -m pkg.module instead.

__name__ == '__main__'
-----------------------
When a file is run directly, __name__ is '__main__'.
When imported, __name__ is the module's dotted name.
Standard guard: if __name__ == '__main__': — keeps importable modules runnable.

IMPORTANT ATTRIBUTES
---------------------
  __file__    : absolute path to the .py source (or .pyc if no source)
  __spec__    : ModuleSpec — name, origin, submodule_search_locations
  __package__ : dotted package name ('' for top-level, None for __main__ run)
  __path__    : only on packages — list of directories to search for submodules
"""

import sys
import importlib
import importlib.util

# ── SYS.MODULES — module cache ────────────────────────────────────────────────
print("=== sys.modules (singleton cache) ===")
import os
print(f"os already cached: {'os' in sys.modules}")      # True after first import
print(f"same object: {sys.modules['os'] is os}")        # True — same singleton

# Reloading (rarely needed — e.g. for hot-reloading config modules)
# importlib.reload(module)  ← re-executes module code, updates existing namespace

# ── SYS.PATH — where Python looks ─────────────────────────────────────────────
print("\n=== sys.path ===")
for p in sys.path[:4]:
    print(f"  {p or '(current dir)'}")
print("  ...")
# Prepend at runtime: sys.path.insert(0, '/path/to/my/libs')
# Better: use PYTHONPATH env var or install as editable: pip install -e .

# ── DYNAMIC IMPORTS (importlib) ───────────────────────────────────────────────
print("\n=== Dynamic imports ===")

# Load a module by string name — useful for plugin systems, config-driven loading
module_name = "json"
mod = importlib.import_module(module_name)
print(f"dynamically loaded: {mod.__name__}")
print(f"json.dumps: {mod.dumps({'model': 'gpt-4', 'tokens': 1024})}")

# Load from arbitrary file path (e.g. user-supplied plugin)
def load_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod   # register so relative imports inside work
    spec.loader.exec_module(mod)
    return mod

# ── LAZY IMPORTS — speed up startup ──────────────────────────────────────────
print("\n=== Lazy imports ===")
# Heavy libraries (torch, transformers) slow startup. Two patterns:

# Pattern 1: module-level lazy (import inside function)
def get_numpy():
    import numpy as np    # only imported when first called
    return np

# Pattern 2: importlib.util.LazyLoader (transparent lazy proxy)
def lazy_import(name: str):
    spec   = importlib.util.find_spec(name)
    if spec is None:
        return None
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod

# np = lazy_import("numpy")   # numpy imported only when np.array(...) is first called

# ── __name__ GUARD ────────────────────────────────────────────────────────────
print("\n=== __name__ guard ===")
print(f"__name__ = {__name__!r}")   # '__main__' when run directly

# ── INSPECTING A MODULE ───────────────────────────────────────────────────────
print("\n=== Module introspection ===")
import json
print(f"__file__   : {json.__file__}")
print(f"__package__ : {json.__package__!r}")
public_api = [x for x in dir(json) if not x.startswith('_')]
print(f"public API : {public_api}")

# ── PLUGIN REGISTRY PATTERN (common in ML frameworks) ────────────────────────
print("\n=== Plugin / registry pattern ===")

_REGISTRY: dict[str, type] = {}

def register(name: str):
    """Decorator that registers a class by name — used in optimizers, losses, etc."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

@register("adam")
class AdamOptimizer:
    def __init__(self, lr=1e-3): self.lr = lr
    def __repr__(self): return f"Adam(lr={self.lr})"

@register("sgd")
class SGDOptimizer:
    def __init__(self, lr=1e-2): self.lr = lr
    def __repr__(self): return f"SGD(lr={self.lr})"

def build_optimizer(name: str, **kwargs):
    """Config-driven instantiation — common pattern in Hydra/yaml configs."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown optimizer {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)

opt = build_optimizer("adam", lr=3e-4)
print(f"built from config: {opt}")
print(f"registry: {_REGISTRY}")
