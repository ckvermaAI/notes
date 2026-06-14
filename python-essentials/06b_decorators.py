"""
Decorators
===========
A decorator is a callable that takes a callable and returns a callable.
The @ syntax is pure sugar — no special runtime support.

DESUGARING RULES
-----------------
  @decorator
  def foo(): ...
  ↓
  foo = decorator(foo)          # decorator called with the function

  @decorator(args)
  def foo(): ...
  ↓
  foo = decorator(args)(foo)    # two calls: factory first, then the decorator it returns

  @A
  @B
  @C
  def foo(): ...
  ↓
  foo = A(B(C(foo)))            # bottom-up application

FUNCTOOLS.WRAPS
----------------
Without it, the wrapper steals the wrapped function's identity:
  wrapped.__name__  → 'wrapper'   (broken)
  wrapped.__doc__   → None        (broken)
@functools.wraps(fn) copies __name__, __doc__, __module__,
__annotations__, __qualname__, and sets __wrapped__ = fn (allows unwrapping).
Always use it when writing wrappers intended to be transparent.
"""

import functools
import time

# ── BASIC DECORATOR ───────────────────────────────────────────────────────────
print("=== Basic decorator (desugaring) ===")

def my_decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"  before {fn.__name__}")
        result = fn(*args, **kwargs)
        print(f"  after  {fn.__name__}")
        return result
    return wrapper

@my_decorator
def train():
    """Train the model."""
    print("  training...")

train()
print(f"  __name__ preserved : {train.__name__}")
print(f"  __doc__  preserved : {train.__doc__}")
print(f"  __wrapped__        : {train.__wrapped__}")

# ── DECORATOR FACTORY (parameterized) ────────────────────────────────────────
print("\n=== Decorator factory — @register('adam') pattern ===")

_REGISTRY: dict[str, type] = {}

def register(name: str):
    """
    register("adam") → returns decorator
    decorator(cls)   → registers cls, returns cls unchanged
    """
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls          # return cls unmodified — just a side effect
    return decorator

@register("adam")
class AdamOptimizer:
    def __init__(self, lr=1e-3): self.lr = lr
    def __repr__(self): return f"Adam(lr={self.lr})"

@register("sgd")
class SGDOptimizer:
    def __init__(self, lr=1e-2): self.lr = lr
    def __repr__(self): return f"SGD(lr={self.lr})"

# Equivalent without sugar:
# AdamOptimizer = register("adam")(AdamOptimizer)

print(f"  registry: {_REGISTRY}")
print(f"  build:    {_REGISTRY['adam'](lr=3e-4)}")

# ── CLASS-BASED DECORATOR (stateful) ─────────────────────────────────────────
print("\n=== Class-based decorator (stateful) ===")

class retry:
    """Decorator class — __call__ makes instances callable."""
    def __init__(self, max_attempts: int = 3, exceptions=(Exception,)):
        self.max_attempts = max_attempts
        self.exceptions   = exceptions

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except self.exceptions as e:
                    last_exc = e
                    print(f"  attempt {attempt}/{self.max_attempts} failed: {e}")
            raise last_exc
        return wrapper

_call_count = 0

@retry(max_attempts=3)
def flaky_api_call():
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ConnectionError("timeout")
    return "success"

result = flaky_api_call()
print(f"  result: {result}")

# ── STACKING DECORATORS ───────────────────────────────────────────────────────
print("\n=== Stacking decorators (bottom-up) ===")

def log_call(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"  [log]   calling {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper

def timeit_dec(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"  [timer] {fn.__name__} took {(time.perf_counter()-t)*1000:.2f}ms")
        return result
    return wrapper

@log_call          # applied second (outermost)
@timeit_dec        # applied first (innermost)
def forward_pass(x: list) -> float:
    return sum(x) / len(x)

# Equivalent: forward_pass = log_call(timeit_dec(forward_pass))
# Call order: log_call wrapper → timeit_dec wrapper → original fn
forward_pass(list(range(1000)))

# ── PRACTICAL ML DECORATORS ───────────────────────────────────────────────────
print("\n=== Practical ML decorators ===")

# 1. torch.no_grad() equivalent — context manager used as decorator
from contextlib import contextmanager

class no_grad:
    """Disables gradient tracking (stub — real version modifies thread-local state)."""
    _enabled = True

    def __enter__(self):
        self.__class__._enabled = False
        return self

    def __exit__(self, *args):
        self.__class__._enabled = True

    def __call__(self, fn):
        """Makes no_grad() usable as @no_grad() decorator."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return fn(*args, **kwargs)
        return wrapper

@no_grad()
def evaluate(model, batch):
    print(f"  grad enabled during eval: {no_grad._enabled}")  # False
    return 0.95

evaluate(None, None)
print(f"  grad enabled after eval: {no_grad._enabled}")       # True

# 2. cached_property — computes once, stores on instance (not on class)
from functools import cached_property

class TransformerModel:
    def __init__(self, num_layers: int, hidden_dim: int):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self._weights   = [[0.1] * hidden_dim for _ in range(num_layers)]

    @cached_property
    def num_parameters(self) -> int:
        print("  computing num_parameters (expensive, runs once)")
        return sum(len(w) for w in self._weights)

model = TransformerModel(6, 512)
print(f"  first access : {model.num_parameters}")    # computes
print(f"  second access: {model.num_parameters}")    # cached — no recompute
print(f"  stored in __dict__: {'num_parameters' in model.__dict__}")

# 3. lru_cache — memoize pure functions (common for tokenization, feature lookup)
from functools import lru_cache

@lru_cache(maxsize=1024)
def encode_token(token: str) -> int:
    print(f"  encoding {token!r} (cache miss)")
    return hash(token) % 50257

encode_token("hello")
encode_token("world")
encode_token("hello")   # cache hit — no print
print(f"  cache info: {encode_token.cache_info()}")
