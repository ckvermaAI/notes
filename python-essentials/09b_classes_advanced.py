"""
Section 9b: Classes — Advanced Patterns
=========================================
FOCUS: dataclasses, metaclasses, abstract base classes, protocols (structural
typing), __init_subclass__ — patterns used extensively in ML frameworks.

DATACLASSES (3.7+)
-------------------
@dataclass auto-generates __init__, __repr__, __eq__ from field annotations.
  field()          : per-field config (default_factory, compare, repr, init, etc.)
  @dataclass(frozen=True): immutable, auto-generates __hash__
  @dataclass(slots=True) : auto __slots__ (3.10+)
  @dataclass(kw_only=True): all fields keyword-only
  __post_init__    : runs after generated __init__ — for validation/derived fields

ABSTRACT BASE CLASSES
----------------------
ABCs enforce interface contracts at instantiation time (not at call time).
  @abstractmethod  : subclass must implement or ABC cannot be instantiated
  ABCs + __subclasshook__: customize isinstance/issubclass checks without inheritance

PROTOCOLS (3.8+, PEP 544)
--------------------------
Structural subtyping — "if it has these methods, it qualifies."
Duck typing + static type checker support. No registration or inheritance needed.
Protocol is the modern alternative to ABCs for type-checking purposes.

METACLASSES
-----------
A metaclass controls how a CLASS is created (type is the default metaclass).
type(name, bases, namespace) creates a class programmatically.
Use cases: auto-registration, enforcing APIs, ORM field collection (Django models),
           auto-wrapping methods.
In modern Python, prefer __init_subclass__ over metaclasses for most use cases.
"""

from dataclasses import dataclass, field, fields, asdict
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import math

# ── DATACLASSES ───────────────────────────────────────────────────────────────
print("=== @dataclass ===")

@dataclass
class LayerConfig:
    hidden_dim: int
    num_heads:  int
    dropout:    float  = 0.1
    bias:       bool   = True
    # mutable default must use field(default_factory=...)
    activations: list  = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        self.head_dim = self.hidden_dim // self.num_heads  # derived field

cfg = LayerConfig(hidden_dim=512, num_heads=8)
print(cfg)
print(f"head_dim (derived): {cfg.head_dim}")

try:
    LayerConfig(hidden_dim=512, num_heads=7)
except ValueError as e:
    print(f"  validation: {e}")

# asdict — deep conversion to plain dict (great for JSON serialization)
print(f"as dict: {asdict(cfg)}")

# fields() — introspect at runtime
for f in fields(cfg):
    print(f"  field: {f.name:15} type={f.type}")

# Frozen dataclass — immutable, hashable, usable as dict key / set element
@dataclass(frozen=True)
class ModelId:
    provider: str
    name:     str
    version:  str = "latest"

    def __str__(self):
        return f"{self.provider}/{self.name}:{self.version}"

m = ModelId("anthropic", "claude-sonnet-4-6")
print(f"\nfrozen: {m}")
print(f"hashable (set): {len({m, m, ModelId('openai','gpt-4')}) == 2}")
try:
    m.name = "opus"   # type: ignore
except Exception as e:
    print(f"  immutable: {type(e).__name__}: {e}")

# ── ABSTRACT BASE CLASSES ─────────────────────────────────────────────────────
print("\n=== ABC ===")

class BaseModel(ABC):
    """Interface every model in the system must satisfy."""

    @abstractmethod
    def forward(self, x: list) -> list:
        ...

    @abstractmethod
    def parameters(self) -> list:
        ...

    def num_parameters(self) -> int:
        return sum(len(p) if hasattr(p, '__len__') else 1
                   for p in self.parameters())

    # Non-abstract method with default — subclasses inherit, may override
    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump({"class": self.__class__.__name__}, f)
        print(f"  saved {self.__class__.__name__} to {path}")

class DummyModel(BaseModel):
    def __init__(self, dim: int):
        self.weights = [0.1] * dim

    def forward(self, x: list) -> list:
        return [sum(a * b for a, b in zip(x, self.weights))]

    def parameters(self) -> list:
        return [self.weights]

try:
    BaseModel()          # can't instantiate ABC directly
except TypeError as e:
    print(f"  ABC blocks instantiation: {e}")

m = DummyModel(64)
print(f"  num_parameters: {m.num_parameters()}")
print(f"  forward: {m.forward([1.0]*64)}")

# ── PROTOCOLS — structural typing ─────────────────────────────────────────────
print("\n=== Protocol ===")

@runtime_checkable   # enables isinstance checks at runtime
class Embedder(Protocol):
    """Anything with encode() qualifies — no inheritance needed."""
    def encode(self, text: str) -> list[float]:
        ...

    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        ...

class SentenceTransformer:
    """Doesn't inherit Embedder — but satisfies it structurally."""
    def encode(self, text: str) -> list[float]:
        return [hash(c) % 100 / 100 for c in text[:8]]

    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]

class OpenAIEmbedder:
    """Another implementation — also satisfies Protocol."""
    def encode(self, text: str) -> list[float]:
        return [0.1] * 1536

    def batch_encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]

def embed_documents(embedder: Embedder, docs: list[str]) -> list[list[float]]:
    """Accepts any Embedder — SentenceTransformer, OpenAI, custom, etc."""
    return embedder.batch_encode(docs)

st  = SentenceTransformer()
oai = OpenAIEmbedder()
print(f"isinstance check (runtime_checkable): {isinstance(st, Embedder)}")
print(f"ST embeddings shape: {len(embed_documents(st, ['hello', 'world']))} × {len(st.encode('hello'))}")
print(f"OAI embeddings shape: {len(embed_documents(oai, ['hello', 'world']))} × {len(oai.encode('hello'))}")

# ── __INIT_SUBCLASS__ — lightweight hook without metaclass ────────────────────
print("\n=== __init_subclass__ ===")

class RegisteredLayer:
    """Auto-registers all subclasses by name — no decorator needed."""
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, layer_type: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        name = layer_type or cls.__name__.lower()
        RegisteredLayer._registry[name] = cls
        print(f"  registered layer: {name!r} → {cls.__name__}")

class AttentionLayer(RegisteredLayer, layer_type="attention"):
    def forward(self, x): return x

class FFNLayer(RegisteredLayer, layer_type="ffn"):
    def forward(self, x): return x

class DropoutLayer(RegisteredLayer):   # uses class name
    def forward(self, x): return x

print(f"registry: {list(RegisteredLayer._registry)}")
print(f"build: {RegisteredLayer._registry['attention']}")

# ── METACLASS — when you really need it ───────────────────────────────────────
print("\n=== Metaclass ===")

class SingletonMeta(type):
    """Metaclass that enforces singleton per class."""
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ModelRegistry(metaclass=SingletonMeta):
    def __init__(self):
        self.models: dict = {}

    def register(self, name: str, model):
        self.models[name] = model

r1 = ModelRegistry()
r2 = ModelRegistry()
print(f"singleton via metaclass: {r1 is r2}")
r1.register("classifier", DummyModel(32))
print(f"r2 sees r1's models: {'classifier' in r2.models}")

# type() — dynamic class creation (what class statement desugars to)
DynamicConfig = type(
    'DynamicConfig',                     # class name
    (object,),                           # bases
    {'lr': 3e-4, 'epochs': 10,           # namespace
     '__repr__': lambda self: f"DynamicConfig(lr={self.lr}, epochs={self.epochs})"}
)
dc = DynamicConfig()
print(f"dynamically created class: {dc}")
