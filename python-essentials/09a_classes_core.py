"""
Section 9a: Classes — Core OOP Mechanics
==========================================
FOCUS: Python's object model internals — what actually happens when you define
a class, instantiate it, and call methods. Essential for understanding PyTorch
Module, dataclasses, descriptors, and metaclasses.

PYTHON'S OBJECT MODEL
----------------------
Everything is an object. Classes are objects too (instances of `type`).
  type(int)       → <class 'type'>
  type(type)      → <class 'type'>   (type is its own metaclass)

Class definition creates a namespace (dict), then calls type.__new__ to
build the class object. The class object is itself stored as a name binding.

METHOD RESOLUTION ORDER (MRO)
------------------------------
Python uses C3 linearization for multiple inheritance. Determines which class's
method is called. Accessible via ClassName.__mro__ or mro().
  super() uses MRO to find the next class in the chain — critical for
  cooperative multiple inheritance. Always call super().__init__() in __init__.

ATTRIBUTE LOOKUP ORDER
-----------------------
For obj.attr:
  1. Data descriptors in type(obj).__mro__ (have __set__ or __delete__)
  2. Instance __dict__
  3. Non-data descriptors and other class attrs in type(obj).__mro__
  (Functions are non-data descriptors — that's how methods work)

DUNDER METHODS (key ones)
--------------------------
  __init__      : initializer (object already allocated by __new__)
  __new__       : allocator — override for singleton, immutable subclasses
  __repr__      : unambiguous string (for devs), used in REPL and logging
  __str__       : readable string (for users), fallback to __repr__
  __eq__ / __hash__: equality and hashing — if you define __eq__, Python
                     sets __hash__=None (unhashable) unless you also define it
  __len__, __getitem__, __setitem__, __contains__ : sequence/mapping protocol
  __enter__, __exit__ : context manager protocol
  __call__      : makes instances callable — used in PyTorch layers
  __slots__     : replaces __dict__ with fixed array — saves memory, faster attr access
"""

# ── BASIC CLASS + MRO ─────────────────────────────────────────────────────────
print("=== MRO and cooperative inheritance ===")

class Module:
    """Minimal PyTorch-like Module base."""
    def __init__(self, name: str = ""):
        self.name = name
        self._parameters: dict = {}
        self._modules: dict    = {}
        self.training: bool    = True

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)  # __call__ wraps forward

    def train(self, mode: bool = True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

class Regularizable:
    """Mixin: adds L2 regularization loss."""
    def __init__(self, weight_decay: float = 1e-4, **kwargs):
        super().__init__(**kwargs)   # cooperative — passes remaining kwargs up
        self.weight_decay = weight_decay

    def reg_loss(self) -> float:
        total = 0.0
        for p in self._parameters.values():
            total += sum(v**2 for v in p)
        return self.weight_decay * total

class Loggable:
    """Mixin: structured logging."""
    def __init__(self, log_prefix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.log_prefix = log_prefix

    def log(self, msg: str):
        prefix = f"[{self.log_prefix}] " if self.log_prefix else ""
        print(f"  {prefix}{msg}")

class LinearLayer(Regularizable, Loggable, Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__(**kwargs)   # cooperative chain: Regularizable → Loggable → Module
        import random
        self._parameters['weight'] = [random.gauss(0, 0.1) for _ in range(in_dim * out_dim)]
        self._parameters['bias']   = [0.0] * out_dim
        self.in_dim  = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        self.log(f"forward pass: input shape ({len(x)},)")
        return x   # stub

# MRO determines method resolution order
print(f"MRO: {[c.__name__ for c in LinearLayer.__mro__]}")

layer = LinearLayer(128, 64, name="fc1", weight_decay=1e-3, log_prefix="FC1")
print(layer)
layer(list(range(128)))
print(f"reg loss: {layer.reg_loss():.6f}")

# ── DESCRIPTORS ───────────────────────────────────────────────────────────────
print("\n=== Descriptors ===")
# Descriptors define __get__, __set__, __delete__ on a class.
# Python properties, classmethods, staticmethods are all descriptors.
# They enable attribute access interception WITHOUT subclassing.

class Validated:
    """Non-negative float descriptor — reusable validation."""
    def __set_name__(self, owner, name):
        self._name = name           # called at class creation time

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self             # accessed on class, return descriptor itself
        return obj.__dict__.get(self._name, 0.0)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self._name} must be numeric, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{self._name} must be non-negative, got {value}")
        obj.__dict__[self._name] = float(value)

class TrainingConfig:
    lr           = Validated()
    weight_decay = Validated()
    dropout      = Validated()

    def __init__(self, lr, weight_decay=1e-4, dropout=0.1):
        self.lr           = lr          # triggers Validated.__set__
        self.weight_decay = weight_decay
        self.dropout      = dropout

cfg = TrainingConfig(lr=3e-4)
print(f"config: lr={cfg.lr}, wd={cfg.weight_decay}, drop={cfg.dropout}")

try:
    cfg.lr = -1.0
except ValueError as e:
    print(f"  descriptor caught: {e}")

# property is syntactic sugar for a descriptor
class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32

t = Temperature(100)
print(f"100°C = {t.fahrenheit}°F")

# ── __SLOTS__ ─────────────────────────────────────────────────────────────────
print("\n=== __slots__ ===")
# Replaces per-instance __dict__ with a fixed C array of slots.
# Memory saving: ~200 bytes per instance (no dict overhead).
# Critical for high-frequency objects: tokens, graph nodes, embeddings.

import sys

class TokenWithDict:
    def __init__(self, id: int, pos: int):
        self.id  = id
        self.pos = pos

class TokenWithSlots:
    __slots__ = ('id', 'pos')
    def __init__(self, id: int, pos: int):
        self.id  = id
        self.pos = pos

t_dict  = TokenWithDict(42, 0)
t_slots = TokenWithSlots(42, 0)
print(f"with __dict__ : {sys.getsizeof(t_dict)} bytes + {sys.getsizeof(t_dict.__dict__)} dict")
print(f"with __slots__: {sys.getsizeof(t_slots)} bytes, no __dict__")
# no __dict__ means no dynamic attribute assignment:
try:
    t_slots.new_attr = "oops"
except AttributeError as e:
    print(f"  slots blocked: {e}")

# ── __REPR__ / __EQ__ / __HASH__ TRIANGLE ────────────────────────────────────
print("\n=== __repr__ / __eq__ / __hash__ ===")

class ModelCheckpoint:
    def __init__(self, path: str, epoch: int, val_loss: float):
        self.path     = path
        self.epoch    = epoch
        self.val_loss = val_loss

    def __repr__(self):
        return f"Checkpoint(epoch={self.epoch}, loss={self.val_loss:.4f}, path={self.path!r})"

    def __eq__(self, other):
        if not isinstance(other, ModelCheckpoint):
            return NotImplemented      # NOT False — lets Python try reflected op
        return self.path == other.path

    def __hash__(self):
        return hash(self.path)         # define hash when defining __eq__

    def __lt__(self, other):           # enables sorted(), min(), max()
        return self.val_loss < other.val_loss

ckpts = [
    ModelCheckpoint("ckpt_3.pt", 3, 0.54),
    ModelCheckpoint("ckpt_1.pt", 1, 0.92),
    ModelCheckpoint("ckpt_2.pt", 2, 0.71),
]
best = min(ckpts)
print(f"best: {best}")
print(f"sorted: {sorted(ckpts)}")
print(f"in set: {len({ckpts[0], ckpts[0]}) == 1}")  # deduplication works

# ── CLASSMETHOD / STATICMETHOD ────────────────────────────────────────────────
print("\n=== classmethod / staticmethod ===")

class Tokenizer:
    _instances: dict = {}

    def __init__(self, vocab_size: int, name: str):
        self.vocab_size = vocab_size
        self.name       = name

    @classmethod
    def from_pretrained(cls, name: str) -> 'Tokenizer':
        """Factory method — alternative constructor. cls = subclass if subclassed."""
        if name not in cls._instances:
            # In reality: load from disk/hub
            cls._instances[name] = cls(vocab_size=50257, name=name)
        return cls._instances[name]

    @staticmethod
    def is_valid_vocab_size(n: int) -> bool:
        """Utility that doesn't need self or cls — no hidden first arg."""
        return n > 0 and (n & (n - 1)) == 0   # power of 2 check

tok1 = Tokenizer.from_pretrained("gpt2")
tok2 = Tokenizer.from_pretrained("gpt2")
print(f"singleton via classmethod: {tok1 is tok2}")
print(f"valid vocab size 512: {Tokenizer.is_valid_vocab_size(512)}")
print(f"valid vocab size 500: {Tokenizer.is_valid_vocab_size(500)}")
