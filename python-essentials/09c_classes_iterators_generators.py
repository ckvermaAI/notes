"""
Section 9c: Iterators, Generators, and the Iteration Protocol
==============================================================
FOCUS: how for-loops actually work, generator functions vs expressions,
send/throw/close, yield from, async generators — all critical for building
data pipelines, streaming inference, and coroutine-based systems.

ITERATION PROTOCOL
-------------------
for x in obj:  desugars to:
  iter_obj = iter(obj)         # calls obj.__iter__()
  while True:
    try:
      x = next(iter_obj)       # calls iter_obj.__next__()
    except StopIteration:
      break

Any object implementing __iter__ + __next__ is an iterator.
Any object implementing only __iter__ (returning a fresh iterator) is an iterable.
Iterators are their own iterables (return self from __iter__).

GENERATORS
-----------
A generator function contains yield. Calling it returns a generator object
(which is an iterator). Execution is SUSPENDED at each yield and resumed on
the next next() call. State (locals, instruction pointer) is preserved.

Generator lifecycle:
  created → suspended-at-start → running ↔ suspended-at-yield → closed

send(value): resumes generator, value becomes the result of the yield expression
throw(exc) : injects an exception at the yield point
close()    : injects GeneratorExit — triggers finally blocks

yield from: delegates to a sub-generator, transparently forwarding
            send/throw/return. Critical for composing generators and
            implementing coroutines (asyncio's original mechanism).

ASYNC GENERATORS (3.6+)
------------------------
async def + yield = async generator. Consumed with async for.
Used for: streaming API responses, async dataset loading, async inference.
"""

import time
import itertools
from typing import Iterator, Generator, AsyncIterator

# ── CUSTOM ITERATOR ───────────────────────────────────────────────────────────
print("=== Custom Iterator ===")

class DatasetIterator:
    """Iterator over fixed-size batches — stateful, not re-entrant."""
    def __init__(self, data: list, batch_size: int):
        self.data       = data
        self.batch_size = batch_size
        self._idx       = 0

    def __iter__(self):
        return self            # iterator returns self

    def __next__(self):
        if self._idx >= len(self.data):
            raise StopIteration
        batch = self.data[self._idx : self._idx + self.batch_size]
        self._idx += self.batch_size
        return batch

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

import math

dataset = list(range(10))
loader  = DatasetIterator(dataset, batch_size=3)
for batch in loader:
    print(f"  batch: {batch}")

# iterables vs iterators — key distinction
print(f"\n  list is iterable, not iterator: {hasattr([], '__next__')}")
print(f"  iter(list) is iterator:         {hasattr(iter([]), '__next__')}")

# ── GENERATOR FUNCTIONS ───────────────────────────────────────────────────────
print("\n=== Generator Functions ===")

def infinite_counter(start: int = 0, step: int = 1):
    """Infinite sequence — only computes values on demand."""
    n = start
    while True:
        yield n
        n += step

# Use itertools.islice to consume lazily
first_10 = list(itertools.islice(infinite_counter(0, 2), 10))
print(f"first 10 even numbers: {first_10}")

def tokenize_stream(text: str, chunk_size: int = 4) -> Generator[str, None, None]:
    """Generator for streaming tokenization — no full materialization."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield words[i : i + chunk_size]

corpus = "the quick brown fox jumps over the lazy dog and then some more words here"
for chunk in tokenize_stream(corpus, chunk_size=3):
    print(f"  chunk: {chunk}")

# Generator return value — accessible via StopIteration.value
def count_tokens(texts: list[str]) -> Generator[int, None, int]:
    total = 0
    for text in texts:
        n = len(text.split())
        yield n          # yield per-doc count
        total += n
    return total         # return value via StopIteration.value

gen = count_tokens(["hello world", "foo bar baz", "one"])
try:
    while True:
        count = next(gen)
        print(f"  doc tokens: {count}")
except StopIteration as e:
    print(f"  total tokens: {e.value}")   # return value here

# ── SEND / THROW / CLOSE ──────────────────────────────────────────────────────
print("\n=== send / throw ===")

def running_average() -> Generator[float, float, None]:
    """Two-way generator: receive values via send(), yield running average."""
    total = 0.0
    count = 0
    value = yield 0.0                 # first next() primes the generator
    while True:
        total += value
        count += 1
        value = yield total / count   # yield avg, receive next value

avg_gen = running_average()
next(avg_gen)   # prime (advance to first yield)

losses = [2.3, 1.8, 1.4, 0.9, 0.7]
for loss in losses:
    avg = avg_gen.send(loss)
    print(f"  loss={loss:.1f}, running_avg={avg:.3f}")

avg_gen.close()  # triggers GeneratorExit inside generator

# ── YIELD FROM ────────────────────────────────────────────────────────────────
print("\n=== yield from ===")

def epoch_generator(data: list, epochs: int, batch_size: int):
    """Compose generators with yield from — transparent delegation."""
    for epoch in range(epochs):
        print(f"  epoch {epoch+1}/{epochs}")
        yield from DatasetIterator(data, batch_size)  # delegates entire iterator

total_batches = 0
for batch in epoch_generator(list(range(6)), epochs=2, batch_size=2):
    total_batches += 1
print(f"  total batches across 2 epochs: {total_batches}")

# ── GENERATOR PIPELINE ────────────────────────────────────────────────────────
print("\n=== Generator Pipeline (lazy data pipeline) ===")

# Each stage is a generator — nothing materializes until consumed
# Classic pattern for large-scale data processing

def read_records(n: int):
    """Source: simulate reading from disk/DB lazily."""
    for i in range(n):
        yield {"id": i, "text": f"sample text number {i}", "label": i % 3}

def filter_records(records, min_len: int = 10):
    """Filter stage — lazy."""
    for r in records:
        if len(r["text"]) >= min_len:
            yield r

def tokenize_records(records, max_len: int = 8):
    """Transform stage — lazy."""
    for r in records:
        tokens = r["text"].split()[:max_len]
        yield {**r, "tokens": tokens, "n_tokens": len(tokens)}

def batch_records(records, batch_size: int = 4):
    """Batch stage — accumulates then yields."""
    batch = []
    for r in records:
        batch.append(r)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch   # final partial batch

# Compose pipeline — nothing runs until we consume
pipeline = batch_records(
    tokenize_records(
        filter_records(
            read_records(20),
            min_len=5
        )
    ),
    batch_size=4
)

for i, batch in enumerate(pipeline):
    print(f"  batch {i}: {len(batch)} records, ids={[r['id'] for r in batch]}")

# ── ASYNC GENERATORS ──────────────────────────────────────────────────────────
print("\n=== Async Generators ===")
import asyncio

async def stream_llm_response(prompt: str, tokens: list[str]) -> AsyncIterator[str]:
    """Simulate streaming token generation — like OpenAI streaming API."""
    for token in tokens:
        await asyncio.sleep(0.01)   # simulate network latency per token
        yield token

async def stream_pipeline():
    response_tokens = ["The", " answer", " is", " 42", ".", " This", " is", " correct", "."]
    full_response = []
    async for token in stream_llm_response("What is the answer?", response_tokens):
        full_response.append(token)
        print(f"  token: {token!r:15} (accumulated: {''.join(full_response)!r})")
    return "".join(full_response)

final = asyncio.run(stream_pipeline())
print(f"\n  final: {final!r}")

# Async generator with cleanup
async def managed_stream(items: list):
    """Async generator that guarantees cleanup via try/finally."""
    print("  stream: opening connection")
    try:
        for item in items:
            await asyncio.sleep(0)   # yield to event loop
            yield item
    finally:
        print("  stream: connection closed")  # runs on close() or exhaustion

async def consume_managed():
    async for item in managed_stream([1, 2, 3]):
        print(f"  consumed: {item}")

asyncio.run(consume_managed())
