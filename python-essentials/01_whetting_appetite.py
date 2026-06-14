"""
Section 1: Whetting Your Appetite
==================================

── HIGH-LEVEL LANGUAGE ──────────────────────────────────────────────────────
Abstracts away memory management, pointer arithmetic, hardware details.
You write "what", not "how" (mostly). The price: less control, more overhead.
Python is about as high-level as you get while still being general-purpose.

── INTERPRETED vs COMPILED ──────────────────────────────────────────────────
The binary isn't the full story — it's a spectrum:

  Fully compiled (C/C++/Rust):
    Source → machine code (via compiler). Runs directly on CPU.
    Fast. No runtime needed. Platform-specific binary.

  Bytecode-compiled + interpreted (Python, Java/JVM):
    Source → bytecode (CPython compiles .py → .pyc at import time)
    Bytecode → executed by a virtual machine (the interpreter loop)
    Python's VM is a stack-based bytecode interpreter (ceval.c in CPython).
    "Interpreted" is a shorthand — there IS a compile step, it's just hidden.

  JIT-compiled (PyPy, Java HotSpot, V8):
    Bytecode → machine code at runtime, based on observed hot paths.
    PyPy can be 5-10× faster than CPython for pure Python loops.

  Python's position: bytecode-compiled + interpreted by default.
  BUT your numerical code (NumPy, PyTorch) runs compiled C/CUDA — Python
  is just the glue. The "interpreted" overhead is rarely the bottleneck.

── DYNAMICALLY TYPED ────────────────────────────────────────────────────────
Type is a property of the OBJECT, not the variable name.
  x = 42        # x refers to an int object
  x = "hello"   # x now refers to a str object — perfectly valid

  Contrast with statically typed (C++/Java/Rust): type is checked at compile
  time, bound to the variable declaration. Catches errors earlier, enables
  better compiler optimization.

  Python's type hints (PEP 484, 3.5+) are OPTIONAL and NOT enforced at
  runtime — they're metadata for tools (mypy, pyright, IDEs). The runtime
  is still fully dynamic.

  Duck typing: if it has .fit() and .predict(), it's a model — no interface
  declaration needed. Powerful, but shifts error detection to runtime.

── GARBAGE COLLECTED ────────────────────────────────────────────────────────
CPython uses TWO mechanisms:

  1. Reference counting (primary):
     Every object has a refcount. When it hits 0, memory is freed immediately.
     Predictable, low-latency deallocation.
     Problem: cannot collect reference CYCLES (a → b → a, both stuck at rc=1).

  2. Cyclic garbage collector (secondary, gc module):
     Periodically scans for unreachable cycles and collects them.
     Runs in "generations" (gen0 most frequent, gen2 least).
     Can be tuned or disabled: gc.disable() in tight loops if you know
     there are no cycles.

  Unlike JVM/Go GC, CPython's refcounting means no "stop-the-world" pauses
  for most allocations. The cyclic GC does pause briefly, but rarely matters.

── THE GIL (Global Interpreter Lock) ────────────────────────────────────────
The GIL is a mutex in CPython that ensures only ONE thread executes Python
bytecode at a time — even on multi-core hardware.

  Why it exists:
    CPython's object model (refcounts, memory allocator) is not thread-safe.
    The GIL was the pragmatic fix that made the C extension API simple.

  What it means in practice:
    threading.Thread → concurrent but NOT parallel for CPU-bound Python code.
    The GIL is released during I/O, so threads ARE useful for I/O-bound work
    (network calls, disk reads, waiting on subprocesses).
    C extensions (NumPy, PyTorch ops) release the GIL → they run truly parallel
    alongside Python threads. This is why training loops aren't GIL-bound.

  Status in Python 3.13+:
    PEP 703 introduced a "free-threaded" build (--disable-gil / nogil).
    Python 3.13: experimental, opt-in at build time.
    Python 3.14: more stable, still not the default CPython build.
    The ecosystem (C extensions) needs to catch up before it's mainstream.

── CAN PYTHON LAUNCH TRULY PARALLEL PROGRAMS? ───────────────────────────────
Yes — four distinct strategies:

  1. multiprocessing (cpu-bound Python code):
     Spawns separate OS processes, each with its own GIL + heap.
     True parallelism. IPC via Queue/Pipe/shared memory.
     Overhead: process startup, pickling objects across processes.
     Use: data preprocessing, CPU-bound inference, embarrassingly parallel jobs.

  2. concurrent.futures (high-level API over both):
     ProcessPoolExecutor → multiprocessing under the hood
     ThreadPoolExecutor  → threading under the hood
     Uniform interface, easy map/submit pattern.

  3. asyncio (I/O-bound concurrency, NOT parallelism):
     Single-threaded cooperative multitasking via an event loop.
     No GIL issue because only one coroutine runs at a time.
     Ideal for: thousands of simultaneous network calls, async APIs (FastAPI),
     LLM inference servers, streaming responses.
     NOT suitable for CPU-bound work.

  4. NumPy / PyTorch / native extensions:
     Release the GIL. CUDA kernels run on GPU threads entirely outside Python.
     A torch.matmul() call is parallel across thousands of CUDA cores
     while Python is "blocked" waiting — effectively free parallelism.

  Summary table:
    Strategy            | GIL impact | Use case
    --------------------|------------|-----------------------------
    threading           | blocked    | I/O-bound, GUI callbacks
    multiprocessing     | bypassed   | CPU-bound Python code
    asyncio             | N/A        | high-concurrency I/O
    C extensions/CUDA   | released   | numerical compute (the common path)
"""

import time
import threading
import multiprocessing
import concurrent.futures
import asyncio

# ── DEMO 1: GIL in action — threads don't help CPU-bound work ────────────────

def cpu_burn(n=5_000_000):
    """Pure Python CPU work — GIL makes threads useless here."""
    total = 0
    for i in range(n):
        total += i
    return total

def benchmark(label, fn):
    t = time.perf_counter()
    fn()
    print(f"  {label:<40} {(time.perf_counter()-t)*1000:.1f}ms")

def run_sequential():
    cpu_burn(); cpu_burn()

def run_threaded():
    t1 = threading.Thread(target=cpu_burn)
    t2 = threading.Thread(target=cpu_burn)
    t1.start(); t2.start()
    t1.join();  t2.join()

def run_multiprocess():
    with multiprocessing.Pool(2) as pool:
        pool.map(cpu_burn, [5_000_000, 5_000_000])

if __name__ == "__main__":
    print("=== CPU-BOUND PARALLELISM ===")
    benchmark("sequential (baseline)",          run_sequential)
    benchmark("threading (GIL → no speedup)",   run_threaded)
    benchmark("multiprocessing (true parallel)", run_multiprocess)

    # ── DEMO 2: concurrent.futures — clean unified API ───────────────────────

    def slow_io(task_id):
        time.sleep(0.1)
        return f"result_{task_id}"

    print("\n=== I/O-BOUND: ThreadPoolExecutor ===")
    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(slow_io, range(10)))   # 10 × 100ms "requests"
    elapsed = (time.perf_counter() - t) * 1000
    print(f"  10 × 100ms tasks, 10 threads: {elapsed:.0f}ms  (expected ~100ms)")

    # ── DEMO 3: asyncio — single-threaded high concurrency ───────────────────

    async def fetch(session_id: int) -> str:
        await asyncio.sleep(0.1)
        return f"response_{session_id}"

    async def main():
        return await asyncio.gather(*[fetch(i) for i in range(10)])

    print("\n=== I/O-BOUND: asyncio ===")
    t = time.perf_counter()
    results = asyncio.run(main())
    elapsed = (time.perf_counter() - t) * 1000
    print(f"  10 × 100ms coroutines, asyncio: {elapsed:.0f}ms  (expected ~100ms)")

    # ── DEMO 4: NumPy releases the GIL ───────────────────────────────────────
    try:
        import numpy as np
        print("\n=== NUMPY RELEASES THE GIL ===")
        arr = np.random.randn(500, 500)

        def np_work():
            for _ in range(20):
                np.dot(arr, arr)

        t = time.perf_counter()
        threads = [threading.Thread(target=np_work) for _ in range(4)]
        for th in threads: th.start()
        for th in threads: th.join()
        print(f"  4 threads × 20 matmuls: {(time.perf_counter()-t)*1000:.0f}ms")
        print("  (threads ran in parallel — NumPy released the GIL)")
    except ImportError:
        print("\n  (numpy not installed — skipping)")
