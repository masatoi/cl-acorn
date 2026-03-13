# Benchmarks Design

**Date**: 2026-03-14
**Topic**: cl-acorn performance benchmarking with JAX/PyTorch comparison
**Status**: Approved

## Goal

Implement a full-stack benchmark suite covering the AD engine, distribution functions,
and inference algorithms. Results are compared against equivalent Python code using
JAX and PyTorch (CPU, no GPU).

## Directory Structure

```
benchmarks/
  cl/
    package.lisp                ← cl-acorn.benchmarks package
    bench-utils.lisp            ← defbench macro (timing, statistics)
    bench-ad.lisp               ← AD engine benchmarks
    bench-distributions.lisp    ← Distribution log-pdf benchmarks
    bench-inference.lisp        ← HMC / NUTS / VI benchmarks
    run-all.lisp                ← Run all benchmarks, print table
  python/
    bench_ad_jax.py             ← JAX jax.grad benchmarks
    bench_ad_torch.py           ← PyTorch autograd benchmarks
    bench_inference_numpyro.py  ← NumPyro HMC/NUTS/VI benchmarks
    requirements.txt
  README.md                     ← Results comparison table (manual update)
```

`cl-acorn.asd` gains a `cl-acorn/benchmarks` subsystem.

## ASDF Subsystem

```lisp
(defsystem "cl-acorn/benchmarks"
  :depends-on ("cl-acorn")
  :components ((:module "benchmarks/cl"
                :serial t
                :components
                ((:file "package")
                 (:file "bench-utils")
                 (:file "bench-ad")
                 (:file "bench-distributions")
                 (:file "bench-inference")
                 (:file "run-all")))))
```

## `defbench` Macro

Measures wall-clock time using `get-internal-real-time` and allocation using
`sb-ext:get-bytes-consed`. Runs `n-warmup` iterations (default 10) discarded,
then `n-runs` iterations measured.

```lisp
(defmacro defbench (name (&key (n-runs 100) (n-warmup 10)) &body body)
  ...)
```

Returns a `bench-result` struct:
```lisp
(defstruct bench-result
  name        ; string
  mean-us     ; double-float microseconds
  min-us      ; double-float
  max-us      ; double-float
  gc-bytes)   ; integer bytes consed
```

`run-all` collects results and prints a formatted table.

## Benchmark Tasks

### AD Engine (`bench-ad.lisp`)

Target function: `f(x) = sum(x_i^2)`  (analytic gradient: `2*x`)

| Task | N | n-runs |
|------|---|--------|
| `derivative-1d` | 1 | 10000 |
| `gradient-1d` | 1 | 10000 |
| `gradient-10d` | 10 | 1000 |
| `gradient-100d` | 100 | 500 |
| `gradient-1000d` | 1000 | 100 |
| `hessian-vector-product-10d` | 10 | 500 |

Python equivalents:
- JAX: `jax.grad(f)(x)`, `jax.jvp(jax.grad(f), (x,), (v,))`
- PyTorch: `y.backward()`, `torch.autograd.functional.jvp`

### Distributions (`bench-distributions.lisp`)

Single scalar call repeated `n-runs` times (no vectorisation — tests the CL
function call overhead honestly).

| Task | n-runs |
|------|--------|
| `normal-log-pdf` | 10000 |
| `gamma-log-pdf` | 10000 |
| `beta-log-pdf` | 10000 |
| `poisson-log-pdf` | 10000 |

Python: `jax.scipy.stats.norm.logpdf` (scalar), `jax.scipy.stats.gamma.logpdf`, etc.

### Inference (`bench-inference.lisp`)

Model: 2D standard normal, `log p(x) = -0.5 * (x1^2 + x2^2)`

| Task | Config | Metric |
|------|--------|--------|
| `hmc-2d` | 500 samples, 200 warmup, adapt-step-size t | samples/sec |
| `nuts-2d` | 500 samples, 200 warmup, adapt-step-size t | samples/sec |
| `vi-2d` | 1000 iterations, n-elbo-samples 10 | iterations/sec |

Python: NumPyro `NUTS`, `HMC`, `SVI` with same model and sample counts.

## Output Format

```
cl-acorn benchmark suite
========================
SBCL 2.x.x  |  2026-03-14

[AD Engine]
Task                         | Mean (μs) | Min (μs) | Max (μs) | GC (bytes)
-----------------------------|-----------|----------|----------|------------
derivative-1d                |       0.8 |      0.7 |      1.2 |          0
gradient-1d                  |       1.5 |      1.3 |      2.1 |        256
gradient-10d                 |      12.4 |     11.8 |     15.0 |       2048
gradient-100d                |     145.2 |    138.0 |    162.0 |      20480
gradient-1000d               |    1523.0 |   1490.0 |   1601.0 |     204800
hessian-vector-product-10d   |      48.3 |     45.1 |     55.2 |       8192

[Distributions]
Task                         | Mean (μs) | Min (μs) | Max (μs) | GC (bytes)
-----------------------------|-----------|----------|----------|------------
normal-log-pdf               |       0.4 |      0.3 |      0.6 |          0
...

[Inference]
Task                         | samples/sec |
-----------------------------|-------------|
hmc-2d-standard-normal       |        42.3 |
nuts-2d-standard-normal      |        38.7 |
vi-2d-standard-normal        |      1250.0 |
```

## Execution

```lisp
;; From REPL
(asdf:load-system :cl-acorn/benchmarks)
(cl-acorn.benchmarks:run-all)
```

```bash
# Command line
sbcl --eval "(asdf:load-system :cl-acorn/benchmarks)" \
     --eval "(cl-acorn.benchmarks:run-all)" \
     --quit

# Python side
pip install -r benchmarks/python/requirements.txt
python benchmarks/python/bench_ad_jax.py
python benchmarks/python/bench_ad_torch.py
python benchmarks/python/bench_inference_numpyro.py
```

## Success Criteria

- `(cl-acorn.benchmarks:run-all)` completes without error
- All AD benchmark tasks produce plausible μs timings
- Python scripts run independently and print comparable numbers
- `benchmarks/README.md` documents how to reproduce results
