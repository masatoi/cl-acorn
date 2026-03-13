# cl-acorn Benchmark Suite

Benchmarks comparing cl-acorn's AD engine, distributions, and inference samplers against JAX, PyTorch, and NumPyro.

## Overview

The suite covers:
- **AD engine**: forward-mode (`derivative`), reverse-mode (`gradient`), `hessian-vector-product`
- **Distributions**: log-pdf evaluation for Normal, Gamma, Beta, Poisson
- **Inference**: HMC, NUTS, and variational inference throughput

## Running CL Benchmarks

**From the REPL:**
```lisp
(asdf:load-system :cl-acorn/benchmarks)
(cl-acorn.benchmarks:run-all)
```

**Command line:**
```bash
sbcl --eval "(asdf:load-system :cl-acorn/benchmarks)" \
     --eval "(cl-acorn.benchmarks:run-all)" \
     --quit
```

## Running Python Benchmarks

**Install dependencies:**
```bash
pip install -r benchmarks/python/requirements.txt
```

**Run each benchmark:**
```bash
python benchmarks/python/bench_ad_jax.py
python benchmarks/python/bench_ad_torch.py
python benchmarks/python/bench_inference_numpyro.py
```

## Results

> Fill in after running on your machine. Results below are illustrative only.
>
> Test machine: fill in CPU, RAM, SBCL version, Python version, JAX version, PyTorch version, NumPyro version.

### AD Engine

| Task | cl-acorn (μs) | JAX (μs) | PyTorch (μs) |
|------|--------------|----------|--------------|
| derivative-1d | — | — | — |
| gradient-1d | — | — | — |
| gradient-10d | — | — | — |
| gradient-100d | — | — | — |
| gradient-1000d | — | — | — |
| hessian-vector-product-10d | — | — | — |

### Distributions

| Task | cl-acorn (μs) | JAX (μs) |
|------|--------------|----------|
| normal-log-pdf | — | — |
| gamma-log-pdf | — | — |
| beta-log-pdf | — | — |
| poisson-log-pdf | — | — |

### Inference

| Task | cl-acorn (samples/sec) | NumPyro (samples/sec) |
|------|----------------------|----------------------|
| hmc-2d-standard-normal | — | — |
| nuts-2d-standard-normal | — | — |
| vi-2d-standard-normal | — | — |
