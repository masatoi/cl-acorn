# Benchmarks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a full-stack benchmark suite for cl-acorn covering AD, distributions, and inference, with parallel Python (JAX/PyTorch) scripts for comparison.

**Architecture:** A new `cl-acorn/benchmarks` ASDF subsystem lives in `benchmarks/cl/`. A `defbench` macro times N-run batches across multiple trials to compute mean/min/max μs and bytes-consed. Python scripts in `benchmarks/python/` implement identical tasks using JAX and PyTorch.

**Tech Stack:** SBCL (`get-internal-real-time`, `sb-ext:get-bytes-consed`), Rove (not used — this is a standalone runner), JAX, PyTorch, NumPyro.

---

## Key Context

- Package nicknames: `ad:`, `dist:`, `infer:` — use `Write`/`Edit` tools, never `lisp-edit-form`, for files in `benchmarks/cl/`
- ASDF system: `cl-acorn.asd` uses `lisp-edit-form` (it has no nicknames)
- `infer:hmc` and `infer:nuts` return `(values samples accept-rate diagnostics)` — use `diagnostics-elapsed-seconds` for inference timing
- `infer:vi` returns `(values mu-list sigma-list elbo-history diagnostics)`
- `ad:gradient` takes `(fn params-list)` where `fn` must use `ad:` arithmetic
- `dist:normal-sample` takes no required args, `&key mu sigma` (defaults 0.0, 1.0)
- Run tests: `(asdf:load-system :cl-acorn/benchmarks)` then `(cl-acorn.benchmarks:run-all)`

---

## Task 1: ASDF scaffold and package

**Files:**
- Create: `benchmarks/cl/package.lisp`
- Modify: `cl-acorn.asd`

**Step 1: Create `benchmarks/cl/package.lisp`**

```lisp
(defpackage #:cl-acorn.benchmarks
  (:use #:cl)
  (:export #:run-all))

(in-package #:cl-acorn.benchmarks)
```

**Step 2: Add `cl-acorn/benchmarks` subsystem to `cl-acorn.asd`**

After the closing paren of the `cl-acorn/tests` defsystem, add:

```lisp
(defsystem "cl-acorn/benchmarks"
  :author ""
  :license "MIT"
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

Use `lisp-edit-form` on `cl-acorn.asd` with `operation: "insert_after"` after the `cl-acorn/tests` defsystem.

**Step 3: Verify the asd parses**

```bash
sbcl --eval "(asdf:load-asd #P\"/home/wiz/cl-acorn/cl-acorn.asd\")" \
     --eval "(print (asdf:find-system :cl-acorn/benchmarks))" \
     --quit 2>&1 | grep -v "^$"
```

Expected: prints the system object without error.

**Step 4: Commit**

```bash
git add benchmarks/cl/package.lisp cl-acorn.asd
git commit -m "feat(benchmarks): add ASDF subsystem scaffold and package"
```

---

## Task 2: `bench-utils.lisp` — timing infrastructure

**Files:**
- Create: `benchmarks/cl/bench-utils.lisp`

**Step 1: Create the file**

```lisp
(in-package #:cl-acorn.benchmarks)

;;; bench-result struct

(defstruct bench-result
  "Timing result from a single benchmark task."
  (name       "" :type string)
  (mean-us  0.0d0 :type double-float)   ; mean microseconds per call
  (min-us   0.0d0 :type double-float)
  (max-us   0.0d0 :type double-float)
  (gc-bytes   0   :type integer))        ; bytes consed per call (approx)

;;; Low-level timer

(defun time-batch (thunk n-runs)
  "Run THUNK N-RUNS times and return mean microseconds per call."
  (let ((t0 (get-internal-real-time)))
    (dotimes (i n-runs)
      (declare (ignore i))
      (funcall thunk))
    (let ((elapsed (- (get-internal-real-time) t0)))
      (* 1.0d6
         (/ (float elapsed 0.0d0)
            (* (float n-runs 0.0d0)
               (float internal-time-units-per-second 0.0d0)))))))

;;; Main macro

(defmacro defbench (name (&key (n-runs 100) (n-trials 5) (n-warmup 10)) &body body)
  "Benchmark BODY, returning a BENCH-RESULT.
N-WARMUP discarded runs, then N-TRIALS batches of N-RUNS each.
Mean/min/max are computed across the trial batch-means."
  `(let ((thunk (lambda () ,@body)))
     ;; warmup
     (dotimes (i ,n-warmup)
       (declare (ignore i))
       (funcall thunk))
     ;; measure gc per run
     (let ((gc-start (sb-ext:get-bytes-consed)))
       (dotimes (i ,n-runs)
         (declare (ignore i))
         (funcall thunk))
       (let* ((gc-bytes-per-run
               (floor (- (sb-ext:get-bytes-consed) gc-start) ,n-runs))
              (trial-means
               (loop repeat ,n-trials
                     collect (time-batch thunk ,n-runs))))
         (make-bench-result
          :name ,name
          :mean-us (/ (reduce #'+ trial-means) (float ,n-trials 0.0d0))
          :min-us  (reduce #'min trial-means)
          :max-us  (reduce #'max trial-means)
          :gc-bytes gc-bytes-per-run)))))

;;; Table printer

(defun print-bench-table (section results)
  "Print RESULTS as a formatted table under SECTION heading."
  (format t "~%[~A]~%" section)
  (format t "~30A | ~9A | ~9A | ~9A | ~10A~%"
          "Task" "Mean(μs)" "Min(μs)" "Max(μs)" "GC(bytes)")
  (format t "~30A-+-~9A-+-~9A-+-~9A-+-~10A~%"
          (make-string 30 :initial-element #\-)
          (make-string 9 :initial-element #\-)
          (make-string 9 :initial-element #\-)
          (make-string 9 :initial-element #\-)
          (make-string 10 :initial-element #\-))
  (dolist (r results)
    (format t "~30A | ~9,2F | ~9,2F | ~9,2F | ~10D~%"
            (bench-result-name r)
            (bench-result-mean-us r)
            (bench-result-min-us r)
            (bench-result-max-us r)
            (bench-result-gc-bytes r))))

(defun print-inference-table (section results)
  "Print inference RESULTS showing samples/sec."
  (format t "~%[~A]~%" section)
  (format t "~30A | ~12A~%"
          "Task" "samples/sec")
  (format t "~30A-+-~12A~%"
          (make-string 30 :initial-element #\-)
          (make-string 12 :initial-element #\-))
  (dolist (r results)
    ;; mean-us = microseconds per sample, so samples/sec = 1e6 / mean-us
    (let ((sps (if (> (bench-result-mean-us r) 0.0d0)
                   (/ 1.0d6 (bench-result-mean-us r))
                   0.0d0)))
      (format t "~30A | ~12,1F~%"
              (bench-result-name r)
              sps))))
```

**Step 2: Verify it loads**

```bash
sbcl --eval "(asdf:load-system :cl-acorn/benchmarks)" --quit 2>&1 | tail -5
```

Expected: loads without error (bench-ad etc. not yet present, so stop after creating stubs first — see note below).

> **Note:** Since the ASDF system is `:serial t`, you need stub files for `bench-ad`, `bench-distributions`, `bench-inference`, and `run-all` before the system can load. Create them as minimal stubs:

```bash
echo '(in-package #:cl-acorn.benchmarks)' > benchmarks/cl/bench-ad.lisp
echo '(in-package #:cl-acorn.benchmarks)' > benchmarks/cl/bench-distributions.lisp
echo '(in-package #:cl-acorn.benchmarks)' > benchmarks/cl/bench-inference.lisp
echo '(in-package #:cl-acorn.benchmarks)' > benchmarks/cl/run-all.lisp
```

Then verify:
```bash
sbcl --eval "(asdf:load-system :cl-acorn/benchmarks)" --quit 2>&1 | grep -E "error|Error" | head -5
```

Expected: no errors.

**Step 3: Commit**

```bash
git add benchmarks/cl/bench-utils.lisp benchmarks/cl/bench-ad.lisp \
        benchmarks/cl/bench-distributions.lisp benchmarks/cl/bench-inference.lisp \
        benchmarks/cl/run-all.lisp
git commit -m "feat(benchmarks): add bench-utils (defbench macro, result struct, table printer)"
```

---

## Task 3: `bench-ad.lisp` — AD engine benchmarks

**Files:**
- Modify: `benchmarks/cl/bench-ad.lisp` (replace stub)

**Step 1: Write `bench-ad.lisp`**

```lisp
(in-package #:cl-acorn.benchmarks)

;;; Target function: f(x) = sum(x_i^2)
;;; Must use ad: arithmetic to be differentiable by ad:gradient

(defun sum-squares (params)
  "Compute sum of squares: f(x) = sum(x_i^2). Uses ad: arithmetic."
  (reduce #'ad:+ params :key (lambda (x) (ad:* x x))))

;;; Pre-allocated input lists (avoid measuring allocation of inputs)

(defvar *x-1d*    (list 1.0d0))
(defvar *x-10d*   (make-list 10  :initial-element 1.0d0))
(defvar *x-100d*  (make-list 100 :initial-element 1.0d0))
(defvar *x-1000d* (make-list 1000 :initial-element 1.0d0))
(defvar *v-10d*   (make-list 10  :initial-element 1.0d0))   ; vector for HVP

(defun run-ad-benchmarks ()
  "Run all AD engine benchmarks and return a list of BENCH-RESULT."
  (list
   (defbench "derivative-1d" (:n-runs 10000 :n-warmup 100)
     (ad:derivative #'sum-squares 1.0d0))

   (defbench "gradient-1d" (:n-runs 10000 :n-warmup 100)
     (ad:gradient #'sum-squares *x-1d*))

   (defbench "gradient-10d" (:n-runs 1000 :n-warmup 50)
     (ad:gradient #'sum-squares *x-10d*))

   (defbench "gradient-100d" (:n-runs 500 :n-warmup 20)
     (ad:gradient #'sum-squares *x-100d*))

   (defbench "gradient-1000d" (:n-runs 100 :n-warmup 10)
     (ad:gradient #'sum-squares *x-1000d*))

   (defbench "hessian-vector-product-10d" (:n-runs 500 :n-warmup 20)
     (ad:hessian-vector-product #'sum-squares *x-10d* *v-10d*))))
```

> **Note on `ad:derivative`:** Forward-mode `derivative` takes a scalar function `(fn x)` where `x` is a scalar, not a list. Use `#'sum-squares` with a single-element but call it differently:

Actually, `ad:derivative` signature is `(derivative fn x)` where `x` is a single number. Use a scalar function:

```lisp
(defun scalar-square (x) (ad:* x x))

(defbench "derivative-1d" (:n-runs 10000 :n-warmup 100)
  (ad:derivative #'scalar-square 1.0d0))
```

Update the file accordingly.

**Step 2: Reload and quick-verify in REPL**

```lisp
(asdf:load-system :cl-acorn/benchmarks :force t)
(first (cl-acorn.benchmarks::run-ad-benchmarks))
;; => #S(BENCH-RESULT :NAME "derivative-1d" :MEAN-US <small positive> ...)
```

Expected: returns a `bench-result` with a positive `mean-us`.

**Step 3: Commit**

```bash
git add benchmarks/cl/bench-ad.lisp
git commit -m "feat(benchmarks): add AD engine benchmarks (derivative, gradient 1-1000d, HVP)"
```

---

## Task 4: `bench-distributions.lisp` — distribution benchmarks

**Files:**
- Modify: `benchmarks/cl/bench-distributions.lisp` (replace stub)

**Step 1: Write `bench-distributions.lisp`**

```lisp
(in-package #:cl-acorn.benchmarks)

(defun run-distributions-benchmarks ()
  "Run all distribution log-pdf benchmarks and return a list of BENCH-RESULT."
  (list
   (defbench "normal-log-pdf" (:n-runs 10000 :n-warmup 100)
     (dist:normal-log-pdf 0.5d0 :mu 0.0d0 :sigma 1.0d0))

   (defbench "gamma-log-pdf" (:n-runs 10000 :n-warmup 100)
     (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate 1.0d0))

   (defbench "beta-log-pdf" (:n-runs 10000 :n-warmup 100)
     (dist:beta-log-pdf 0.5d0 :alpha 2.0d0 :beta 3.0d0))

   (defbench "poisson-log-pdf" (:n-runs 10000 :n-warmup 100)
     (dist:poisson-log-pdf 3 :rate 2.0d0))))
```

> **Check API:** Confirm the exact keyword names for each distribution function. From `src/distributions/`:
> - `dist:normal-log-pdf x &key mu sigma`
> - `dist:gamma-log-pdf x &key shape rate`
> - `dist:beta-log-pdf x &key alpha beta`
> - `dist:poisson-log-pdf k &key rate`
>
> If they differ, read `src/distributions/normal.lisp` etc. with `lisp-read-file` and adjust.

**Step 2: Verify**

```lisp
(asdf:load-system :cl-acorn/benchmarks :force t)
(first (cl-acorn.benchmarks::run-distributions-benchmarks))
```

Expected: `bench-result` with positive `mean-us`.

**Step 3: Commit**

```bash
git add benchmarks/cl/bench-distributions.lisp
git commit -m "feat(benchmarks): add distribution log-pdf benchmarks"
```

---

## Task 5: `bench-inference.lisp` — HMC / NUTS / VI benchmarks

**Files:**
- Modify: `benchmarks/cl/bench-inference.lisp` (replace stub)

**Step 1: Write `bench-inference.lisp`**

For inference, we time ONE full run (not N-runs batches) and use `diagnostics-elapsed-seconds` for accuracy. We report `mean-us` = microseconds per sample.

```lisp
(in-package #:cl-acorn.benchmarks)

;;; 2D standard normal: log p(x1,x2) = -0.5*(x1^2 + x2^2)

(defun log-normal-2d (params)
  "Log-density of 2D standard normal. Uses ad: arithmetic."
  (let ((x1 (first params))
        (x2 (second params)))
    (ad:* -0.5d0 (ad:+ (ad:* x1 x1) (ad:* x2 x2)))))

(defparameter +bench-n-samples+ 500)
(defparameter +bench-n-warmup+  200)

(defun bench-hmc ()
  "Benchmark HMC on 2D standard normal. Returns BENCH-RESULT with mean-us = μs/sample."
  (multiple-value-bind (samples ar diag)
      (infer:hmc #'log-normal-2d '(0.0d0 0.0d0)
                 :n-samples +bench-n-samples+
                 :n-warmup  +bench-n-warmup+
                 :adapt-step-size t)
    (declare (ignore samples ar))
    (let* ((elapsed (infer:diagnostics-elapsed-seconds diag))
           (mean-us (* 1.0d6 (/ elapsed (float +bench-n-samples+ 0.0d0)))))
      (make-bench-result
       :name "hmc-2d-standard-normal"
       :mean-us mean-us
       :min-us  mean-us
       :max-us  mean-us
       :gc-bytes 0))))

(defun bench-nuts ()
  "Benchmark NUTS on 2D standard normal. Returns BENCH-RESULT."
  (multiple-value-bind (samples ar diag)
      (infer:nuts #'log-normal-2d '(0.0d0 0.0d0)
                  :n-samples +bench-n-samples+
                  :n-warmup  +bench-n-warmup+
                  :adapt-step-size t)
    (declare (ignore samples ar))
    (let* ((elapsed (infer:diagnostics-elapsed-seconds diag))
           (mean-us (* 1.0d6 (/ elapsed (float +bench-n-samples+ 0.0d0)))))
      (make-bench-result
       :name "nuts-2d-standard-normal"
       :mean-us mean-us
       :min-us  mean-us
       :max-us  mean-us
       :gc-bytes 0))))

(defun bench-vi ()
  "Benchmark VI (ADVI) on 2D standard normal. Returns BENCH-RESULT with mean-us = μs/iter."
  (let ((n-iterations 1000))
    (multiple-value-bind (mu sigma elbo diag)
        (infer:vi #'log-normal-2d 2
                  :n-iterations  n-iterations
                  :n-elbo-samples 10
                  :lr 0.01d0)
      (declare (ignore mu sigma elbo))
      (let* ((elapsed (infer:diagnostics-elapsed-seconds diag))
             (mean-us (* 1.0d6 (/ elapsed (float n-iterations 0.0d0)))))
        (make-bench-result
         :name "vi-2d-standard-normal"
         :mean-us mean-us
         :min-us  mean-us
         :max-us  mean-us
         :gc-bytes 0)))))

(defun run-inference-benchmarks ()
  "Run all inference benchmarks and return a list of BENCH-RESULT.
Each result's mean-us represents microseconds per sample (HMC/NUTS) or per iteration (VI)."
  (list (bench-hmc) (bench-nuts) (bench-vi)))
```

**Step 2: Verify**

```lisp
(asdf:load-system :cl-acorn/benchmarks :force t)
(cl-acorn.benchmarks::bench-hmc)
;; => #S(BENCH-RESULT :NAME "hmc-2d-standard-normal" :MEAN-US <positive> ...)
```

Expected: `mean-us` in the range of hundreds to tens-of-thousands (depends on machine).

**Step 3: Commit**

```bash
git add benchmarks/cl/bench-inference.lisp
git commit -m "feat(benchmarks): add HMC/NUTS/VI inference benchmarks using diagnostics timing"
```

---

## Task 6: `run-all.lisp` — runner and output

**Files:**
- Modify: `benchmarks/cl/run-all.lisp` (replace stub)

**Step 1: Write `run-all.lisp`**

```lisp
(in-package #:cl-acorn.benchmarks)

(defun run-all ()
  "Run all cl-acorn benchmarks and print results to *STANDARD-OUTPUT*.

Usage:
  (asdf:load-system :cl-acorn/benchmarks)
  (cl-acorn.benchmarks:run-all)"
  (format t "~%cl-acorn benchmark suite~%")
  (format t "========================~%")
  (format t "SBCL ~A  |  ~A~%"
          (lisp-implementation-version)
          (multiple-value-bind (s mi h d mo y)
              (decode-universal-time (get-universal-time))
            (declare (ignore s mi h))
            (format nil "~4,'0D-~2,'0D-~2,'0D" y mo d)))

  (print-bench-table "AD Engine" (run-ad-benchmarks))
  (print-bench-table "Distributions" (run-distributions-benchmarks))

  ;; Inference uses a different table format (samples/sec)
  (let ((inf-results (run-inference-benchmarks)))
    (format t "~%[Inference]~%")
    (format t "~30A | ~12A | ~12A~%"
            "Task" "μs/sample" "samples/sec")
    (format t "~30A-+-~12A-+-~12A~%"
            (make-string 30 :initial-element #\-)
            (make-string 12 :initial-element #\-)
            (make-string 12 :initial-element #\-))
    (dolist (r inf-results)
      (let ((sps (if (> (bench-result-mean-us r) 0.0d0)
                     (/ 1.0d6 (bench-result-mean-us r))
                     0.0d0)))
        (format t "~30A | ~12,2F | ~12,1F~%"
                (bench-result-name r)
                (bench-result-mean-us r)
                sps))))
  (values))
```

**Step 2: Run the full benchmark**

```bash
sbcl --eval "(asdf:load-system :cl-acorn/benchmarks)" \
     --eval "(cl-acorn.benchmarks:run-all)" \
     --quit 2>&1
```

Expected output resembles:
```
cl-acorn benchmark suite
========================
SBCL 2.x.x  |  2026-03-14

[AD Engine]
Task                           | Mean(μs)  | Min(μs)   | Max(μs)   | GC(bytes)
...

[Distributions]
...

[Inference]
Task                           | μs/sample   | samples/sec
...
```

**Step 3: Commit**

```bash
git add benchmarks/cl/run-all.lisp
git commit -m "feat(benchmarks): add run-all runner with formatted table output"
```

---

## Task 7: Python benchmark scripts

**Files:**
- Create: `benchmarks/python/requirements.txt`
- Create: `benchmarks/python/bench_ad_jax.py`
- Create: `benchmarks/python/bench_ad_torch.py`
- Create: `benchmarks/python/bench_inference_numpyro.py`

**Step 1: Create `requirements.txt`**

```
jax[cpu]>=0.4
torch>=2.0
numpyro>=0.13
```

**Step 2: Create `bench_ad_jax.py`**

```python
"""JAX autograd benchmarks — compare with cl-acorn AD engine."""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # use float64 like cl-acorn


def sum_squares(x):
    return jnp.sum(x ** 2)


grad_fn = jax.jit(jax.grad(sum_squares))


def bench(fn, x, n_runs=1000, n_warmup=10, n_trials=5):
    # warmup (and JIT compile)
    for _ in range(n_warmup):
        fn(x).block_until_ready()
    # trials
    trial_means = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn(x).block_until_ready()
        elapsed = time.perf_counter() - t0
        trial_means.append(elapsed / n_runs * 1e6)
    return {
        "mean": sum(trial_means) / len(trial_means),
        "min":  min(trial_means),
        "max":  max(trial_means),
    }


def main():
    print("\n[JAX AD Engine]")
    print(f"{'Task':<30} | {'Mean(μs)':>9} | {'Min(μs)':>9} | {'Max(μs)':>9}")
    print(f"{'-'*30}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    configs = [
        ("gradient-1d",    jnp.ones(1),    10000, 10),
        ("gradient-10d",   jnp.ones(10),    1000, 50),
        ("gradient-100d",  jnp.ones(100),    500, 20),
        ("gradient-1000d", jnp.ones(1000),   100, 10),
    ]
    for name, x, n_runs, n_warmup in configs:
        r = bench(grad_fn, x, n_runs=n_runs, n_warmup=n_warmup)
        print(f"{name:<30} | {r['mean']:>9.2f} | {r['min']:>9.2f} | {r['max']:>9.2f}")

    # HVP via jax.jvp(jax.grad(...))
    def hvp(x, v):
        return jax.jvp(jax.grad(sum_squares), (x,), (v,))[1]

    hvp_jit = jax.jit(hvp)
    x10 = jnp.ones(10)
    v10 = jnp.ones(10)
    r = bench(lambda x: hvp_jit(x, v10), x10, n_runs=500, n_warmup=20)
    print(f"{'hessian-vector-product-10d':<30} | {r['mean']:>9.2f} | {r['min']:>9.2f} | {r['max']:>9.2f}")


if __name__ == "__main__":
    main()
```

**Step 3: Create `bench_ad_torch.py`**

```python
"""PyTorch autograd benchmarks — compare with cl-acorn AD engine."""
import time
import torch


def sum_squares(x):
    return (x ** 2).sum()


def bench_grad(d, n_runs=1000, n_warmup=10, n_trials=5):
    # warmup
    for _ in range(n_warmup):
        x = torch.ones(d, requires_grad=True, dtype=torch.float64)
        sum_squares(x).backward()
    # trials
    trial_means = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            x = torch.ones(d, requires_grad=True, dtype=torch.float64)
            sum_squares(x).backward()
        elapsed = time.perf_counter() - t0
        trial_means.append(elapsed / n_runs * 1e6)
    return {
        "mean": sum(trial_means) / len(trial_means),
        "min":  min(trial_means),
        "max":  max(trial_means),
    }


def main():
    print("\n[PyTorch AD Engine]")
    print(f"{'Task':<30} | {'Mean(μs)':>9} | {'Min(μs)':>9} | {'Max(μs)':>9}")
    print(f"{'-'*30}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    configs = [
        ("gradient-1d",    1,    10000, 10),
        ("gradient-10d",   10,    1000, 50),
        ("gradient-100d",  100,    500, 20),
        ("gradient-1000d", 1000,   100, 10),
    ]
    for name, d, n_runs, n_warmup in configs:
        r = bench_grad(d, n_runs=n_runs, n_warmup=n_warmup)
        print(f"{name:<30} | {r['mean']:>9.2f} | {r['min']:>9.2f} | {r['max']:>9.2f}")


if __name__ == "__main__":
    main()
```

**Step 4: Create `bench_inference_numpyro.py`**

```python
"""NumPyro inference benchmarks — compare with cl-acorn HMC/NUTS/VI."""
import time
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import optax

jax.config.update("jax_enable_x64", True)

N_SAMPLES = 500
N_WARMUP  = 200


def model():
    numpyro.sample("x1", dist.Normal(0.0, 1.0))
    numpyro.sample("x2", dist.Normal(0.0, 1.0))


def bench_nuts():
    kernel = NUTS(model)
    mcmc   = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES, progress_bar=False)
    rng    = jax.random.PRNGKey(0)
    # warmup run
    mcmc.run(rng)
    # timed run
    rng = jax.random.PRNGKey(1)
    t0  = time.perf_counter()
    mcmc.run(rng)
    elapsed = time.perf_counter() - t0
    sps = N_SAMPLES / elapsed
    print(f"{'nuts-2d-standard-normal':<30} | {elapsed/N_SAMPLES*1e6:>12.2f} | {sps:>12.1f}")


def bench_hmc():
    kernel = HMC(model)
    mcmc   = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES, progress_bar=False)
    rng    = jax.random.PRNGKey(0)
    mcmc.run(rng)
    rng = jax.random.PRNGKey(1)
    t0  = time.perf_counter()
    mcmc.run(rng)
    elapsed = time.perf_counter() - t0
    sps = N_SAMPLES / elapsed
    print(f"{'hmc-2d-standard-normal':<30} | {elapsed/N_SAMPLES*1e6:>12.2f} | {sps:>12.1f}")


def bench_vi():
    n_iter = 1000
    guide  = AutoNormal(model)
    optimizer = numpyro.optim.Adam(step_size=0.01)
    svi    = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=10))
    rng    = jax.random.PRNGKey(0)
    # warmup
    state  = svi.init(rng)
    # timed
    t0 = time.perf_counter()
    for _ in range(n_iter):
        state, loss = svi.update(state)
    elapsed = time.perf_counter() - t0
    ups = n_iter / elapsed
    print(f"{'vi-2d-standard-normal':<30} | {elapsed/n_iter*1e6:>12.2f} | {ups:>12.1f}  (iter/s)")


def main():
    print("\n[NumPyro Inference]")
    print(f"{'Task':<30} | {'μs/sample':>12} | {'samples/sec':>12}")
    print(f"{'-'*30}-+-{'-'*12}-+-{'-'*12}")
    bench_hmc()
    bench_nuts()
    bench_vi()


if __name__ == "__main__":
    main()
```

**Step 5: Quick smoke test (if Python available)**

```bash
cd /home/wiz/cl-acorn
pip install -q -r benchmarks/python/requirements.txt 2>&1 | tail -3
python benchmarks/python/bench_ad_jax.py
```

Expected: prints a table with μs timings (no errors).

**Step 6: Commit**

```bash
git add benchmarks/python/
git commit -m "feat(benchmarks): add JAX, PyTorch, and NumPyro Python benchmark scripts"
```

---

## Task 8: `benchmarks/README.md`

**Files:**
- Create: `benchmarks/README.md`

**Step 1: Create README**

````markdown
# cl-acorn Benchmarks

Performance measurements for the cl-acorn AD engine, distribution functions,
and inference algorithms, compared against JAX and PyTorch (CPU only).

## Running CL Benchmarks

```lisp
;; From a REPL with cl-acorn on the ASDF path
(asdf:load-system :cl-acorn/benchmarks)
(cl-acorn.benchmarks:run-all)
```

Or from the command line:

```bash
sbcl --eval "(asdf:load-system :cl-acorn/benchmarks)" \
     --eval "(cl-acorn.benchmarks:run-all)" \
     --quit
```

## Running Python Benchmarks

```bash
pip install -r benchmarks/python/requirements.txt

python benchmarks/python/bench_ad_jax.py
python benchmarks/python/bench_ad_torch.py
python benchmarks/python/bench_inference_numpyro.py
```

## Interpreting Results

- **AD Engine / Distributions**: `Mean(μs)` = microseconds per individual call.
  Lower is better. GC(bytes) indicates allocation pressure.
- **Inference**: `μs/sample` = time per posterior sample after warmup.
  `samples/sec` = throughput. Higher is better for samples/sec.

## Example Results

*(Update this table after each run on your machine)*

### AD Engine

| Task | cl-acorn (μs) | JAX (μs) | PyTorch (μs) |
|------|:---:|:---:|:---:|
| gradient-1d | | | |
| gradient-10d | | | |
| gradient-100d | | | |
| gradient-1000d | | | |
| hessian-vector-product-10d | | | |

### Inference

| Task | cl-acorn (samples/sec) | NumPyro (samples/sec) |
|------|:---:|:---:|
| hmc-2d-standard-normal | | |
| nuts-2d-standard-normal | | |
| vi-2d-standard-normal | | |

## Notes

- cl-acorn uses scalar lists; JAX/PyTorch use arrays — the overhead includes
  list traversal which array-based libraries avoid.
- JAX results include JIT-compiled times (after warmup).
- NumPyro inference also JIT-compiles; first run includes compilation.
- All measurements: CPU only, single-threaded, no GPU.
````

**Step 2: Commit**

```bash
git add benchmarks/README.md
git commit -m "docs(benchmarks): add README with run instructions and results template"
```

---

## Final Verification

```bash
# Full CL benchmark run
sbcl --eval "(asdf:load-system :cl-acorn/benchmarks)" \
     --eval "(cl-acorn.benchmarks:run-all)" \
     --quit 2>&1

# Existing tests still pass
rove cl-acorn.asd 2>&1 | grep -E "completed|failed"
```

Expected:
- Benchmark output with all three sections (AD Engine, Distributions, Inference)
- `2426 tests completed, all passed` (no regressions)
