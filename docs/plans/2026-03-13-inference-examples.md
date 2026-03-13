# Inference Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three self-contained examples under `examples/` that demonstrate `infer:hmc`, `infer:nuts`, and `infer:vi` using the same Bayesian linear regression model, allowing direct method comparison.

**Architecture:** Three independent directories (`09-hmc-bayes-regression`, `10-nuts-bayes-regression`, `11-vi-bayes-regression`), each with a `main.lisp` and `README.md`. All three share the same hardcoded 20-point synthetic dataset (y = 2x + 1 + noise, σ=0.5) and the same `log-posterior` definition. Inline data — no separate `data.lisp`.

**Tech Stack:** cl-acorn (ad, dist, infer packages), SBCL, Rove (existing test suite for regression check only)

---

## Shared constants (copy into each main.lisp)

These are used in all three examples verbatim:

```lisp
;;; Synthetic data: y = 2x + 1 + noise, noise ~ N(0, 0.5^2)
(defparameter *data-xs*
  #(0.1d0 0.2d0 0.3d0 0.4d0 0.5d0 0.6d0 0.7d0 0.8d0 0.9d0 1.0d0
    1.1d0 1.2d0 1.3d0 1.4d0 1.5d0 1.6d0 1.7d0 1.8d0 1.9d0 2.0d0))

(defparameter *data-ys*
  #(1.35d0 1.52d0 1.58d0 1.84d0 2.12d0 2.31d0 2.45d0 2.87d0 2.91d0 3.08d0
    3.22d0 3.44d0 3.58d0 3.87d0 4.01d0 4.28d0 4.38d0 4.67d0 4.81d0 5.12d0))

(defun log-posterior (params)
  "Log-posterior for y = w*x + b, sigma=0.5.
PARAMS is a list (w b). Uses AD-transparent arithmetic."
  (let* ((w (first params))
         (b (second params))
         (sigma 0.5d0)
         (log-prior (ad:+ (dist:normal-log-pdf w 0.0d0 1.0d0)
                          (dist:normal-log-pdf b 0.0d0 10.0d0)))
         (log-lik 0.0d0))
    (dotimes (i (length *data-xs*))
      (let ((pred (ad:+ (ad:* w (aref *data-xs* i)) b)))
        (setf log-lik (ad:+ log-lik
                            (dist:normal-log-pdf pred (aref *data-ys* i) sigma)))))
    (ad:+ log-prior log-lik)))

(defun sample-mean (xs)
  "Sample mean of a list of numbers."
  (/ (reduce #'+ xs) (length xs)))

(defun sample-std (xs)
  "Sample standard deviation of a list of numbers."
  (let* ((mean (sample-mean xs))
         (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) xs))
                      (max 1 (1- (length xs))))))
    (sqrt variance)))
```

---

## Task 1: Example 09 — HMC

**Files:**
- Create: `examples/09-hmc-bayes-regression/main.lisp`
- Create: `examples/09-hmc-bayes-regression/README.md`

### Step 1: Create the directory

```bash
mkdir -p examples/09-hmc-bayes-regression
```

### Step 2: Write `main.lisp`

```lisp
;;;; main.lisp --- Bayesian linear regression via Hamiltonian Monte Carlo
;;;;
;;;; Demonstrates using cl-acorn's HMC sampler to estimate the posterior
;;;; distribution of w and b in the model y = w*x + b (sigma=0.5).
;;;;
;;;; Usage:
;;;;   (load "examples/09-hmc-bayes-regression/main.lisp")

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.hmc-bayes-regression
  (:use #:cl)
  (:export #:log-posterior
           #:run-example))

(in-package #:cl-acorn.examples.hmc-bayes-regression)

;;; --------------------------------------------------------------------------
;;; Synthetic data: y = 2x + 1 + noise, noise ~ N(0, 0.5^2)
;;; --------------------------------------------------------------------------

(defparameter *data-xs*
  #(0.1d0 0.2d0 0.3d0 0.4d0 0.5d0 0.6d0 0.7d0 0.8d0 0.9d0 1.0d0
    1.1d0 1.2d0 1.3d0 1.4d0 1.5d0 1.6d0 1.7d0 1.8d0 1.9d0 2.0d0))

(defparameter *data-ys*
  #(1.35d0 1.52d0 1.58d0 1.84d0 2.12d0 2.31d0 2.45d0 2.87d0 2.91d0 3.08d0
    3.22d0 3.44d0 3.58d0 3.87d0 4.01d0 4.28d0 4.38d0 4.67d0 4.81d0 5.12d0))

;;; --------------------------------------------------------------------------
;;; Model: log p(w, b | data)
;;; --------------------------------------------------------------------------

(defun log-posterior (params)
  "Log-posterior for y = w*x + b, sigma=0.5.
PARAMS is a list (w b). Uses AD-transparent arithmetic for gradient computation."
  (let* ((w (first params))
         (b (second params))
         (sigma 0.5d0)
         (log-prior (ad:+ (dist:normal-log-pdf w 0.0d0 1.0d0)
                          (dist:normal-log-pdf b 0.0d0 10.0d0)))
         (log-lik 0.0d0))
    (dotimes (i (length *data-xs*))
      (let ((pred (ad:+ (ad:* w (aref *data-xs* i)) b)))
        (setf log-lik (ad:+ log-lik
                            (dist:normal-log-pdf pred (aref *data-ys* i) sigma)))))
    (ad:+ log-prior log-lik)))

;;; --------------------------------------------------------------------------
;;; Statistics helpers
;;; --------------------------------------------------------------------------

(defun sample-mean (xs)
  "Sample mean of a list of numbers."
  (/ (reduce #'+ xs) (length xs)))

(defun sample-std (xs)
  "Sample standard deviation of a list of numbers."
  (let* ((mean (sample-mean xs))
         (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) xs))
                      (max 1 (1- (length xs))))))
    (sqrt variance)))

;;; --------------------------------------------------------------------------
;;; Example runner
;;; --------------------------------------------------------------------------

(defun run-example ()
  "Run HMC on the Bayesian linear regression model and print posterior statistics."
  (format t "~%=== HMC Bayesian Linear Regression ===~%")
  (format t "Model: y = w*x + b,  sigma=0.5~%")
  (format t "True values: w=2.0, b=1.0~%")
  (format t "Data: ~D observations~%~%" (length *data-xs*))
  (format t "Running HMC (1000 samples, 500 warmup)...~%")
  (multiple-value-bind (samples accept-rate)
      (infer:hmc #'log-posterior '(0.0d0 0.0d0)
                 :n-samples 1000
                 :n-warmup 500
                 :step-size 0.05d0
                 :n-leapfrog 10
                 :adapt-step-size t)
    (let ((ws (mapcar #'first samples))
          (bs (mapcar #'second samples)))
      (format t "~%Posterior statistics (~D samples after 500 warmup):~%"
              (length samples))
      (format t "  w:  mean=~6,3F  std=~6,3F   [true: 2.0]~%"
              (sample-mean ws) (sample-std ws))
      (format t "  b:  mean=~6,3F  std=~6,3F   [true: 1.0]~%"
              (sample-mean bs) (sample-std bs))
      (format t "Accept rate: ~,3F~%" accept-rate))))

(run-example)
```

### Step 3: Load and verify it runs

In SBCL REPL:
```lisp
(load "examples/09-hmc-bayes-regression/main.lisp")
```

Expected: Prints posterior statistics. `w` mean should be within ±0.3 of 2.0, `b` mean within ±0.3 of 1.0. Accept rate should be between 0.5 and 0.99.

### Step 4: Write `README.md`

```markdown
# Bayesian Linear Regression with HMC

Demonstrates using cl-acorn's Hamiltonian Monte Carlo (HMC) sampler to estimate
the posterior distribution of parameters in a Bayesian linear regression model.

## The Model

We fit a linear relationship y = w·x + b to observed data with Gaussian noise:

    y_i = w · x_i + b + ε_i,   ε_i ~ N(0, 0.5²)

With priors:
- w ~ N(0, 1)    — slope
- b ~ N(0, 10)   — intercept

Unlike maximum likelihood estimation, Bayesian inference gives the full posterior
distribution p(w, b | data), capturing parameter uncertainty.

## HMC

Hamiltonian Monte Carlo uses gradient information to propose efficient moves in
parameter space. It simulates Hamiltonian dynamics (like a ball rolling in a
potential energy landscape) to explore the posterior with low autocorrelation.

Key parameters:
- `:step-size` — leapfrog integration step size (adapted during warmup)
- `:n-leapfrog` — number of leapfrog steps per proposal
- `:adapt-step-size` — enable dual averaging adaptation during warmup

## Usage

```lisp
(load "examples/09-hmc-bayes-regression/main.lisp")
```

## Expected Output

```
=== HMC Bayesian Linear Regression ===
Model: y = w*x + b,  sigma=0.5
True values: w=2.0, b=1.0
Data: 20 observations

Running HMC (1000 samples, 500 warmup)...

Posterior statistics (1000 samples after 500 warmup):
  w:  mean= 2.012  std= 0.085   [true: 2.0]
  b:  mean= 0.997  std= 0.118   [true: 1.0]
Accept rate: 0.832
```

Results vary due to sampling randomness; posterior means should be close to the true values.
```

### Step 5: Commit

```bash
git add examples/09-hmc-bayes-regression/
git commit -m "feat(examples): add HMC Bayesian linear regression example"
```

---

## Task 2: Example 10 — NUTS

**Files:**
- Create: `examples/10-nuts-bayes-regression/main.lisp`
- Create: `examples/10-nuts-bayes-regression/README.md`

### Step 1: Create the directory

```bash
mkdir -p examples/10-nuts-bayes-regression
```

### Step 2: Write `main.lisp`

Same structure as Task 1, with these changes:
- Package: `#:cl-acorn.examples.nuts-bayes-regression`
- Header comment: "Bayesian linear regression via No-U-Turn Sampler"
- `run-example` uses `infer:nuts` instead of `infer:hmc`:

```lisp
;;;; main.lisp --- Bayesian linear regression via No-U-Turn Sampler (NUTS)
;;;;
;;;; Demonstrates using cl-acorn's NUTS sampler to estimate the posterior
;;;; distribution of w and b in the model y = w*x + b (sigma=0.5).
;;;; NUTS adapts trajectory length automatically; no need to tune n-leapfrog.
;;;;
;;;; Usage:
;;;;   (load "examples/10-nuts-bayes-regression/main.lisp")

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.nuts-bayes-regression
  (:use #:cl)
  (:export #:log-posterior
           #:run-example))

(in-package #:cl-acorn.examples.nuts-bayes-regression)

;;; --------------------------------------------------------------------------
;;; Synthetic data: y = 2x + 1 + noise, noise ~ N(0, 0.5^2)
;;; --------------------------------------------------------------------------

(defparameter *data-xs*
  #(0.1d0 0.2d0 0.3d0 0.4d0 0.5d0 0.6d0 0.7d0 0.8d0 0.9d0 1.0d0
    1.1d0 1.2d0 1.3d0 1.4d0 1.5d0 1.6d0 1.7d0 1.8d0 1.9d0 2.0d0))

(defparameter *data-ys*
  #(1.35d0 1.52d0 1.58d0 1.84d0 2.12d0 2.31d0 2.45d0 2.87d0 2.91d0 3.08d0
    3.22d0 3.44d0 3.58d0 3.87d0 4.01d0 4.28d0 4.38d0 4.67d0 4.81d0 5.12d0))

;;; --------------------------------------------------------------------------
;;; Model: log p(w, b | data)
;;; --------------------------------------------------------------------------

(defun log-posterior (params)
  "Log-posterior for y = w*x + b, sigma=0.5.
PARAMS is a list (w b). Uses AD-transparent arithmetic for gradient computation."
  (let* ((w (first params))
         (b (second params))
         (sigma 0.5d0)
         (log-prior (ad:+ (dist:normal-log-pdf w 0.0d0 1.0d0)
                          (dist:normal-log-pdf b 0.0d0 10.0d0)))
         (log-lik 0.0d0))
    (dotimes (i (length *data-xs*))
      (let ((pred (ad:+ (ad:* w (aref *data-xs* i)) b)))
        (setf log-lik (ad:+ log-lik
                            (dist:normal-log-pdf pred (aref *data-ys* i) sigma)))))
    (ad:+ log-prior log-lik)))

;;; --------------------------------------------------------------------------
;;; Statistics helpers
;;; --------------------------------------------------------------------------

(defun sample-mean (xs)
  "Sample mean of a list of numbers."
  (/ (reduce #'+ xs) (length xs)))

(defun sample-std (xs)
  "Sample standard deviation of a list of numbers."
  (let* ((mean (sample-mean xs))
         (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) xs))
                      (max 1 (1- (length xs))))))
    (sqrt variance)))

;;; --------------------------------------------------------------------------
;;; Example runner
;;; --------------------------------------------------------------------------

(defun run-example ()
  "Run NUTS on the Bayesian linear regression model and print posterior statistics."
  (format t "~%=== NUTS Bayesian Linear Regression ===~%")
  (format t "Model: y = w*x + b,  sigma=0.5~%")
  (format t "True values: w=2.0, b=1.0~%")
  (format t "Data: ~D observations~%~%" (length *data-xs*))
  (format t "Running NUTS (1000 samples, 500 warmup)...~%")
  (multiple-value-bind (samples accept-rate)
      (infer:nuts #'log-posterior '(0.0d0 0.0d0)
                  :n-samples 1000
                  :n-warmup 500
                  :adapt-step-size t)
    (let ((ws (mapcar #'first samples))
          (bs (mapcar #'second samples)))
      (format t "~%Posterior statistics (~D samples after 500 warmup):~%"
              (length samples))
      (format t "  w:  mean=~6,3F  std=~6,3F   [true: 2.0]~%"
              (sample-mean ws) (sample-std ws))
      (format t "  b:  mean=~6,3F  std=~6,3F   [true: 1.0]~%"
              (sample-mean bs) (sample-std bs))
      (format t "Accept rate: ~,3F~%" accept-rate))))

(run-example)
```

### Step 3: Load and verify it runs

```lisp
(load "examples/10-nuts-bayes-regression/main.lisp")
```

Expected: Same acceptance criteria as Task 1. NUTS accept rate is typically higher (0.8–0.99) because it uses multinomial sampling with automatic trajectory length.

### Step 4: Write `README.md`

```markdown
# Bayesian Linear Regression with NUTS

Demonstrates using cl-acorn's No-U-Turn Sampler (NUTS) to estimate the posterior
distribution of parameters in a Bayesian linear regression model.

## The Model

    y_i = w · x_i + b + ε_i,   ε_i ~ N(0, 0.5²)

Priors:
- w ~ N(0, 1)    — slope
- b ~ N(0, 10)   — intercept

## NUTS vs HMC

HMC requires tuning the number of leapfrog steps (`:n-leapfrog`). NUTS eliminates
this by building a balanced binary tree of leapfrog steps and stopping when it
detects a U-turn in trajectory direction. This makes NUTS more robust and usually
achieves higher effective sample size per gradient evaluation.

Key parameters:
- `:adapt-step-size` — dual averaging step-size adaptation during warmup (default: t)
- `:max-tree-depth` — upper bound on tree depth (default: 10, i.e. max 1024 steps)

## Usage

```lisp
(load "examples/10-nuts-bayes-regression/main.lisp")
```

## Expected Output

```
=== NUTS Bayesian Linear Regression ===
Model: y = w*x + b,  sigma=0.5
True values: w=2.0, b=1.0
Data: 20 observations

Running NUTS (1000 samples, 500 warmup)...

Posterior statistics (1000 samples after 500 warmup):
  w:  mean= 2.008  std= 0.083   [true: 2.0]
  b:  mean= 1.002  std= 0.115   [true: 1.0]
Accept rate: 0.921
```

Compare with example 09 (HMC) — NUTS typically achieves a higher effective sample
rate without requiring manual tuning of leapfrog steps.
```

### Step 5: Commit

```bash
git add examples/10-nuts-bayes-regression/
git commit -m "feat(examples): add NUTS Bayesian linear regression example"
```

---

## Task 3: Example 11 — VI (ADVI)

**Files:**
- Create: `examples/11-vi-bayes-regression/main.lisp`
- Create: `examples/11-vi-bayes-regression/README.md`

### Step 1: Create the directory

```bash
mkdir -p examples/11-vi-bayes-regression
```

### Step 2: Write `main.lisp`

VI returns `(values mu-list sigma-list elbo-history)`, not samples.

```lisp
;;;; main.lisp --- Bayesian linear regression via Variational Inference (ADVI)
;;;;
;;;; Demonstrates using cl-acorn's mean-field ADVI to approximate the posterior
;;;; distribution of w and b in the model y = w*x + b (sigma=0.5).
;;;; VI is faster than MCMC but returns a Gaussian approximation, not exact samples.
;;;;
;;;; Usage:
;;;;   (load "examples/11-vi-bayes-regression/main.lisp")

(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.vi-bayes-regression
  (:use #:cl)
  (:export #:log-posterior
           #:run-example))

(in-package #:cl-acorn.examples.vi-bayes-regression)

;;; --------------------------------------------------------------------------
;;; Synthetic data: y = 2x + 1 + noise, noise ~ N(0, 0.5^2)
;;; --------------------------------------------------------------------------

(defparameter *data-xs*
  #(0.1d0 0.2d0 0.3d0 0.4d0 0.5d0 0.6d0 0.7d0 0.8d0 0.9d0 1.0d0
    1.1d0 1.2d0 1.3d0 1.4d0 1.5d0 1.6d0 1.7d0 1.8d0 1.9d0 2.0d0))

(defparameter *data-ys*
  #(1.35d0 1.52d0 1.58d0 1.84d0 2.12d0 2.31d0 2.45d0 2.87d0 2.91d0 3.08d0
    3.22d0 3.44d0 3.58d0 3.87d0 4.01d0 4.28d0 4.38d0 4.67d0 4.81d0 5.12d0))

;;; --------------------------------------------------------------------------
;;; Model: log p(w, b | data)
;;; --------------------------------------------------------------------------

(defun log-posterior (params)
  "Log-posterior for y = w*x + b, sigma=0.5.
PARAMS is a list (w b). Uses AD-transparent arithmetic for gradient computation."
  (let* ((w (first params))
         (b (second params))
         (sigma 0.5d0)
         (log-prior (ad:+ (dist:normal-log-pdf w 0.0d0 1.0d0)
                          (dist:normal-log-pdf b 0.0d0 10.0d0)))
         (log-lik 0.0d0))
    (dotimes (i (length *data-xs*))
      (let ((pred (ad:+ (ad:* w (aref *data-xs* i)) b)))
        (setf log-lik (ad:+ log-lik
                            (dist:normal-log-pdf pred (aref *data-ys* i) sigma)))))
    (ad:+ log-prior log-lik)))

;;; --------------------------------------------------------------------------
;;; Example runner
;;; --------------------------------------------------------------------------

(defun run-example ()
  "Run VI (ADVI) on the Bayesian linear regression model and print results."
  (format t "~%=== VI (ADVI) Bayesian Linear Regression ===~%")
  (format t "Model: y = w*x + b,  sigma=0.5~%")
  (format t "True values: w=2.0, b=1.0~%")
  (format t "Data: ~D observations~%~%" (length *data-xs*))
  (format t "Running VI (2000 iterations)...~%")
  (multiple-value-bind (mu-list sigma-list elbo-history)
      (infer:vi #'log-posterior 2
                :n-iterations 2000
                :n-elbo-samples 10
                :lr 0.01d0)
    (format t "~%Variational posterior (mean-field Gaussian):~%")
    (format t "  w:  mean=~6,3F  std=~6,3F   [true: 2.0]~%"
            (first mu-list) (first sigma-list))
    (format t "  b:  mean=~6,3F  std=~6,3F   [true: 1.0]~%"
            (second mu-list) (second sigma-list))
    (format t "Final ELBO: ~,3F  (higher is better)~%"
            (car (last elbo-history)))))

(run-example)
```

### Step 3: Load and verify it runs

```lisp
(load "examples/11-vi-bayes-regression/main.lisp")
```

Expected: Prints variational means and standard deviations. Means should be within ±0.3 of true values. ELBO should be a finite negative number (typically around -40 to -60 for this dataset).

### Step 4: Write `README.md`

```markdown
# Bayesian Linear Regression with VI (ADVI)

Demonstrates using cl-acorn's mean-field Automatic Differentiation Variational
Inference (ADVI) to approximate the posterior distribution of parameters in a
Bayesian linear regression model.

## The Model

    y_i = w · x_i + b + ε_i,   ε_i ~ N(0, 0.5²)

Priors:
- w ~ N(0, 1)    — slope
- b ~ N(0, 10)   — intercept

## Variational Inference vs MCMC

Instead of drawing samples from the posterior (as HMC/NUTS do), VI finds the
best Gaussian approximation q(w, b) = N(μ_w, σ_w) · N(μ_b, σ_b) by maximizing
the Evidence Lower BOund (ELBO). This is faster than MCMC but approximate —
the mean-field assumption (independent Gaussians) misses posterior correlations.

Returns directly: posterior means and standard deviations (no samples needed).

Key parameters:
- `:n-iterations` — Adam optimizer steps
- `:n-elbo-samples` — Monte Carlo samples per ELBO gradient estimate
- `:lr` — Adam learning rate

## Usage

```lisp
(load "examples/11-vi-bayes-regression/main.lisp")
```

## Expected Output

```
=== VI (ADVI) Bayesian Linear Regression ===
Model: y = w*x + b,  sigma=0.5
True values: w=2.0, b=1.0
Data: 20 observations

Running VI (2000 iterations)...

Variational posterior (mean-field Gaussian):
  w:  mean= 2.009  std= 0.088   [true: 2.0]
  b:  mean= 1.001  std= 0.121   [true: 1.0]
Final ELBO: -42.317  (higher is better)
```

Compare with examples 09 (HMC) and 10 (NUTS): VI is faster but gives an
approximate posterior. For this well-conditioned model the difference is small.
```

### Step 5: Commit

```bash
git add examples/11-vi-bayes-regression/
git commit -m "feat(examples): add VI (ADVI) Bayesian linear regression example"
```

---

## Task 4: Regression check — existing test suite still passes

After adding the examples, verify the library itself is unaffected.

### Step 1: Run the full test suite

```bash
# From the project root
(run-tests :system "cl-acorn/tests")
```

Or via REPL:
```lisp
(asdf:test-system :cl-acorn)
```

Expected: All 170 tests pass. Zero failures. If anything fails it is a pre-existing issue unrelated to the examples.

### Step 2: Commit if needed

If the test run required any fix:
```bash
git add <fixed-files>
git commit -m "fix: <description>"
```

---

## Success Criteria

- `(load "examples/09-hmc-bayes-regression/main.lisp")` runs without error
- `(load "examples/10-nuts-bayes-regression/main.lisp")` runs without error
- `(load "examples/11-vi-bayes-regression/main.lisp")` runs without error
- All three print posterior means within ±0.3 of true values (w=2.0, b=1.0)
- Existing 170 tests still pass
- Each README explains the method, the model, usage, and expected output
