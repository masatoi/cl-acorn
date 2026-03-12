# Inference Building Blocks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add probability distributions (6 types), optimizers (SGD, Adam), and HMC sampler as composable building blocks for probabilistic modeling.

**Architecture:** Three new sub-packages (`cl-acorn.distributions`, `cl-acorn.optimizers`, `cl-acorn.inference`) with global nicknames (`dist`, `opt`, `infer`). Distribution log-pdf functions are AD-transparent (work with dual/tape-node values). Optimizers are pure-functional (return new parameter lists). HMC uses `ad:gradient` internally for leapfrog integration.

**Tech Stack:** Common Lisp, cl-acorn.ad (AD engine), Rove (tests), no external dependencies.

---

### Task 1: ASDF Scaffolding and Package Definitions

Create directory structure, package definitions for all 3 sub-packages, and update ASDF system.

**Files:**
- Create: `src/distributions/package.lisp`
- Create: `src/optimizers/package.lisp`
- Create: `src/inference/package.lisp`
- Modify: `cl-acorn.asd`

**Step 1: Create directory structure**

```bash
mkdir -p src/distributions src/optimizers src/inference
```

**Step 2: Create `src/distributions/package.lisp`**

```lisp
(defpackage #:cl-acorn.distributions
  (:nicknames #:dist)
  (:use #:cl)
  (:export
   ;; Utilities
   #:log-gammaln
   ;; Normal
   #:normal-log-pdf #:normal-sample
   ;; Gamma
   #:gamma-log-pdf #:gamma-sample
   ;; Beta
   #:beta-log-pdf #:beta-sample
   ;; Uniform
   #:uniform-log-pdf #:uniform-sample
   ;; Bernoulli
   #:bernoulli-log-pdf #:bernoulli-sample
   ;; Poisson
   #:poisson-log-pdf #:poisson-sample))
```

**Step 3: Create `src/optimizers/package.lisp`**

```lisp
(defpackage #:cl-acorn.optimizers
  (:nicknames #:opt)
  (:use #:cl)
  (:export
   #:sgd-step
   #:adam-state
   #:make-adam-state
   #:adam-step))
```

**Step 4: Create `src/inference/package.lisp`**

```lisp
(defpackage #:cl-acorn.inference
  (:nicknames #:infer)
  (:use #:cl)
  (:export
   #:hmc))
```

**Step 5: Update `cl-acorn.asd`**

Add three new modules inside the existing `"src"` module (after `"gradient"`). Update version to `"0.3.0"`. Add stub files for compilation. Also add test file entries.

The `"src"` module already uses `:serial t`, so load order is guaranteed: AD files first, then distributions, then optimizers, then inference.

```lisp
(defsystem "cl-acorn"
  :version "0.3.0"
  :author ""
  :license "MIT"
  :depends-on ()
  :components ((:module "src"
                :serial t
                :components
                ((:file "package")
                 (:file "dual")
                 (:file "arithmetic")
                 (:file "transcendental")
                 (:file "interface")
                 (:file "tape")
                 (:file "reverse-arithmetic")
                 (:file "reverse-transcendental")
                 (:file "gradient")
                 (:module "distributions"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "util")
                   (:file "normal")
                   (:file "uniform")
                   (:file "bernoulli")
                   (:file "gamma")
                   (:file "beta")
                   (:file "poisson")))
                 (:module "optimizers"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "sgd")
                   (:file "adam")))
                 (:module "inference"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "hmc"))))))
  :description "Automatic differentiation and probabilistic inference building blocks"
  :in-order-to ((test-op (test-op "cl-acorn/tests"))))

(defsystem "cl-acorn/tests"
  :author ""
  :license "MIT"
  :depends-on ("cl-acorn"
               "rove")
  :components ((:module "tests"
                :serial t
                :components
                ((:file "util")
                 (:file "dual-test")
                 (:file "arithmetic-test")
                 (:file "transcendental-test")
                 (:file "derivative-test")
                 (:file "tape-test")
                 (:file "reverse-arithmetic-test")
                 (:file "reverse-transcendental-test")
                 (:file "gradient-test")
                 (:file "distributions-test")
                 (:file "optimizers-test")
                 (:file "hmc-test"))))
  :description "Test system for cl-acorn"
  :perform (test-op (op c) (symbol-call :rove :run c)))
```

**Step 6: Create stub files**

Create minimal stub files so the system compiles. Each stub has just `(in-package ...)`.

Stubs needed:
- `src/distributions/util.lisp` — `(in-package #:cl-acorn.distributions)`
- `src/distributions/normal.lisp` — same
- `src/distributions/uniform.lisp` — same
- `src/distributions/bernoulli.lisp` — same
- `src/distributions/gamma.lisp` — same
- `src/distributions/beta.lisp` — same
- `src/distributions/poisson.lisp` — same
- `src/optimizers/sgd.lisp` — `(in-package #:cl-acorn.optimizers)`
- `src/optimizers/adam.lisp` — same
- `src/inference/hmc.lisp` — `(in-package #:cl-acorn.inference)`

Test stubs:
- `tests/distributions-test.lisp`:
```lisp
(defpackage #:cl-acorn/tests/distributions-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/distributions-test)
```
- `tests/optimizers-test.lisp` — same pattern with `optimizers-test`
- `tests/hmc-test.lisp` — same pattern with `hmc-test`

**Step 7: Verify system loads**

Run: `load-system cl-acorn` + `run-tests cl-acorn/tests`
Expected: System loads, all 107 existing tests pass.

**Step 8: Commit**

```
feat: add scaffolding for distributions, optimizers, inference packages
```

---

### Task 2: Distribution Utilities (log-gammaln)

Implement the Lanczos approximation for the log-gamma function. Used internally by gamma, beta, and poisson distributions for normalization constants.

**Files:**
- Modify: `src/distributions/util.lisp`
- Modify: `tests/distributions-test.lisp`

**Step 1: Write tests**

Add to `tests/distributions-test.lisp`:

```lisp
;;; --- log-gammaln tests ---

(deftest test-log-gammaln-integers
  (testing "log-gammaln at integer values matches log((n-1)!)"
    (ok (approx= (dist:log-gammaln 1.0d0) 0.0d0 1d-10))          ; Γ(1) = 0! = 1
    (ok (approx= (dist:log-gammaln 2.0d0) 0.0d0 1d-10))          ; Γ(2) = 1! = 1
    (ok (approx= (dist:log-gammaln 3.0d0) (log 2.0d0) 1d-10))    ; Γ(3) = 2! = 2
    (ok (approx= (dist:log-gammaln 5.0d0) (log 24.0d0) 1d-10))   ; Γ(5) = 4! = 24
    (ok (approx= (dist:log-gammaln 7.0d0) (log 720.0d0) 1d-10)))) ; Γ(7) = 6! = 720

(deftest test-log-gammaln-half
  (testing "log-gammaln at 0.5 equals log(sqrt(pi))"
    (ok (approx= (dist:log-gammaln 0.5d0)
                 (* 0.5d0 (log pi))
                 1d-10))))
```

**Step 2: Implement `src/distributions/util.lisp`**

```lisp
(in-package #:cl-acorn.distributions)

;;; Lanczos approximation coefficients (g=7, n=9)
(defconstant +lanczos-g+ 7.0d0)

(defconstant +lanczos-coefficients+
  #(0.99999999999980993d0
    676.5203681218851d0
    -1259.1392167224028d0
    771.32342877765313d0
    -176.61502916214059d0
    12.507343278686905d0
    -0.13857109526572012d0
    9.9843695780195716d-6
    1.5056327351493116d-7))

(defconstant +log-2pi/2+ (* 0.5d0 (log (* 2.0d0 pi)))
  "Precomputed 0.5 * log(2 * pi).")

(defun log-gammaln (z)
  "Log of the gamma function via Lanczos approximation.
Z must be a positive real number. Returns a double-float.
Used for normalization constants in gamma, beta, and poisson distributions."
  (let ((z (coerce z 'double-float)))
    (if (< z 0.5d0)
        ;; Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        (- (log (/ pi (sin (* pi z))))
           (log-gammaln (- 1.0d0 z)))
        ;; Lanczos approximation for z >= 0.5
        (let* ((z (- z 1.0d0))
               (x (aref +lanczos-coefficients+ 0)))
          (loop for i from 1 below (length +lanczos-coefficients+)
                do (incf x (/ (aref +lanczos-coefficients+ i) (+ z (coerce i 'double-float)))))
          (let ((t-val (+ z +lanczos-g+ 0.5d0)))
            (+ +log-2pi/2+
               (* (+ z 0.5d0) (log t-val))
               (- t-val)
               (log x)))))))
```

**Step 3: Run tests**

Run: `load-system cl-acorn` + `run-tests cl-acorn/tests`
Expected: All tests pass (107 existing + 2 new = 109).

**Step 4: Commit**

```
feat(distributions): add log-gammaln via Lanczos approximation
```

---

### Task 3: Normal Distribution

The most fundamental distribution. Required by HMC for momentum sampling.

**Files:**
- Modify: `src/distributions/normal.lisp`
- Modify: `tests/distributions-test.lisp`

**Step 1: Write tests**

Add to `tests/distributions-test.lisp`:

```lisp
;;; --- normal distribution tests ---

(deftest test-normal-log-pdf-standard
  (testing "standard normal log-pdf at known points"
    ;; log N(0|0,1) = -0.5*log(2π) ≈ -0.9189
    (ok (approx= (dist:normal-log-pdf 0.0d0)
                 -0.9189385332046727d0 1d-10))
    ;; log N(1|0,1) = -0.5*log(2π) - 0.5 ≈ -1.4189
    (ok (approx= (dist:normal-log-pdf 1.0d0)
                 -1.4189385332046727d0 1d-10))))

(deftest test-normal-log-pdf-nonstandard
  (testing "normal log-pdf with explicit mu and sigma"
    ;; log N(3|2,0.5): z=(3-2)/0.5=2, -0.5*log(2π) - log(0.5) - 0.5*4
    (let ((expected (- (- (* -0.5d0 (log (* 2.0d0 pi)))
                          (log 0.5d0))
                       2.0d0)))
      (ok (approx= (dist:normal-log-pdf 3.0d0 :mu 2.0d0 :sigma 0.5d0)
                   expected 1d-10)))))

(deftest test-normal-log-pdf-ad-forward
  (testing "normal log-pdf differentiable via forward-mode"
    ;; d/dx log N(x|0,1) at x=1 = -(x-mu)/sigma^2 = -1.0
    (multiple-value-bind (val deriv)
        (ad:derivative (lambda (x) (dist:normal-log-pdf x)) 1.0d0)
      (ok (approx= val -1.4189385332046727d0 1d-10))
      (ok (approx= deriv -1.0d0 1d-10)))))

(deftest test-normal-log-pdf-ad-reverse
  (testing "normal log-pdf differentiable via reverse-mode (gradient w.r.t. mu)"
    ;; d/dmu log N(1|mu,1) at mu=0 = (x-mu)/sigma^2 = 1.0
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:normal-log-pdf 1.0d0 :mu (first p) :sigma 1.0d0))
                     '(0.0d0))
      (ok (approx= val -1.4189385332046727d0 1d-10))
      (ok (approx= (first grad) 1.0d0 1d-10)))))

(deftest test-normal-sample-range
  (testing "normal samples have reasonable mean"
    (let* ((n 10000)
           (samples (loop repeat n collect (dist:normal-sample :mu 5.0d0 :sigma 0.1d0)))
           (mean (/ (reduce #'+ samples) n)))
      (ok (approx= mean 5.0d0 0.05d0)))))
```

**Step 2: Implement `src/distributions/normal.lisp`**

```lisp
(in-package #:cl-acorn.distributions)

(defun normal-log-pdf (x &key (mu 0.0d0) (sigma 1.0d0))
  "Log probability density of Normal(mu, sigma).
X, MU, and SIGMA may be AD values (dual or tape-node).
Returns log N(x | mu, sigma) = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²."
  (let ((z (ad:/ (ad:- x mu) sigma)))
    (ad:- (ad:* -0.5d0 (ad:* z z))
          (ad:log sigma)
          +log-2pi/2+)))

(defun normal-sample (&key (mu 0.0d0) (sigma 1.0d0))
  "Sample from Normal(mu, sigma) using Box-Muller transform.
Returns a double-float."
  (let* ((mu (coerce mu 'double-float))
         (sigma (coerce sigma 'double-float))
         (u1 (max double-float-epsilon (random 1.0d0)))
         (u2 (random 1.0d0))
         (z (* (sqrt (* -2.0d0 (log u1)))
               (cos (* 2.0d0 pi u2)))))
    (+ mu (* sigma z))))
```

**Step 3: Run tests**

Expected: 114 pass (109 + 5 new).

**Step 4: Commit**

```
feat(distributions): add normal distribution (log-pdf + sample)
```

---

### Task 4: Uniform and Bernoulli Distributions

Two simple distributions with minimal math.

**Files:**
- Modify: `src/distributions/uniform.lisp`
- Modify: `src/distributions/bernoulli.lisp`
- Modify: `tests/distributions-test.lisp`

**Step 1: Write tests**

```lisp
;;; --- uniform distribution tests ---

(deftest test-uniform-log-pdf
  (testing "uniform log-pdf at known points"
    ;; Uniform(0,1) at x=0.5: log(1/(1-0)) = 0
    (ok (approx= (dist:uniform-log-pdf 0.5d0) 0.0d0 1d-10))
    ;; Uniform(2,5) at x=3: log(1/(5-2)) = -log(3)
    (ok (approx= (dist:uniform-log-pdf 3.0d0 :low 2.0d0 :high 5.0d0)
                 (- (log 3.0d0)) 1d-10))))

(deftest test-uniform-log-pdf-out-of-bounds
  (testing "uniform log-pdf returns very negative value out of bounds"
    (ok (< (dist:uniform-log-pdf -1.0d0 :low 0.0d0 :high 1.0d0) -1d100))))

(deftest test-uniform-sample-range
  (testing "uniform samples stay within bounds"
    (let ((lo 2.0d0) (hi 5.0d0))
      (loop repeat 1000
            do (let ((s (dist:uniform-sample :low lo :high hi)))
                 (ok (>= s lo))
                 (ok (<= s hi)))))))

;;; --- bernoulli distribution tests ---

(deftest test-bernoulli-log-pdf
  (testing "bernoulli log-pdf at known points"
    ;; Bernoulli(0.7) at x=1: log(0.7)
    (ok (approx= (dist:bernoulli-log-pdf 1.0d0 :prob 0.7d0)
                 (log 0.7d0) 1d-10))
    ;; Bernoulli(0.7) at x=0: log(0.3)
    (ok (approx= (dist:bernoulli-log-pdf 0.0d0 :prob 0.7d0)
                 (log 0.3d0) 1d-10))))

(deftest test-bernoulli-log-pdf-ad
  (testing "bernoulli log-pdf differentiable w.r.t. prob"
    ;; d/dp log Bernoulli(1|p) at p=0.5 = 1/p = 2.0
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:bernoulli-log-pdf 1.0d0 :prob (first p)))
                     '(0.5d0))
      (ok (approx= val (log 0.5d0) 1d-10))
      (ok (approx= (first grad) 2.0d0 1d-10)))))

(deftest test-bernoulli-sample
  (testing "bernoulli samples have correct mean"
    (let* ((n 10000)
           (samples (loop repeat n collect (dist:bernoulli-sample :prob 0.7d0)))
           (mean (/ (reduce #'+ samples) n)))
      (ok (approx= mean 0.7d0 0.05d0)))))
```

**Step 2: Implement `src/distributions/uniform.lisp`**

```lisp
(in-package #:cl-acorn.distributions)

(defun real-value (x)
  "Extract plain numeric value from X (dual, tape-node, or number)."
  (typecase x
    (ad:dual (ad:dual-real x))
    (ad:tape-node (let ((v (ad:node-value x)))
                    (if (typep v 'ad:dual) (ad:dual-real v) v)))
    (t x)))

(defun uniform-log-pdf (x &key (low 0.0d0) (high 1.0d0))
  "Log probability density of Uniform(low, high).
X may be an AD value. LOW and HIGH may be AD values.
Returns -log(high - low) if low <= x <= high, else most-negative-double-float."
  (let ((x-real (coerce (real-value x) 'double-float))
        (lo-real (coerce (real-value low) 'double-float))
        (hi-real (coerce (real-value high) 'double-float)))
    (if (and (>= x-real lo-real) (<= x-real hi-real))
        (ad:- (ad:log (ad:- high low)))
        most-negative-double-float)))

(defun uniform-sample (&key (low 0.0d0) (high 1.0d0))
  "Sample from Uniform(low, high). Returns a double-float."
  (let ((lo (coerce low 'double-float))
        (hi (coerce high 'double-float)))
    (+ lo (* (random 1.0d0) (- hi lo)))))
```

**Step 3: Implement `src/distributions/bernoulli.lisp`**

```lisp
(in-package #:cl-acorn.distributions)

(defun bernoulli-log-pdf (x &key (prob 0.5d0))
  "Log probability mass of Bernoulli(prob).
X must be 0 or 1 (as a number). PROB may be an AD value.
Returns x*log(p) + (1-x)*log(1-p)."
  (let ((x (coerce x 'double-float)))
    (ad:+ (ad:* x (ad:log prob))
          (ad:* (- 1.0d0 x) (ad:log (ad:- 1.0d0 prob))))))

(defun bernoulli-sample (&key (prob 0.5d0))
  "Sample from Bernoulli(prob). Returns 1.0d0 or 0.0d0."
  (if (< (random 1.0d0) (coerce prob 'double-float)) 1.0d0 0.0d0))
```

**Step 4: Run tests**

Expected: 121 pass (114 + 7 new).

**Step 5: Commit**

```
feat(distributions): add uniform and bernoulli distributions
```

---

### Task 5: Gamma Distribution

More complex distribution with Marsaglia-Tsang sampling algorithm. Gamma sample is required by beta-sample (Task 6).

**Files:**
- Modify: `src/distributions/gamma.lisp`
- Modify: `tests/distributions-test.lisp`

**Step 1: Write tests**

```lisp
;;; --- gamma distribution tests ---

(deftest test-gamma-log-pdf
  (testing "gamma log-pdf at known points"
    ;; Gamma(2, 1) at x=1: (2-1)*log(1) - 1*1 + 2*log(1) - logΓ(2) = -1
    (ok (approx= (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate 1.0d0)
                 -1.0d0 1d-10))
    ;; Gamma(1, 1) at x=1: Exponential(1) at x=1: -1.0
    (ok (approx= (dist:gamma-log-pdf 1.0d0 :shape 1.0d0 :rate 1.0d0)
                 -1.0d0 1d-10))))

(deftest test-gamma-log-pdf-ad
  (testing "gamma log-pdf differentiable w.r.t. rate"
    ;; d/dr [Gamma(2,r) log-pdf at x=1] = 2/r - 1
    ;; At r=1: 2 - 1 = 1.0
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:gamma-log-pdf 1.0d0 :shape 2.0d0 :rate (first p)))
                     '(1.0d0))
      (declare (ignore val))
      (ok (approx= (first grad) 1.0d0 1d-10)))))

(deftest test-gamma-sample-mean
  (testing "gamma samples have correct mean (shape/rate)"
    (let* ((n 10000)
           (k 3.0d0) (r 2.0d0)
           (samples (loop repeat n collect (dist:gamma-sample :shape k :rate r)))
           (mean (/ (reduce #'+ samples) n)))
      ;; Mean = k/r = 1.5
      (ok (approx= mean 1.5d0 0.1d0)))))
```

**Step 2: Implement `src/distributions/gamma.lisp`**

```lisp
(in-package #:cl-acorn.distributions)

(defun gamma-log-pdf (x &key (shape 1.0d0) (rate 1.0d0))
  "Log probability density of Gamma(shape, rate).
SHAPE must be a positive number (not AD-differentiable).
X and RATE may be AD values (dual or tape-node).
Returns (shape-1)*log(x) - rate*x + shape*log(rate) - logΓ(shape)."
  (let ((k (coerce shape 'double-float)))
    (ad:- (ad:+ (ad:* (- k 1.0d0) (ad:log x))
                (ad:* k (ad:log rate)))
          (ad:* rate x)
          (log-gammaln k))))

(defun gamma-sample (&key (shape 1.0d0) (rate 1.0d0))
  "Sample from Gamma(shape, rate) using Marsaglia-Tsang method.
Returns a double-float."
  (let ((k (coerce shape 'double-float))
        (r (coerce rate 'double-float)))
    (if (< k 1.0d0)
        ;; For shape < 1: X = Y * U^(1/shape) where Y ~ Gamma(shape+1, 1)
        (let ((u (max double-float-epsilon (random 1.0d0))))
          (/ (* (gamma-sample :shape (+ k 1.0d0) :rate 1.0d0)
                (expt u (/ 1.0d0 k)))
             r))
        ;; Marsaglia & Tsang for shape >= 1
        (let* ((d (- k (/ 1.0d0 3.0d0)))
               (c (/ 1.0d0 (sqrt (* 9.0d0 d)))))
          (/ (loop
               (let* ((x (normal-sample))
                      (v (+ 1.0d0 (* c x))))
                 (when (> v 0.0d0)
                   (let* ((v (* v v v))
                          (u (random 1.0d0)))
                     (when (or (< u (- 1.0d0 (* 0.0331d0 x x x x)))
                               (< (log u) (+ (* 0.5d0 x x)
                                             (* d (+ (- 1.0d0 v) (log v))))))
                       (return (* d v)))))))
             r)))))
```

**Step 3: Run tests**

Expected: 124 pass (121 + 3 new).

**Step 4: Commit**

```
feat(distributions): add gamma distribution (log-pdf + Marsaglia-Tsang sampling)
```

---

### Task 6: Beta and Poisson Distributions

Beta uses gamma-sample internally. Poisson uses log-gammaln for normalization.

**Files:**
- Modify: `src/distributions/beta.lisp`
- Modify: `src/distributions/poisson.lisp`
- Modify: `tests/distributions-test.lisp`

**Step 1: Write tests**

```lisp
;;; --- beta distribution tests ---

(deftest test-beta-log-pdf
  (testing "beta log-pdf at known points"
    ;; Beta(2,3) at x=0.5: pdf = 1.5, log(1.5) ≈ 0.4055
    (let ((expected (log 1.5d0)))
      (ok (approx= (dist:beta-log-pdf 0.5d0 :alpha 2.0d0 :beta 3.0d0)
                   expected 1d-8)))
    ;; Beta(1,1) = Uniform(0,1): log-pdf = 0
    (ok (approx= (dist:beta-log-pdf 0.5d0 :alpha 1.0d0 :beta 1.0d0)
                 0.0d0 1d-10))))

(deftest test-beta-sample-mean
  (testing "beta samples have correct mean (alpha/(alpha+beta))"
    (let* ((n 10000)
           (a 2.0d0) (b 3.0d0)
           (samples (loop repeat n collect (dist:beta-sample :alpha a :beta b)))
           (mean (/ (reduce #'+ samples) n)))
      ;; Mean = 2/5 = 0.4
      (ok (approx= mean 0.4d0 0.05d0)))))

;;; --- poisson distribution tests ---

(deftest test-poisson-log-pdf
  (testing "poisson log-pdf at known points"
    ;; Poisson(5) at k=3: log(e^(-5) * 5^3 / 3!) = log(0.1404) ≈ -1.9635
    (let ((expected (+ (* 3.0d0 (log 5.0d0)) (- 5.0d0) (- (dist:log-gammaln 4.0d0)))))
      (ok (approx= (dist:poisson-log-pdf 3 :rate 5.0d0)
                   expected 1d-10)))
    ;; Poisson(1) at k=0: log(e^(-1)) = -1.0
    (ok (approx= (dist:poisson-log-pdf 0 :rate 1.0d0)
                 -1.0d0 1d-10))))

(deftest test-poisson-log-pdf-ad
  (testing "poisson log-pdf differentiable w.r.t. rate"
    ;; d/dλ [Poisson(λ) log-pdf at k=3] = k/λ - 1
    ;; At λ=5: 3/5 - 1 = -0.4
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (dist:poisson-log-pdf 3 :rate (first p)))
                     '(5.0d0))
      (declare (ignore val))
      (ok (approx= (first grad) -0.4d0 1d-10)))))

(deftest test-poisson-sample-mean
  (testing "poisson samples have correct mean"
    (let* ((n 10000)
           (rate 4.0d0)
           (samples (loop repeat n collect (dist:poisson-sample :rate rate)))
           (mean (/ (reduce #'+ samples) n)))
      (ok (approx= mean rate 0.2d0)))))
```

**Step 2: Implement `src/distributions/beta.lisp`**

```lisp
(in-package #:cl-acorn.distributions)

(defun beta-log-pdf (x &key (alpha 1.0d0) (beta 1.0d0))
  "Log probability density of Beta(alpha, beta).
ALPHA and BETA must be positive numbers (not AD-differentiable).
X may be an AD value (dual or tape-node).
Returns (α-1)*log(x) + (β-1)*log(1-x) - logB(α,β)."
  (let* ((a (coerce alpha 'double-float))
         (b (coerce beta 'double-float))
         (log-beta (- (+ (log-gammaln a) (log-gammaln b))
                      (log-gammaln (+ a b)))))
    (ad:- (ad:+ (ad:* (- a 1.0d0) (ad:log x))
                (ad:* (- b 1.0d0) (ad:log (ad:- 1.0d0 x))))
          log-beta)))

(defun beta-sample (&key (alpha 1.0d0) (beta 1.0d0))
  "Sample from Beta(alpha, beta) using ratio of gamma samples.
Returns a double-float in (0, 1)."
  (let ((x (gamma-sample :shape alpha :rate 1.0d0))
        (y (gamma-sample :shape beta :rate 1.0d0)))
    (/ x (+ x y))))
```

**Step 3: Implement `src/distributions/poisson.lisp`**

```lisp
(in-package #:cl-acorn.distributions)

(defun poisson-log-pdf (k &key (rate 1.0d0))
  "Log probability mass of Poisson(rate).
K must be a non-negative integer. RATE may be an AD value.
Returns k*log(λ) - λ - logΓ(k+1)."
  (let ((k-float (coerce k 'double-float)))
    (ad:- (ad:* k-float (ad:log rate))
          rate
          (log-gammaln (+ k-float 1.0d0)))))

(defun poisson-sample (&key (rate 1.0d0))
  "Sample from Poisson(rate) using Knuth's algorithm.
Returns a double-float (integer value)."
  (let ((l (exp (- (coerce rate 'double-float))))
        (k 0)
        (p 1.0d0))
    (loop
      (incf k)
      (setf p (* p (random 1.0d0)))
      (when (<= p l)
        (return (coerce (1- k) 'double-float))))))
```

**Step 4: Run tests**

Expected: 130 pass (124 + 6 new).

**Step 5: Commit**

```
feat(distributions): add beta and poisson distributions
```

---

### Task 7: SGD Optimizer

Simple stateless optimizer. Replaces hand-written SGD loops in examples.

**Files:**
- Modify: `src/optimizers/sgd.lisp`
- Modify: `tests/optimizers-test.lisp`

**Step 1: Write tests**

```lisp
(defpackage #:cl-acorn/tests/optimizers-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/optimizers-test)

;;; --- SGD tests ---

(deftest test-sgd-step-basic
  (testing "SGD step updates params correctly: p - lr*g"
    (let ((result (opt:sgd-step '(1.0d0 2.0d0) '(0.5d0 -1.0d0) :lr 0.1d0)))
      ;; p1 = 1.0 - 0.1*0.5 = 0.95
      ;; p2 = 2.0 - 0.1*(-1.0) = 2.1
      (ok (approx= (first result) 0.95d0))
      (ok (approx= (second result) 2.1d0)))))

(deftest test-sgd-converges-quadratic
  (testing "SGD converges to minimum of f(x) = (x-3)^2"
    ;; grad f(x) = 2*(x-3)
    (let ((params '(0.0d0)))
      (dotimes (i 100)
        (let* ((x (first params))
               (grad (list (* 2.0d0 (- x 3.0d0)))))
          (setf params (opt:sgd-step params grad :lr 0.01d0))))
      (ok (approx= (first params) 3.0d0 0.1d0)))))
```

**Step 2: Implement `src/optimizers/sgd.lisp`**

```lisp
(in-package #:cl-acorn.optimizers)

(defun sgd-step (params grads &key (lr 0.01d0))
  "Stochastic gradient descent step: p_i <- p_i - lr * g_i.
PARAMS and GRADS are lists of numbers of equal length.
Returns a new list of updated parameters (no side effects)."
  (mapcar (lambda (p g) (- p (* lr g))) params grads))
```

**Step 3: Run tests**

Expected: 132 pass (130 + 2 new).

**Step 4: Commit**

```
feat(optimizers): add SGD optimizer
```

---

### Task 8: Adam Optimizer

Stateful optimizer with momentum and adaptive learning rate.

**Files:**
- Modify: `src/optimizers/adam.lisp`
- Modify: `tests/optimizers-test.lisp`

**Step 1: Write tests**

Add to `tests/optimizers-test.lisp`:

```lisp
;;; --- Adam tests ---

(deftest test-adam-state-initialization
  (testing "make-adam-state creates zero-initialized state"
    (let ((state (opt:make-adam-state 3)))
      (ok (= (length (opt:adam-state-m state)) 3))
      (ok (= (length (opt:adam-state-v state)) 3))
      (ok (every (lambda (x) (= x 0.0d0)) (opt:adam-state-m state)))
      (ok (every (lambda (x) (= x 0.0d0)) (opt:adam-state-v state)))
      (ok (= (opt:adam-state-step state) 0)))))

(deftest test-adam-step-basic
  (testing "Adam step returns updated params and increments step"
    (let ((state (opt:make-adam-state 2))
          (params '(1.0d0 2.0d0))
          (grads '(0.5d0 -1.0d0)))
      (let ((result (opt:adam-step params grads state)))
        (ok (= (length result) 2))
        (ok (= (opt:adam-state-step state) 1))
        ;; After one step, params should have changed
        (ok (not (approx= (first result) 1.0d0 1d-15)))))))

(deftest test-adam-converges-quadratic
  (testing "Adam converges to minimum of f(x,y) = (x-1)^2 + (y-2)^2"
    (let ((params '(0.0d0 0.0d0))
          (state (opt:make-adam-state 2)))
      (dotimes (i 1000)
        (let* ((x (first params))
               (y (second params))
               (grads (list (* 2.0d0 (- x 1.0d0))
                            (* 2.0d0 (- y 2.0d0)))))
          (setf params (opt:adam-step params grads state :lr 0.01d0))))
      (ok (approx= (first params) 1.0d0 0.05d0))
      (ok (approx= (second params) 2.0d0 0.05d0)))))
```

**Step 2: Implement `src/optimizers/adam.lisp`**

```lisp
(in-package #:cl-acorn.optimizers)

(defstruct (adam-state (:constructor %make-adam-state))
  "State for the Adam optimizer.
M holds first moment estimates, V holds second moment estimates,
STEP tracks the number of update steps taken."
  (m nil :type list)
  (v nil :type list)
  (step 0 :type fixnum))

(defun make-adam-state (n-params)
  "Create an Adam optimizer state for N-PARAMS parameters.
Initializes moment estimates to zero."
  (%make-adam-state
   :m (make-list n-params :initial-element 0.0d0)
   :v (make-list n-params :initial-element 0.0d0)
   :step 0))

(defun adam-step (params grads state
                  &key (lr 0.001d0) (beta1 0.9d0) (beta2 0.999d0) (epsilon 1d-8))
  "Adam optimizer step. Updates STATE in-place (m, v, step counter).
Returns a new list of updated parameters.
PARAMS and GRADS are lists of numbers of equal length."
  (incf (adam-state-step state))
  (let ((t-step (adam-state-step state)))
    ;; Update biased first moment estimate: m <- beta1*m + (1-beta1)*g
    (setf (adam-state-m state)
          (mapcar (lambda (m g) (+ (* beta1 m) (* (- 1.0d0 beta1) g)))
                  (adam-state-m state) grads))
    ;; Update biased second moment estimate: v <- beta2*v + (1-beta2)*g^2
    (setf (adam-state-v state)
          (mapcar (lambda (v g) (+ (* beta2 v) (* (- 1.0d0 beta2) (* g g))))
                  (adam-state-v state) grads))
    ;; Bias correction
    (let ((bc1 (- 1.0d0 (expt beta1 t-step)))
          (bc2 (- 1.0d0 (expt beta2 t-step))))
      ;; Update params: p <- p - lr * m_hat / (sqrt(v_hat) + epsilon)
      (mapcar (lambda (p m v)
                (let ((m-hat (/ m bc1))
                      (v-hat (/ v bc2)))
                  (- p (/ (* lr m-hat) (+ (sqrt v-hat) epsilon)))))
              params (adam-state-m state) (adam-state-v state)))))
```

**Step 3: Run tests**

Expected: 135 pass (132 + 3 new).

**Step 4: Commit**

```
feat(optimizers): add Adam optimizer with bias-corrected moments
```

---

### Task 9: HMC Inference

Hamiltonian Monte Carlo sampler. The most complex task. Depends on `ad:gradient` and `dist:normal-sample`.

**Files:**
- Modify: `src/inference/hmc.lisp`
- Modify: `tests/hmc-test.lisp`

**Step 1: Write tests**

```lisp
(defpackage #:cl-acorn/tests/hmc-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/hmc-test)

(deftest test-hmc-leapfrog-energy-conservation
  (testing "leapfrog approximately conserves Hamiltonian energy"
    ;; Use standard normal: log-pdf(x) = -0.5*x^2 (ignore constant)
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (q '(1.0d0))
           (p-mom '(0.5d0))
           ;; Hamiltonian = -log-pdf(q) + 0.5*p^2 = 0.5*q^2 + 0.5*p^2
           (h-initial (+ (* 0.5d0 1.0d0 1.0d0) (* 0.5d0 0.5d0 0.5d0))))
      (multiple-value-bind (q-new p-new)
          (infer::leapfrog log-pdf q p-mom 0.1d0 20)
        (let ((h-final (+ (* 0.5d0 (first q-new) (first q-new))
                          (* 0.5d0 (first p-new) (first p-new)))))
          ;; Energy should be approximately conserved
          (ok (approx= h-initial h-final 0.01d0)))))))

(deftest test-hmc-standard-normal
  (testing "HMC samples from standard normal have correct mean and variance"
    (let* ((log-pdf (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))
           (n-samples 2000)
           (n-warmup 500))
      (multiple-value-bind (samples accept-rate)
          (infer:hmc log-pdf '(0.0d0)
            :n-samples n-samples :n-warmup n-warmup
            :step-size 0.1d0 :n-leapfrog 20)
        ;; Accept rate should be reasonable (> 50%)
        (ok (> accept-rate 0.5d0))
        ;; Check mean ≈ 0
        (let ((mean (/ (reduce #'+ samples :key #'first) n-samples)))
          (ok (approx= mean 0.0d0 0.15d0)))
        ;; Check variance ≈ 1
        (let* ((mean (/ (reduce #'+ samples :key #'first) n-samples))
               (var (/ (reduce #'+ samples
                               :key (lambda (s) (expt (- (first s) mean) 2)))
                       n-samples)))
          (ok (approx= var 1.0d0 0.2d0)))))))

(deftest test-hmc-with-distributions
  (testing "HMC works with dist:normal-log-pdf for a shifted normal"
    ;; Sample from N(3, 1) using dist:normal-log-pdf
    (let ((log-pdf (lambda (p)
                     (dist:normal-log-pdf (first p) :mu 3.0d0 :sigma 1.0d0))))
      (multiple-value-bind (samples accept-rate)
          (infer:hmc log-pdf '(0.0d0)
            :n-samples 2000 :n-warmup 500
            :step-size 0.1d0 :n-leapfrog 20)
        (declare (ignore accept-rate))
        (let ((mean (/ (reduce #'+ samples :key #'first) 2000)))
          (ok (approx= mean 3.0d0 0.2d0)))))))
```

**Step 2: Implement `src/inference/hmc.lisp`**

```lisp
(in-package #:cl-acorn.inference)

(defun compute-kinetic-energy (momentum)
  "Kinetic energy: 0.5 * sum(p_i^2)."
  (* 0.5d0 (reduce #'+ momentum :key (lambda (p) (* p p)))))

(defun leapfrog (log-pdf-fn q p step-size n-steps)
  "Leapfrog integrator for Hamiltonian dynamics.
LOG-PDF-FN accepts a parameter list and returns a scalar.
Q and P are lists of position and momentum values.
Returns (values q-new p-new)."
  (let ((q (copy-list q))
        (p (copy-list p)))
    ;; Half step for momentum
    (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
      (declare (ignore val))
      (setf p (mapcar (lambda (pi gi) (+ pi (* 0.5d0 step-size gi))) p grad)))
    ;; Full steps
    (dotimes (i (1- n-steps))
      (declare (ignore i))
      ;; Full step for position
      (setf q (mapcar (lambda (qi pi) (+ qi (* step-size pi))) q p))
      ;; Full step for momentum
      (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
        (declare (ignore val))
        (setf p (mapcar (lambda (pi gi) (+ pi (* step-size gi))) p grad))))
    ;; Final full step for position
    (setf q (mapcar (lambda (qi pi) (+ qi (* step-size pi))) q p))
    ;; Half step for momentum
    (multiple-value-bind (val grad) (ad:gradient log-pdf-fn q)
      (declare (ignore val))
      (setf p (mapcar (lambda (pi gi) (+ pi (* 0.5d0 step-size gi))) p grad)))
    (values q p)))

(defun hmc (log-pdf-fn initial-params
            &key (n-samples 1000) (n-warmup 500)
                 (step-size 0.01d0) (n-leapfrog 10))
  "Hamiltonian Monte Carlo sampler.
LOG-PDF-FN must accept a list of parameters and return a scalar log-probability
using ad: arithmetic operations (for gradient computation via ad:gradient).
INITIAL-PARAMS is a list of starting parameter values.
Returns (values samples accept-rate) where samples is a list of parameter lists
and accept-rate is the fraction of accepted proposals after warmup."
  (let ((current-q (mapcar (lambda (x) (coerce x 'double-float)) initial-params))
        (n-dim (length initial-params))
        (samples nil)
        (n-accepted 0)
        (total-iterations (+ n-samples n-warmup)))
    (dotimes (iter total-iterations)
      ;; Sample random momentum from N(0, 1)
      (let ((current-p (loop repeat n-dim collect (dist:normal-sample))))
        ;; Current Hamiltonian: H = -log-pdf(q) + 0.5*sum(p^2)
        (let* ((current-log-pdf
                 (coerce (nth-value 0 (ad:gradient log-pdf-fn current-q))
                         'double-float))
               (current-h (- (compute-kinetic-energy current-p) current-log-pdf)))
          ;; Leapfrog integration
          (multiple-value-bind (proposed-q proposed-p)
              (leapfrog log-pdf-fn current-q current-p step-size n-leapfrog)
            ;; Proposed Hamiltonian
            (let* ((proposed-log-pdf
                     (coerce (nth-value 0 (ad:gradient log-pdf-fn proposed-q))
                             'double-float))
                   (proposed-h (- (compute-kinetic-energy proposed-p)
                                  proposed-log-pdf))
                   (log-accept-prob (- current-h proposed-h)))
              ;; Metropolis accept/reject
              (when (or (>= log-accept-prob 0.0d0)
                        (< (log (max double-float-epsilon (random 1.0d0)))
                           log-accept-prob))
                (setf current-q proposed-q)
                (when (>= iter n-warmup)
                  (incf n-accepted)))))))
      ;; Collect sample after warmup
      (when (>= iter n-warmup)
        (push (copy-list current-q) samples)))
    (values (nreverse samples)
            (/ (coerce n-accepted 'double-float)
               (coerce n-samples 'double-float)))))
```

**Step 3: Run tests**

Expected: 138 pass (135 + 3 new).

Note: HMC tests use random sampling and may occasionally be flaky. If a test fails, re-run. Tolerances are set wide (0.15-0.2) to reduce flakiness.

**Step 4: Commit**

```
feat(inference): add HMC sampler with leapfrog integrator
```

---

### Task 10: README Update and Full Regression

Update README.md with new API documentation for distributions, optimizers, and HMC. Run full regression.

**Files:**
- Modify: `README.md`

**Step 1: Run full regression**

Run: `load-system cl-acorn` (with `clear_fasls: true`) + `run-tests cl-acorn/tests`
Expected: All 138 tests pass.

**Step 2: Update README.md**

Add sections after the existing "Differentiation" section:

1. **Probability Distributions** — Table of 6 distributions with API examples
2. **Optimizers** — SGD and Adam with usage examples
3. **HMC Inference** — API and complete Bayesian inference example

Key additions:

```markdown
### Probability Distributions

All symbols are exported from `cl-acorn.distributions` (nickname: `dist`).

| Distribution | Log-PDF | Sample |
|-------------|---------|--------|
| Normal | `(dist:normal-log-pdf x :mu 0 :sigma 1)` | `(dist:normal-sample ...)` |
| Gamma | `(dist:gamma-log-pdf x :shape 2 :rate 1)` | `(dist:gamma-sample ...)` |
| Beta | `(dist:beta-log-pdf x :alpha 2 :beta 3)` | `(dist:beta-sample ...)` |
| Uniform | `(dist:uniform-log-pdf x :low 0 :high 1)` | `(dist:uniform-sample ...)` |
| Bernoulli | `(dist:bernoulli-log-pdf x :prob 0.7)` | `(dist:bernoulli-sample ...)` |
| Poisson | `(dist:poisson-log-pdf k :rate 5)` | `(dist:poisson-sample ...)` |

Log-PDF functions are AD-transparent: parameters accept dual numbers and tape-nodes.

### Optimizers

All symbols are exported from `cl-acorn.optimizers` (nickname: `opt`).

| Optimizer | Usage |
|----------|-------|
| SGD | `(opt:sgd-step params grads :lr 0.01)` |
| Adam | `(opt:adam-step params grads state :lr 0.001)` |

### Bayesian Inference (HMC)

All symbols are exported from `cl-acorn.inference` (nickname: `infer`).

```lisp
;; Infer parameters of a normal distribution from data
(defvar *data* '(2.1 1.8 2.3 1.9 2.0))

(defun model-log-pdf (params)
  (let ((mu (first params))
        (log-sigma (second params)))
    (let ((sigma (ad:exp log-sigma))
          (ll 0.0d0))
      (dolist (x *data*)
        (setf ll (ad:+ ll (dist:normal-log-pdf x :mu mu :sigma sigma))))
      ;; Add priors
      (ad:+ ll
            (dist:normal-log-pdf mu :mu 0.0d0 :sigma 10.0d0)
            (dist:normal-log-pdf log-sigma :mu 0.0d0 :sigma 2.0d0)))))

(infer:hmc #'model-log-pdf '(0.0d0 0.0d0)
  :n-samples 2000 :n-warmup 500
  :step-size 0.01d0 :n-leapfrog 20)
;; => samples, accept-rate
```
```

Update test count in README.

**Step 3: Commit**

```
docs: add distributions, optimizers, HMC inference to README
```

**Step 4: Push**

```bash
git push origin main
```
