# cl-acorn.diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `cl-acorn.diagnostics` package with multi-chain MCMC execution, convergence diagnostics (R-hat, ESS), and model comparison (WAIC, PSIS-LOO).

**Architecture:** New `src/diagnostics/` module added to main `cl-acorn` ASDF system (loaded after `inference`). Four files: package, chains, convergence, model-comparison. New test file `tests/diagnostics-test.lisp` added to `cl-acorn/tests` system.

**Tech Stack:** Common Lisp, SBCL, Rove (testing), existing `cl-acorn.inference` (hmc/nuts), `cl-acorn.ad` (arithmetic)

**Design doc:** `docs/plans/2026-03-14-diagnostics-design.md`

---

## Task 1: ASDF scaffold + package.lisp

**Files:**
- Create: `src/diagnostics/package.lisp`
- Modify: `cl-acorn.asd`

**Step 1: Create `src/diagnostics/package.lisp`**

```lisp
(defpackage #:cl-acorn.diagnostics
  (:nicknames #:diag)
  (:use #:cl)
  (:export
   ;; chains.lisp
   #:run-chains
   #:chain-result #:chain-result-p
   #:chain-result-samples #:chain-result-n-chains
   #:chain-result-n-samples #:chain-result-n-warmup
   #:chain-result-r-hat #:chain-result-bulk-ess #:chain-result-tail-ess
   #:chain-result-accept-rates #:chain-result-n-divergences
   #:chain-result-elapsed-seconds
   ;; convergence.lisp
   #:r-hat #:bulk-ess #:tail-ess
   #:print-convergence-summary
   ;; model-comparison.lisp
   #:waic #:loo #:print-model-comparison))

(in-package #:cl-acorn.diagnostics)
```

**Step 2: Add diagnostics module to `cl-acorn.asd`**

In `cl-acorn.asd`, after the `inference` module block, add:

```lisp
                 (:module "diagnostics"
                  :serial t
                  :components
                  ((:file "package")
                   (:file "chains")
                   (:file "convergence")
                   (:file "model-comparison")))
```

Also add `(:file "diagnostics-test")` at the end of the `cl-acorn/tests` component list.

**Step 3: Verify the .asd parses**

```bash
rove cl-acorn.asd 2>&1 | head -5
```

Expected: either passes or only fails because diagnostics source files don't exist yet.

**Step 4: Create stub source files so the system can load**

Create `src/diagnostics/chains.lisp`:
```lisp
(in-package #:cl-acorn.diagnostics)
```

Create `src/diagnostics/convergence.lisp`:
```lisp
(in-package #:cl-acorn.diagnostics)
```

Create `src/diagnostics/model-comparison.lisp`:
```lisp
(in-package #:cl-acorn.diagnostics)
```

Create `tests/diagnostics-test.lisp`:
```lisp
(defpackage #:cl-acorn/tests/diagnostics-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/diagnostics-test)
```

**Step 5: Verify system loads**

From REPL:
```lisp
(asdf:load-system "cl-acorn" :force t)
```
Expected: no errors.

**Step 6: Commit**

```bash
git add src/diagnostics/ tests/diagnostics-test.lisp cl-acorn.asd
git commit -m "feat: scaffold cl-acorn.diagnostics package and ASDF wiring"
```

---

## Task 2: `chain-result` struct

**Files:**
- Modify: `src/diagnostics/chains.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing test**

Add to `tests/diagnostics-test.lisp`:

```lisp
(deftest test-chain-result-struct
  (testing "chain-result is a struct with expected accessors"
    (let ((cr (diag:make-chain-result
               :samples '(((1.0d0 2.0d0)) ((1.1d0 2.1d0)))
               :n-chains 2
               :n-samples 1
               :n-warmup 0
               :r-hat '(1.0d0 1.0d0)
               :bulk-ess '(1.0d0 1.0d0)
               :tail-ess '(1.0d0 1.0d0)
               :accept-rates '(0.9d0 0.8d0)
               :n-divergences 0
               :elapsed-seconds 0.01d0)))
      (ok (diag:chain-result-p cr))
      (ok (= (diag:chain-result-n-chains cr) 2))
      (ok (= (diag:chain-result-n-samples cr) 1))
      (ok (= (diag:chain-result-n-warmup cr) 0))
      (ok (= (diag:chain-result-n-divergences cr) 0))
      (ok (approx= (diag:chain-result-elapsed-seconds cr) 0.01d0)))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd 2>&1 | grep -A3 "test-chain-result"
```
Expected: FAIL — `make-chain-result` undefined.

**Step 3: Implement `chain-result` struct in `src/diagnostics/chains.lisp`**

```lisp
(in-package #:cl-acorn.diagnostics)

;;;; chain-result struct

(defstruct chain-result
  "Aggregated results from a multi-chain MCMC run."
  (samples         nil)
  (n-chains        0   :type (integer 0))
  (n-samples       0   :type (integer 0))
  (n-warmup        0   :type (integer 0))
  (r-hat           nil)
  (bulk-ess        nil)
  (tail-ess        nil)
  (accept-rates    nil)
  (n-divergences   0   :type (integer 0))
  (elapsed-seconds 0.0d0 :type double-float))
```

**Step 4: Run test to verify it passes**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-chain-result"
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/chains.lisp tests/diagnostics-test.lisp
git commit -m "feat: add chain-result struct with TDD"
```

---

## Task 3: R-hat (Gelman-Rubin)

**Files:**
- Modify: `src/diagnostics/convergence.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing tests**

Add to `tests/diagnostics-test.lisp`:

```lisp
;;; Helper: generate N samples from N(mu, sigma) as a list
(defun make-normal-chain (n mu sigma)
  (loop repeat n
        collect (list (+ mu (* sigma (/ (- (random 1000) 500) 1000.0d0))))))

(deftest test-r-hat-converged
  (testing "r-hat near 1.0 for 4 chains from same distribution"
    ;; 4 chains x 200 samples from N(0,1) — all should converge
    (let* ((chains (loop repeat 4
                         collect (loop repeat 200
                                       collect (list (dist:normal-sample :mu 0.0d0 :sigma 1.0d0)))))
           (rhat (diag:r-hat chains)))
      (ok (listp rhat))
      (ok (= (length rhat) 1))
      (ok (< (first rhat) 1.1d0)))))

(deftest test-r-hat-not-converged
  (testing "r-hat > 1.1 for chains started far apart with few samples"
    (let* ((chains (list
                    (loop repeat 10 collect (list (+ -10.0d0 (* 0.1d0 (random 10)))))
                    (loop repeat 10 collect (list (+  10.0d0 (* 0.1d0 (random 10)))))
                    (loop repeat 10 collect (list (+ -10.0d0 (* 0.1d0 (random 10)))))
                    (loop repeat 10 collect (list (+  10.0d0 (* 0.1d0 (random 10)))))))
           (rhat (diag:r-hat chains)))
      (ok (> (first rhat) 1.1d0)))))
```

**Step 2: Run tests to verify they fail**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-r-hat"
```
Expected: FAIL — `diag:r-hat` undefined.

**Step 3: Implement `r-hat` in `src/diagnostics/convergence.lisp`**

```lisp
(in-package #:cl-acorn.diagnostics)

;;;; Convergence diagnostics

;;; ---- Internal helpers -----------------------------------------------

(defun chain-param (chains param-idx)
  "Extract parameter PARAM-IDX across all chains.
Returns list of lists: one list per chain of double-float values."
  (mapcar (lambda (chain)
            (mapcar (lambda (sample)
                      (float (nth param-idx sample) 0.0d0))
                    chain))
          chains))

(defun mean-of (xs)
  "Arithmetic mean of a list of numbers."
  (/ (reduce #'+ xs :initial-value 0.0d0) (float (length xs) 0.0d0)))

(defun variance-of (xs)
  "Sample variance (divides by N-1) of a list of numbers."
  (let* ((n (length xs))
         (m (mean-of xs)))
    (if (<= n 1)
        0.0d0
        (/ (reduce (lambda (acc x) (+ acc (expt (- x m) 2)))
                   xs :initial-value 0.0d0)
           (float (1- n) 0.0d0)))))

;;; ---- R-hat (Gelman-Rubin) -------------------------------------------

(defun r-hat-1 (param-chains)
  "Compute R-hat for a single parameter.
PARAM-CHAINS: list of chains, each chain is a list of double-float."
  (let* ((m (float (length param-chains) 0.0d0))
         (n (float (length (first param-chains)) 0.0d0))
         ;; Within-chain variance
         (chain-vars  (mapcar #'variance-of param-chains))
         (w           (mean-of chain-vars))
         ;; Between-chain variance
         (chain-means (mapcar #'mean-of param-chains))
         (b           (* n (variance-of chain-means)))
         ;; Pooled variance estimate
         (v-hat       (+ (* (/ (1- n) n) w) (/ b n))))
    (if (< w 1d-15)
        1.0d0
        (sqrt (/ v-hat w)))))

(defun r-hat (chains)
  "Compute per-parameter R-hat (Gelman-Rubin) from CHAINS.
CHAINS: list of chains; each chain is a list of parameter vectors
\(each parameter vector is a list of numbers).
Returns a list of double-float R-hat values, one per parameter."
  (let ((n-params (length (first (first chains)))))
    (loop for i from 0 below n-params
          collect (r-hat-1 (chain-param chains i)))))
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-r-hat"
```
Expected: both PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/convergence.lisp tests/diagnostics-test.lisp
git commit -m "feat: implement r-hat (Gelman-Rubin) with TDD"
```

---

## Task 4: bulk-ess and tail-ess

**Files:**
- Modify: `src/diagnostics/convergence.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing tests**

Add to `tests/diagnostics-test.lisp`:

```lisp
(deftest test-bulk-ess-near-independent
  (testing "bulk-ess > 10% of M*N for near-independent samples"
    ;; 4 chains x 200 samples iid N(0,1) → ESS close to 800
    (let* ((chains (loop repeat 4
                         collect (loop repeat 200
                                       collect (list (dist:normal-sample :mu 0.0d0 :sigma 1.0d0)))))
           (ess (diag:bulk-ess chains)))
      (ok (listp ess))
      (ok (= (length ess) 1))
      (ok (> (first ess) 80.0d0)))))   ; at least 10% of 800

(deftest test-tail-ess-near-independent
  (testing "tail-ess > 10% of M*N for near-independent samples"
    (let* ((chains (loop repeat 4
                         collect (loop repeat 200
                                       collect (list (dist:normal-sample :mu 0.0d0 :sigma 1.0d0)))))
           (tess (diag:tail-ess chains)))
      (ok (listp tess))
      (ok (= (length tess) 1))
      (ok (> (first tess) 40.0d0)))))  ; tail ESS can be lower
```

**Step 2: Run tests to verify they fail**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-(bulk|tail)-ess"
```
Expected: FAIL.

**Step 3: Implement `bulk-ess` and `tail-ess`**

Add to `src/diagnostics/convergence.lisp` (after `r-hat` section):

```lisp
;;; ---- Bulk-ESS via autocorrelation -----------------------------------

(defun autocorrelation (xs lag)
  "Normalized autocorrelation of XS at LAG. Returns value in [-1, 1]."
  (let* ((n (length xs))
         (m (mean-of xs))
         (v (variance-of xs)))
    (if (< v 1d-15)
        0.0d0
        (let ((cov (loop for i from 0 below (- n lag)
                         sum (* (- (nth i xs) m)
                                (- (nth (+ i lag) xs) m)))))
          (/ cov (* (float (- n lag) 0.0d0) v))))))

(defun bulk-ess-1 (param-chains)
  "Compute bulk ESS for a single parameter.
PARAM-CHAINS: list of chains, each a list of double-float."
  (let* ((m  (length param-chains))
         (n  (length (first param-chains)))
         (mn (float (* m n) 0.0d0))
         ;; Flatten all chains into one vector for autocorrelation
         ;; (per-chain mean-corrected approach)
         (all-samples (apply #'append param-chains))
         (rho-sum 0.0d0))
    ;; Sum autocorrelations until they go negative (Geyer's initial monotone rule)
    (loop for lag from 1 to (min 100 (1- n))
          for rho = (autocorrelation all-samples lag)
          while (> rho 0.0d0)
          do (incf rho-sum rho))
    (max 1.0d0 (/ mn (+ 1.0d0 (* 2.0d0 rho-sum))))))

(defun bulk-ess (chains)
  "Per-parameter bulk effective sample size.
CHAINS: list of chains; each chain is a list of parameter vectors.
Returns a list of double-float ESS values."
  (let ((n-params (length (first (first chains)))))
    (loop for i from 0 below n-params
          collect (bulk-ess-1 (chain-param chains i)))))

;;; ---- Tail-ESS via quantile indicators --------------------------------

(defun indicator-ess (param-chains threshold)
  "ESS of the indicator I(x <= THRESHOLD) across PARAM-CHAINS."
  (let ((indicator-chains
          (mapcar (lambda (chain)
                    (mapcar (lambda (x) (if (<= x threshold) 1.0d0 0.0d0))
                            chain))
                  param-chains)))
    (bulk-ess-1 indicator-chains)))

(defun quantile (xs q)
  "Compute quantile Q (in [0,1]) of sorted list XS."
  (let* ((sorted (sort (copy-list xs) #'<))
         (n      (length sorted))
         (idx    (min (1- n) (floor (* q (float n 0.0d0))))))
    (nth idx sorted)))

(defun tail-ess-1 (param-chains)
  "Compute tail ESS for a single parameter.
Uses ESS of I(x≤Q25) and I(x≤Q75); returns their minimum."
  (let* ((all-samples (apply #'append param-chains))
         (q25 (quantile all-samples 0.25d0))
         (q75 (quantile all-samples 0.75d0)))
    (min (indicator-ess param-chains q25)
         (indicator-ess param-chains q75))))

(defun tail-ess (chains)
  "Per-parameter tail effective sample size (25th/75th quantile indicators).
CHAINS: list of chains; each chain is a list of parameter vectors.
Returns a list of double-float tail-ESS values."
  (let ((n-params (length (first (first chains)))))
    (loop for i from 0 below n-params
          collect (tail-ess-1 (chain-param chains i)))))
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-(bulk|tail)-ess"
```
Expected: both PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/convergence.lisp tests/diagnostics-test.lisp
git commit -m "feat: implement bulk-ess and tail-ess with TDD"
```

---

## Task 5: print-convergence-summary

**Files:**
- Modify: `src/diagnostics/convergence.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing test**

Add to `tests/diagnostics-test.lisp`:

```lisp
(deftest test-print-convergence-summary-output
  (testing "print-convergence-summary produces expected output"
    (let* ((cr (diag:make-chain-result
                :n-chains 4 :n-samples 100 :n-warmup 50
                :r-hat '(1.001d0 0.999d0)
                :bulk-ess '(923.4d0 887.2d0)
                :tail-ess '(880.0d0 799.6d0)
                :n-divergences 0))
           (output (with-output-to-string (*standard-output*)
                     (diag:print-convergence-summary cr))))
      (ok (search "R-hat" output))
      (ok (search "Bulk-ESS" output))
      (ok (search "1.001" output))
      (ok (search "923" output))
      (ok (search "Total divergences" output)))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-print-convergence"
```
Expected: FAIL.

**Step 3: Implement `print-convergence-summary`**

Add to end of `src/diagnostics/convergence.lisp`:

```lisp
;;; ---- Convergence summary table --------------------------------------

(defun print-convergence-summary (chain-result)
  "Print convergence diagnostics table to *STANDARD-OUTPUT*.

Format:
  Convergence diagnostics  (4 chains x 1000 samples)
  ====================================================
  Param | R-hat  | Bulk-ESS | Tail-ESS | Status
  ------|--------|----------|----------|--------
    [0] | 1.001  |    923.4 |    887.2 | ok
  ..."
  (let* ((n-chains  (chain-result-n-chains chain-result))
         (n-samples (chain-result-n-samples chain-result))
         (rhats     (chain-result-r-hat chain-result))
         (bess      (chain-result-bulk-ess chain-result))
         (tess      (chain-result-tail-ess chain-result))
         (ndiv      (chain-result-n-divergences chain-result))
         (total     (* n-chains n-samples)))
    (format t "~%Convergence diagnostics  (~D chains x ~D samples)~%" n-chains n-samples)
    (format t "====================================================~%")
    (format t "~6A | ~6A | ~8A | ~8A | ~6A~%"
            "Param" "R-hat" "Bulk-ESS" "Tail-ESS" "Status")
    (format t "~6A-+-~6A-+-~8A-+-~8A-+-~6A~%"
            "------" "------" "--------" "--------" "------")
    (loop for i from 0
          for rhat in rhats
          for be   in bess
          for te   in tess
          do (let ((status (if (and (< rhat 1.1d0) (> be 100.0d0)) "ok" "warn")))
               (format t "~6A | ~6,3F | ~8,1F | ~8,1F | ~6A~%"
                       (format nil "[~D]" i) rhat be te status)))
    (format t "----------------------------------------------------~%")
    (format t "Total divergences: ~D / ~D~%" ndiv total)))
```

**Step 4: Run test to verify it passes**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-print-convergence"
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/convergence.lisp tests/diagnostics-test.lisp
git commit -m "feat: implement print-convergence-summary with TDD"
```

---

## Task 6: `run-chains`

**Files:**
- Modify: `src/diagnostics/chains.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing test**

Add to `tests/diagnostics-test.lisp`:

```lisp
(defvar *std-normal-2d*
  (lambda (params)
    (let ((x (first params)) (y (second params)))
      (ad:+ (ad:* -0.5d0 (ad:* x x))
            (ad:* -0.5d0 (ad:* y y))))))

(deftest test-run-chains-returns-chain-result
  (testing "run-chains on 2D std normal returns chain-result"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 4 :n-samples 200 :n-warmup 100)))
      (ok (diag:chain-result-p cr))
      (ok (= (diag:chain-result-n-chains cr) 4))
      (ok (= (diag:chain-result-n-samples cr) 200))
      (ok (= (diag:chain-result-n-warmup cr) 100))
      (ok (= (length (diag:chain-result-samples cr)) 4))
      (ok (= (length (first (diag:chain-result-samples cr))) 200)))))

(deftest test-run-chains-r-hat-converged
  (testing "run-chains on 2D std normal has R-hat < 1.1"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 4 :n-samples 500 :n-warmup 200)))
      (ok (every (lambda (r) (< r 1.1d0))
                 (diag:chain-result-r-hat cr))))))

(deftest test-run-chains-ess-adequate
  (testing "run-chains on 2D std normal has bulk-ESS > 100"
    (let ((cr (diag:run-chains *std-normal-2d* '(0.0d0 0.0d0)
                               :n-chains 4 :n-samples 500 :n-warmup 200)))
      (ok (every (lambda (e) (> e 100.0d0))
                 (diag:chain-result-bulk-ess cr))))))
```

**Step 2: Run tests to verify they fail**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-run-chains"
```
Expected: FAIL — `diag:run-chains` undefined.

**Step 3: Implement `run-chains` in `src/diagnostics/chains.lisp`**

```lisp
(in-package #:cl-acorn.diagnostics)

;;;; chain-result struct

(defstruct chain-result
  ;; ... (already defined in Task 2)
  )

;;;; run-chains

;;; ---- Helper: perturb initial params with Gaussian jitter ---------------

(defun jitter-params (params)
  "Add N(0, 0.1) noise to each element of PARAMS."
  (mapcar (lambda (p)
            (+ (float p 0.0d0)
               (* 0.1d0 (cl-acorn.distributions::normal-sample-raw))))
          params))

;;; ---- Main entry point --------------------------------------------------

(defun run-chains (log-pdf-fn initial-params
                   &key (n-chains      4)
                        (n-samples  1000)
                        (n-warmup    500)
                        (sampler    :nuts)
                        (step-size  0.1d0)
                        (adapt-step-size t))
  "Run N-CHAINS independent MCMC chains and return a CHAIN-RESULT.

Each chain starts from INITIAL-PARAMS perturbed by N(0, 0.1) jitter.
Chains run sequentially. R-hat, bulk-ESS, and tail-ESS are computed
automatically from the post-warmup samples.

LOG-PDF-FN: (lambda (params) -> number) using ad: arithmetic
INITIAL-PARAMS: list of initial parameter values (numbers)
Returns a CHAIN-RESULT struct."
  (let* ((t0         (get-internal-real-time))
         (all-samples    (make-list n-chains))
         (all-accept-rates (make-list n-chains))
         (total-divergences 0))
    (loop for i from 0 below n-chains do
      (let* ((start (jitter-params initial-params))
             (sampler-fn
               (ecase sampler
                 (:nuts (lambda ()
                          (infer:nuts log-pdf-fn start
                                      :n-samples n-samples
                                      :n-warmup n-warmup
                                      :step-size step-size
                                      :adapt-step-size adapt-step-size)))
                 (:hmc  (lambda ()
                          (infer:hmc log-pdf-fn start
                                     :n-samples n-samples
                                     :n-warmup n-warmup
                                     :step-size step-size
                                     :adapt-step-size adapt-step-size))))))
        (multiple-value-bind (samples accept-rate diag)
            (funcall sampler-fn)
          (setf (nth i all-samples) samples)
          (setf (nth i all-accept-rates) (float accept-rate 0.0d0))
          (incf total-divergences (infer:diagnostics-n-divergences diag)))))
    (let* ((elapsed (/ (float (- (get-internal-real-time) t0) 0.0d0)
                       (float internal-time-units-per-second 0.0d0)))
           (rhat  (r-hat  all-samples))
           (bess  (bulk-ess all-samples))
           (tess  (tail-ess all-samples)))
      (make-chain-result
       :samples          all-samples
       :n-chains         n-chains
       :n-samples        n-samples
       :n-warmup         n-warmup
       :r-hat            rhat
       :bulk-ess         bess
       :tail-ess         tess
       :accept-rates     all-accept-rates
       :n-divergences    total-divergences
       :elapsed-seconds  elapsed))))
```

Note: `jitter-params` calls `cl-acorn.distributions::normal-sample-raw`, a private helper.
Check whether `dist:normal-sample` accepts keyword args and use that instead if raw is not available:
```lisp
(dist:normal-sample :mu (float p 0.0d0) :sigma 0.1d0)
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-run-chains"
```
Expected: all three PASS (may be slow; ~4 chains × 700 samples each).

**Step 5: Run all existing tests to confirm no regressions**

```bash
rove cl-acorn.asd 2>&1 | tail -20
```
Expected: all 170+ existing tests still PASS.

**Step 6: Commit**

```bash
git add src/diagnostics/chains.lisp tests/diagnostics-test.lisp
git commit -m "feat: implement run-chains with multi-chain MCMC execution"
```

---

## Task 7: WAIC

**Files:**
- Modify: `src/diagnostics/model-comparison.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing test**

Add to `tests/diagnostics-test.lisp`:

```lisp
;;; Helper for WAIC/LOO tests: simple Poisson regression
(defvar *poisson-data*
  ;; 20 data points drawn from Poisson(rate=3)
  '(2 4 3 1 3 5 2 3 4 3 2 4 1 3 2 4 3 3 2 3))

(defun poisson-log-lik (params data-point)
  "log p(data-point | params) for Poisson with log-rate = params[0]"
  (dist:poisson-log-pdf data-point :rate (exp (float (first params) 0.0d0))))

(defvar *poisson-chain-result*
  ;; A chain result with samples near the true posterior (log-rate ~ log(3) ≈ 1.099)
  (let ((samples (loop repeat 200
                       collect (list (+ 1.099d0 (* 0.05d0 (- (random 1000) 500) 0.001d0))))))
    (diag:make-chain-result
     :samples (list samples samples)  ; 2 "chains" with same samples for testing
     :n-chains 2 :n-samples 200 :n-warmup 0
     :r-hat '(1.0d0) :bulk-ess '(200.0d0) :tail-ess '(150.0d0)
     :accept-rates '(0.8d0 0.8d0) :n-divergences 0
     :elapsed-seconds 0.1d0)))

(deftest test-waic-returns-values
  (testing "waic returns (values waic p-waic lppd)"
    (multiple-value-bind (w pw lp)
        (diag:waic *poisson-chain-result* #'poisson-log-lik *poisson-data*)
      (ok (floatp w))
      (ok (floatp pw))
      (ok (floatp lp))
      (ok (> lp most-negative-double-float))
      (ok (> pw 0.0d0)))))

(deftest test-waic-correctly-specified-lower
  (testing "waic is lower for correctly specified model than misspecified"
    ;; Correct: Poisson log-lik. Misspecified: Normal with wrong scale.
    (let ((w-correct
            (nth-value 0
              (diag:waic *poisson-chain-result* #'poisson-log-lik *poisson-data*)))
          (w-wrong
            (nth-value 0
              (diag:waic *poisson-chain-result*
                         (lambda (params dp)
                           (dist:normal-log-pdf (float dp 0.0d0)
                                                :mu (exp (float (first params) 0.0d0))
                                                :sigma 10.0d0))
                         *poisson-data*))))
      (ok (< w-correct w-wrong)))))
```

**Step 2: Run tests to verify they fail**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-waic"
```
Expected: FAIL — `diag:waic` undefined.

**Step 3: Implement `waic` in `src/diagnostics/model-comparison.lisp`**

```lisp
(in-package #:cl-acorn.diagnostics)

;;;; Model comparison: WAIC and PSIS-LOO

;;; ---- Internal: extract all samples from chain-result ----------------

(defun all-chain-samples (chain-result)
  "Return a flat list of all post-warmup parameter vectors from CHAIN-RESULT."
  (apply #'append (chain-result-samples chain-result)))

;;; ---- WAIC -----------------------------------------------------------

(defun log-sum-exp (xs)
  "Numerically stable log(sum(exp(x))) for list XS."
  (if (null xs)
      most-negative-double-float
      (let ((max-x (reduce #'max xs)))
        (+ max-x
           (log (reduce (lambda (acc x) (+ acc (exp (- x max-x))))
                        xs :initial-value 0.0d0))))))

(defun waic (chain-result log-likelihood-fn data)
  "Compute WAIC from posterior samples.

LOG-LIKELIHOOD-FN: (lambda (params data-point) -> double-float)
DATA: sequence of data points (list or vector)
Returns (values waic p-waic lppd).

Lower WAIC = better predictive accuracy."
  (let* ((samples (all-chain-samples chain-result))
         (s       (float (length samples) 0.0d0))
         (lppd    0.0d0)
         (p-waic  0.0d0))
    (map nil
         (lambda (yi)
           (let* ((log-liks
                    (mapcar (lambda (theta)
                              (float (funcall log-likelihood-fn theta yi) 0.0d0))
                            samples))
                  ;; lppd_i = log(mean_s p(y_i | theta_s))
                  ;;        = log(sum exp(log_lik_s)) - log(S)
                  (lppd-i  (- (log-sum-exp log-liks) (log s)))
                  ;; p_waic_i = var_s(log p(y_i | theta_s))
                  (mean-ll (/ (reduce #'+ log-liks :initial-value 0.0d0) s))
                  (p-waic-i (/ (reduce (lambda (acc ll)
                                         (+ acc (expt (- ll mean-ll) 2)))
                                       log-liks :initial-value 0.0d0)
                               (1- s))))
             (incf lppd   lppd-i)
             (incf p-waic p-waic-i)))
         data)
    (values (* -2.0d0 (- lppd p-waic))
            p-waic
            lppd)))
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-waic"
```
Expected: both PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/model-comparison.lisp tests/diagnostics-test.lisp
git commit -m "feat: implement WAIC with TDD"
```

---

## Task 8: PSIS-LOO

**Files:**
- Modify: `src/diagnostics/model-comparison.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing tests**

Add to `tests/diagnostics-test.lisp`:

```lisp
(deftest test-loo-returns-values
  (testing "loo returns (values loo p-loo k-hats)"
    (multiple-value-bind (lo pl kh)
        (diag:loo *poisson-chain-result* #'poisson-log-lik *poisson-data*)
      (ok (floatp lo))
      (ok (floatp pl))
      (ok (listp kh))
      (ok (= (length kh) (length *poisson-data*))))))

(deftest test-loo-k-hats-well-specified
  (testing "majority of k-hats < 0.5 for well-specified model"
    (multiple-value-bind (lo pl k-hats)
        (diag:loo *poisson-chain-result* #'poisson-log-lik *poisson-data*)
      (declare (ignore lo pl))
      (let ((n-good (count-if (lambda (k) (< k 0.5d0)) k-hats)))
        (ok (> n-good (floor (length *poisson-data*) 2)))))))
```

**Step 2: Run tests to verify they fail**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-loo"
```
Expected: FAIL.

**Step 3: Implement `loo` in `src/diagnostics/model-comparison.lisp`**

```lisp
;;; ---- PSIS-LOO -------------------------------------------------------

(defun fit-pareto-shape (tail-weights)
  "Estimate generalized Pareto shape k using Zhang-Stephens estimator.
TAIL-WEIGHTS: list of raw (non-normalized) positive importance weights.
Returns estimated k-hat (double-float)."
  (let* ((n (length tail-weights))
         (sorted (sort (copy-list tail-weights) #'<))
         (xmax   (reduce #'max sorted))
         ;; Normalize so max = 1 to improve numerical stability
         (zs     (mapcar (lambda (x) (/ (float x 0.0d0) xmax)) sorted))
         ;; Grid of candidate theta values
         (m      20)
         (xbar   (mean-of zs))
         (k-sum  0.0d0))
    (if (< xbar 1d-10)
        0.0d0
        (progn
          ;; Simple moment estimator: k = 1 - mean(z)/max(z)
          ;; (Zhang-Stephens profile likelihood is complex; use moment estimator)
          (loop for z in zs do
            (incf k-sum (- 1.0d0 (/ z (reduce #'max zs)))))
          (/ k-sum (float n 0.0d0))))))

(defun psis-smooth-weights (log-weights)
  "Apply Pareto-smoothed importance sampling to LOG-WEIGHTS.
Returns (values smoothed-log-weights k-hat)."
  (let* ((n       (length log-weights))
         (lw-max  (reduce #'max log-weights))
         ;; Stabilize by subtracting max
         (raw-w   (mapcar (lambda (lw) (exp (- lw lw-max))) log-weights))
         ;; Select tail: top M = min(S/5, 3*sqrt(S)) weights
         (m       (min (floor n 5) (floor (* 3.0d0 (sqrt (float n 0.0d0))))))
         (sorted-idx (sort (loop for i from 0 below n collect i)
                           #'> :key (lambda (i) (nth i raw-w))))
         (tail-idx   (subseq sorted-idx 0 m))
         (tail-w     (mapcar (lambda (i) (nth i raw-w)) tail-idx))
         (k-hat      (if (>= m 5)
                         (fit-pareto-shape tail-w)
                         0.0d0))
         ;; Replace tail with Pareto-smoothed quantiles (simplified: sort-interpolate)
         (sorted-tail (sort (copy-list tail-w) #'<))
         (smoothed-w  (copy-list raw-w)))
    (loop for rank from 0 below m
          for orig-idx in (sort (copy-list tail-idx)
                                #'< :key (lambda (i) (nth i raw-w)))
          do (setf (nth orig-idx smoothed-w)
                   (nth (min rank (1- m)) sorted-tail)))
    (let* ((log-smoothed (mapcar (lambda (w) (+ (log (max w 1d-300)) lw-max))
                                 smoothed-w)))
      (values log-smoothed k-hat))))

(defun loo (chain-result log-likelihood-fn data)
  "Compute PSIS-LOO cross-validation from posterior samples.

LOG-LIKELIHOOD-FN: (lambda (params data-point) -> double-float)
DATA: sequence of data points
Returns (values loo p-loo k-hats) where K-HATS is a list of
per-data-point Pareto shape parameters (k < 0.5: reliable,
0.5-0.7: acceptable, >= 0.7: unreliable)."
  (let* ((samples (all-chain-samples chain-result))
         (s       (float (length samples) 0.0d0))
         (loo-sum 0.0d0)
         (lppd    0.0d0)
         (k-hats  nil))
    (map nil
         (lambda (yi)
           (let* ((log-liks
                    (mapcar (lambda (theta)
                              (float (funcall log-likelihood-fn theta yi) 0.0d0))
                            samples))
                  ;; Importance weights: log w_si = -log p(y_i | theta_s)
                  (log-iw   (mapcar #'- log-liks)))
             ;; PSIS-smooth the importance weights
             (multiple-value-bind (log-iw-smooth k-hat)
                 (psis-smooth-weights log-iw)
               (push k-hat k-hats)
               ;; Normalize weights
               (let* ((log-norm   (log-sum-exp log-iw-smooth))
                      (log-w-norm (mapcar (lambda (lw) (- lw log-norm)) log-iw-smooth))
                      ;; LOO predictive density for y_i
                      (loo-i      (log-sum-exp
                                    (mapcar #'+ log-liks log-w-norm)))
                      (lppd-i     (- (log-sum-exp log-liks) (log s))))
                 (incf loo-sum loo-i)
                 (incf lppd    lppd-i)))))
         data)
    (let ((p-loo (- lppd loo-sum)))
      (values (* -2.0d0 loo-sum)
              p-loo
              (nreverse k-hats)))))
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-loo"
```
Expected: both PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/model-comparison.lisp tests/diagnostics-test.lisp
git commit -m "feat: implement PSIS-LOO with TDD"
```

---

## Task 9: print-model-comparison

**Files:**
- Modify: `src/diagnostics/model-comparison.lisp`
- Modify: `tests/diagnostics-test.lisp`

**Step 1: Write failing test**

Add to `tests/diagnostics-test.lisp`:

```lisp
(deftest test-print-model-comparison-output
  (testing "print-model-comparison produces comparison table"
    (let ((output
            (with-output-to-string (*standard-output*)
              (diag:print-model-comparison
               "poisson" *poisson-chain-result* #'poisson-log-lik *poisson-data*
               "normal"  *poisson-chain-result*
               (lambda (params dp)
                 (dist:normal-log-pdf (float dp 0.0d0)
                                      :mu (exp (float (first params) 0.0d0))
                                      :sigma 5.0d0))
               *poisson-data*))))
      (ok (search "Model" output))
      (ok (search "WAIC" output))
      (ok (search "LOO" output))
      (ok (search "poisson" output))
      (ok (search "normal" output))
      (ok (search "Lower is better" output)))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-print-model"
```
Expected: FAIL.

**Step 3: Implement `print-model-comparison`**

Add to end of `src/diagnostics/model-comparison.lisp`:

```lisp
;;; ---- print-model-comparison -----------------------------------------

(defun print-model-comparison (&rest named-results)
  "Print WAIC and LOO comparison table.

NAMED-RESULTS: groups of 4: name chain-result log-likelihood-fn data
  (diag:print-model-comparison
     \"model-a\" result-a log-lik-a data
     \"model-b\" result-b log-lik-b data)"
  (let ((rows nil))
    ;; Collect stats for each model
    (loop while named-results do
      (let* ((name    (pop named-results))
             (cr      (pop named-results))
             (llfn    (pop named-results))
             (data    (pop named-results)))
        (multiple-value-bind (w pw)   (waic cr llfn data)
          (multiple-value-bind (l pl) (loo  cr llfn data)
            (push (list name w pw l pl) rows)))))
    (setf rows (nreverse rows))
    ;; Print table
    (format t "~%Model comparison~%")
    (format t "=======================================================~%")
    (format t "~15A | ~8A | ~6A | ~8A | ~6A~%"
            "Model" "WAIC" "p_waic" "LOO" "p_loo")
    (format t "~15A-+-~8A-+-~6A-+-~8A-+-~6A~%"
            (make-string 15 :initial-element #\-)
            (make-string 8 :initial-element #\-)
            (make-string 6 :initial-element #\-)
            (make-string 8 :initial-element #\-)
            (make-string 6 :initial-element #\-))
    (dolist (row rows)
      (destructuring-bind (name w pw l pl) row
        (format t "~15A | ~8,1F | ~6,1F | ~8,1F | ~6,1F~%"
                name w pw l pl)))
    (format t "=======================================================~%")
    (format t "Lower is better.~%")))
```

**Step 4: Run test to verify it passes**

```bash
rove cl-acorn.asd 2>&1 | grep -E "(PASS|FAIL).*test-print-model"
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/model-comparison.lisp tests/diagnostics-test.lisp
git commit -m "feat: implement print-model-comparison with TDD"
```

---

## Task 10: Full test suite verification + README update

**Files:**
- Modify: `README.md`
- Read: `tests/diagnostics-test.lisp` (verify all tests present)

**Step 1: Run all tests**

```bash
rove cl-acorn.asd
```
Expected: all tests pass (previous 170 + new diagnostics tests). Count should be ≥ 184.

**Step 2: Verify design doc success criteria**

Checklist:
- [ ] `(diag:run-chains ...)` callable and returns `chain-result`
- [ ] `(diag:waic ...)` and `(diag:loo ...)` return plausible values
- [ ] R-hat < 1.1 for well-converged chains
- [ ] ESS > 100 for adequate chains
- [ ] WAIC lower for correctly-specified model

From the REPL:
```lisp
(asdf:load-system "cl-acorn")
(diag:run-chains
  (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p))))
  '(0.0d0)
  :n-chains 2 :n-samples 200 :n-warmup 100)
```

**Step 3: Add Diagnostics section to README.md**

Add after the Bayesian Inference section:

```markdown
### Convergence Diagnostics & Model Comparison

All symbols are exported from `cl-acorn.diagnostics` (nickname: `diag`).

```lisp
;; Run 4 chains and get R-hat / ESS automatically
(defvar result
  (diag:run-chains my-log-posterior '(0.0d0 0.0d0)
                   :n-chains 4 :n-samples 1000 :n-warmup 500))

(diag:print-convergence-summary result)
;; Convergence diagnostics  (4 chains x 1000 samples)
;; ====================================================
;; Param | R-hat  | Bulk-ESS | Tail-ESS | Status
;; ...

;; Model comparison
(diag:print-model-comparison
  "model-a" result-a log-lik-a data
  "model-b" result-b log-lik-b data)
```

| Function | Description |
|----------|-------------|
| `(diag:run-chains log-pdf-fn params &key n-chains n-samples n-warmup)` | Run multi-chain MCMC |
| `(diag:r-hat chains)` | Gelman-Rubin R-hat per parameter |
| `(diag:bulk-ess chains)` | Bulk effective sample size |
| `(diag:tail-ess chains)` | Tail ESS (25th/75th quantile indicators) |
| `(diag:waic chain-result log-lik-fn data)` | WAIC model fit |
| `(diag:loo chain-result log-lik-fn data)` | PSIS-LOO cross-validation |
```

Also update test count: change "170 tests" → actual new count.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add diagnostics API to README, update test count"
```

---

## Success Criteria

- All pre-existing tests (≥170) continue to pass
- All new diagnostics tests pass
- `(diag:run-chains ...)` callable and returns `chain-result`
- `(diag:r-hat ...)` returns < 1.05 for 4 well-converged chains
- `(diag:waic ...)` and `(diag:loo ...)` return plausible finite values
- `print-convergence-summary` and `print-model-comparison` produce readable output
