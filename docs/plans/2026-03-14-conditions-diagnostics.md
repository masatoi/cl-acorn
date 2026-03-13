# Conditions & Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add structured error handling (CL conditions/restarts) and post-inference diagnostics
to `cl-acorn.inference`, satisfying concept.md sections 4.2 and 4.4.

**Architecture:** Condition types and the `inference-diagnostics` struct live in a new file
`src/inference/conditions.lisp` (in the `cl-acorn.inference` package, loaded first).
`hmc`, `nuts`, and `vi` gain `restart-case` for initial-params validation and return an
`inference-diagnostics` struct as an extra (backward-compatible) return value.

**Tech Stack:** ANSI Common Lisp, SBCL, Rove (tests), ASDF

---

### Task 1: Scaffold — `src/inference/conditions.lisp`, update `.asd` and `package.lisp`

**Files:**
- Create: `src/inference/conditions.lisp`
- Modify: `src/inference/package.lisp`
- Modify: `cl-acorn.asd`

**Step 1: Create `src/inference/conditions.lisp`**

```lisp
(in-package #:cl-acorn.inference)

;;;; Condition hierarchy and diagnostics struct for cl-acorn.

;;; ---- Base error type -------------------------------------------------

(define-condition acorn-error (error)
  ((message :initarg :message :reader acorn-error-message))
  (:report (lambda (c s)
    (format s "~A" (acorn-error-message c))))
  (:documentation "Base class for all cl-acorn error conditions."))

;;; ---- Model errors (distribution / log-pdf problems) -----------------

(define-condition model-error (acorn-error) ()
  (:documentation "Supertype for errors in model or distribution evaluation."))

(define-condition invalid-parameter-error (model-error)
  ((parameter :initarg :parameter :reader invalid-parameter-error-parameter)
   (value     :initarg :value     :reader invalid-parameter-error-value))
  (:report (lambda (c s)
    (format s "Invalid parameter ~A = ~A: ~A"
            (invalid-parameter-error-parameter c)
            (invalid-parameter-error-value c)
            (acorn-error-message c))))
  (:documentation "Signaled when a distribution parameter has an invalid value."))

(define-condition log-pdf-domain-error (model-error)
  ((distribution :initarg :distribution
                 :reader log-pdf-domain-error-distribution))
  (:report (lambda (c s)
    (format s "~A: ~A"
            (log-pdf-domain-error-distribution c)
            (acorn-error-message c))))
  (:documentation "Signaled when log-pdf is evaluated outside its support."))

;;; ---- Inference errors (sampling / optimization problems) ------------

(define-condition inference-error (acorn-error) ()
  (:documentation "Supertype for errors in inference algorithms."))

(define-condition invalid-initial-params-error (inference-error)
  ((params :initarg :params :reader invalid-initial-params-error-params))
  (:report (lambda (c s)
    (format s "Invalid initial params ~A: ~A"
            (invalid-initial-params-error-params c)
            (acorn-error-message c))))
  (:documentation
   "Signaled when initial parameters produce a non-finite log-probability."))

(define-condition non-finite-gradient-error (inference-error)
  ((params :initarg :params :reader non-finite-gradient-error-params))
  (:report (lambda (c s)
    (format s "Non-finite gradient at params ~A: ~A"
            (non-finite-gradient-error-params c)
            (acorn-error-message c))))
  (:documentation "Signaled when the gradient is non-finite at a parameter set."))

;;; ---- Warnings --------------------------------------------------------

(define-condition high-divergence-warning (warning)
  ((n-divergences :initarg :n-divergences
                  :reader high-divergence-warning-n-divergences)
   (n-samples     :initarg :n-samples
                  :reader high-divergence-warning-n-samples))
  (:report (lambda (c s)
    (format s "High divergence rate: ~A/~A transitions diverged (~,1F%). ~
               Consider reducing step-size or reparameterizing the model."
            (high-divergence-warning-n-divergences c)
            (high-divergence-warning-n-samples c)
            (* 100.0d0
               (/ (float (high-divergence-warning-n-divergences c))
                  (float (max 1 (high-divergence-warning-n-samples c))))))))
  (:documentation
   "Warned when NUTS post-warmup divergence rate exceeds the threshold."))

;;; ---- Post-inference diagnostics struct -------------------------------

(defstruct (inference-diagnostics (:conc-name diagnostics-))
  "Post-inference summary statistics returned by HMC, NUTS, and VI."
  (accept-rate     0.0d0 :type double-float)
  (n-divergences   0     :type (integer 0))
  (final-step-size 0.0d0 :type double-float)
  (n-samples       0     :type (integer 0))
  (n-warmup        0     :type (integer 0))
  (elapsed-seconds 0.0d0 :type double-float))
```

**Step 2: Replace `src/inference/package.lisp` with updated exports**

```lisp
(defpackage #:cl-acorn.inference
  (:nicknames #:infer)
  (:use #:cl)
  (:export
   ;; Inference algorithms
   #:hmc
   #:nuts
   #:vi
   ;; Base condition types
   #:acorn-error
   #:acorn-error-message
   #:model-error
   #:inference-error
   ;; Model error subtypes
   #:invalid-parameter-error
   #:invalid-parameter-error-parameter
   #:invalid-parameter-error-value
   #:log-pdf-domain-error
   #:log-pdf-domain-error-distribution
   ;; Inference error subtypes
   #:invalid-initial-params-error
   #:invalid-initial-params-error-params
   #:non-finite-gradient-error
   #:non-finite-gradient-error-params
   ;; Warnings
   #:high-divergence-warning
   #:high-divergence-warning-n-divergences
   #:high-divergence-warning-n-samples
   ;; Restart names
   #:use-fallback-params
   #:return-empty-samples
   #:continue-with-warnings
   ;; Diagnostics struct + accessors
   #:inference-diagnostics
   #:inference-diagnostics-p
   #:make-inference-diagnostics
   #:diagnostics-accept-rate
   #:diagnostics-n-divergences
   #:diagnostics-final-step-size
   #:diagnostics-n-samples
   #:diagnostics-n-warmup
   #:diagnostics-elapsed-seconds))
```

**Step 3: Update `cl-acorn.asd` — add `conditions` file and two new test files**

In the `"cl-acorn"` system, inside the `"inference"` module, add `(:file "conditions")` as the second component (after `package`, before `dual-avg`):

```lisp
(:module "inference"
 :serial t
 :components
 ((:file "package")
  (:file "conditions")      ; NEW
  (:file "dual-avg")
  (:file "hmc")
  (:file "nuts")
  (:file "vi")))
```

In the `"cl-acorn/tests"` system, append two new test files at the end:

```lisp
(:file "conditions-test")             ; NEW
(:file "inference-diagnostics-test")  ; NEW
```

**Step 4: Load the system to verify it compiles**

```lisp
(asdf:load-system :cl-acorn :force t)
```

Expected: loads without errors or warnings.

**Step 5: Commit**

```bash
git add src/inference/conditions.lisp src/inference/package.lisp cl-acorn.asd
git commit -m "feat: add condition hierarchy and inference-diagnostics struct"
```

---

### Task 2: TDD — condition and diagnostics tests

**Files:**
- Create: `tests/conditions-test.lisp`
- Create: `tests/inference-diagnostics-test.lisp`

**Step 1: Create `tests/conditions-test.lisp`**

```lisp
(defpackage #:cl-acorn/tests/conditions-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/conditions-test)

;;; A log-pdf-fn that always returns +infinity (triggers initial-params error)
(defun always-bad-log-pdf (params)
  (declare (ignore params))
  sb-ext:double-float-positive-infinity)

;;; A log-pdf-fn that is bad for large params but OK near zero
(defun conditional-log-pdf (params)
  (let ((p (first params)))
    (if (> (abs p) 100.0d0)
        sb-ext:double-float-positive-infinity
        (ad:* -0.5d0 (ad:* p p)))))

(deftest test-acorn-error-hierarchy
  (testing "condition hierarchy is correct"
    (ok (subtypep 'infer:model-error 'infer:acorn-error))
    (ok (subtypep 'infer:inference-error 'infer:acorn-error))
    (ok (subtypep 'infer:invalid-initial-params-error 'infer:inference-error))
    (ok (subtypep 'infer:invalid-parameter-error 'infer:model-error))
    (ok (subtypep 'infer:log-pdf-domain-error 'infer:model-error))
    (ok (subtypep 'infer:high-divergence-warning 'warning))))

(deftest test-invalid-initial-params-error-signaled-hmc
  (testing "hmc signals invalid-initial-params-error for non-finite initial log-pdf"
    (let ((errored nil))
      (handler-case
          (infer:hmc #'always-bad-log-pdf '(0.0d0)
                     :n-samples 10 :n-warmup 5)
        (infer:invalid-initial-params-error ()
          (setf errored t)))
      (ok errored))))

(deftest test-invalid-initial-params-error-signaled-nuts
  (testing "nuts signals invalid-initial-params-error for non-finite initial log-pdf"
    (let ((errored nil))
      (handler-case
          (infer:nuts #'always-bad-log-pdf '(0.0d0)
                      :n-samples 10 :n-warmup 5)
        (infer:invalid-initial-params-error ()
          (setf errored t)))
      (ok errored))))

(deftest test-use-fallback-params-restart
  (testing "use-fallback-params restart recovers and returns valid samples"
    (multiple-value-bind (samples ar diag)
        (handler-bind ((infer:invalid-initial-params-error
                         (lambda (c)
                           (declare (ignore c))
                           (invoke-restart 'infer:use-fallback-params '(0.0d0)))))
          (infer:hmc #'conditional-log-pdf '(999.0d0)
                     :n-samples 100 :n-warmup 50))
      (ok (> (length samples) 0))
      (ok (> ar 0.0d0))
      (ok (infer:inference-diagnostics-p diag)))))

(deftest test-return-empty-samples-restart
  (testing "return-empty-samples restart returns empty list without error"
    (multiple-value-bind (samples ar diag)
        (handler-bind ((infer:invalid-initial-params-error
                         (lambda (c)
                           (declare (ignore c))
                           (invoke-restart 'infer:return-empty-samples))))
          (infer:hmc #'always-bad-log-pdf '(0.0d0)
                     :n-samples 10 :n-warmup 5))
      (ok (null samples))
      (ok (= ar 0.0d0))
      (ok (infer:inference-diagnostics-p diag)))))

(deftest test-high-divergence-warning-signaled
  (testing "nuts signals high-divergence-warning for absurdly large step-size"
    (let ((warned nil))
      (handler-bind ((infer:high-divergence-warning
                       (lambda (c)
                         (declare (ignore c))
                         (setf warned t)
                         (invoke-restart 'infer:continue-with-warnings))))
        (infer:nuts (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p))))
                    '(0.0d0)
                    :n-samples 50 :n-warmup 5
                    :step-size 100.0d0 :adapt-step-size nil))
      (ok warned))))

(deftest test-continue-with-warnings-restart
  (testing "continue-with-warnings restart suppresses high-divergence-warning"
    ;; Should complete without unhandled condition
    (ok (handler-bind ((infer:high-divergence-warning
                         (lambda (c)
                           (declare (ignore c))
                           (invoke-restart 'infer:continue-with-warnings))))
          (infer:nuts (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p))))
                      '(0.0d0)
                      :n-samples 50 :n-warmup 5
                      :step-size 100.0d0 :adapt-step-size nil)
          t))))
```

**Step 2: Create `tests/inference-diagnostics-test.lisp`**

```lisp
(defpackage #:cl-acorn/tests/inference-diagnostics-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/inference-diagnostics-test)

(defvar *std-normal*
  (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p)))))

(deftest test-hmc-returns-diagnostics
  (testing "hmc returns inference-diagnostics as 3rd value"
    (multiple-value-bind (samples ar diag)
        (infer:hmc *std-normal* '(0.0d0) :n-samples 100 :n-warmup 50)
      (declare (ignore samples ar))
      (ok (infer:inference-diagnostics-p diag))
      (ok (= (infer:diagnostics-n-samples diag) 100))
      (ok (= (infer:diagnostics-n-warmup diag) 50))
      (ok (>= (infer:diagnostics-accept-rate diag) 0.0d0))
      (ok (<= (infer:diagnostics-accept-rate diag) 1.0d0))
      (ok (= (infer:diagnostics-n-divergences diag) 0))
      (ok (> (infer:diagnostics-elapsed-seconds diag) 0.0d0)))))

(deftest test-hmc-diagnostics-step-size-set
  (testing "hmc diagnostics records final step-size"
    (multiple-value-bind (samples ar diag)
        (infer:hmc *std-normal* '(0.0d0)
                   :n-samples 100 :n-warmup 50
                   :step-size 0.2d0 :adapt-step-size nil)
      (declare (ignore samples ar))
      (ok (> (infer:diagnostics-final-step-size diag) 0.0d0)))))

(deftest test-nuts-returns-diagnostics
  (testing "nuts returns inference-diagnostics as 3rd value"
    (multiple-value-bind (samples ar diag)
        (infer:nuts *std-normal* '(0.0d0) :n-samples 100 :n-warmup 50)
      (declare (ignore samples ar))
      (ok (infer:inference-diagnostics-p diag))
      (ok (= (infer:diagnostics-n-samples diag) 100))
      (ok (= (infer:diagnostics-n-warmup diag) 50))
      (ok (>= (infer:diagnostics-n-divergences diag) 0)))))

(deftest test-nuts-diagnostics-counts-divergences
  (testing "nuts diagnostics counts post-warmup divergences"
    (handler-bind ((infer:high-divergence-warning
                     (lambda (c)
                       (declare (ignore c))
                       (invoke-restart 'infer:continue-with-warnings))))
      (multiple-value-bind (samples ar diag)
          (infer:nuts *std-normal* '(0.0d0)
                      :n-samples 50 :n-warmup 5
                      :step-size 100.0d0 :adapt-step-size nil)
        (declare (ignore samples ar))
        (ok (> (infer:diagnostics-n-divergences diag) 0))))))

(deftest test-vi-returns-diagnostics
  (testing "vi returns inference-diagnostics as 4th value"
    (multiple-value-bind (mu sigma elbo diag)
        (infer:vi *std-normal* 1 :n-iterations 100 :n-elbo-samples 5)
      (declare (ignore mu sigma elbo))
      (ok (infer:inference-diagnostics-p diag))
      (ok (= (infer:diagnostics-n-samples diag) 100))
      (ok (= (infer:diagnostics-n-warmup diag) 0))
      (ok (> (infer:diagnostics-elapsed-seconds diag) 0.0d0)))))

(deftest test-hmc-backward-compatible
  (testing "existing 2-value multiple-value-bind still works"
    (multiple-value-bind (samples ar)
        (infer:hmc *std-normal* '(0.0d0) :n-samples 50 :n-warmup 20)
      (ok (> (length samples) 0))
      (ok (> ar 0.0d0)))))

(deftest test-nuts-backward-compatible
  (testing "existing 2-value multiple-value-bind for nuts still works"
    (multiple-value-bind (samples ar)
        (infer:nuts *std-normal* '(0.0d0) :n-samples 50 :n-warmup 20)
      (ok (> (length samples) 0))
      (ok (>= ar 0.0d0)))))
```

**Step 3: Run tests — expect failures** (hmc/nuts/vi not yet updated)

```bash
rove cl-acorn.asd
```

Expected: `conditions-test` hierarchy/signaling tests may fail; `inference-diagnostics-test` tests fail with "wrong number of values".

**Step 4: Commit test files**

```bash
git add tests/conditions-test.lisp tests/inference-diagnostics-test.lisp
git commit -m "test: add failing tests for conditions and diagnostics (4.2+4.4)"
```

---

### Task 3: Update `src/inference/hmc.lisp` — restart-case + diagnostics

**Files:**
- Modify: `src/inference/hmc.lisp`

The `hmc` function needs three changes:
1. Replace the `assert` initial-log-pdf check with `restart-case`
2. Track `start-time` for elapsed seconds
3. Return `inference-diagnostics` as 3rd value

**Step 1: Apply the patch — replace the let* tail and add validation block**

Find this section near the end of the `let*` binding in `hmc`:

```lisp
         ;; Cache current log-pdf to avoid redundant gradient evaluations
         (current-log-pdf (multiple-value-bind (val grad)
                              (safe-gradient log-pdf-fn current-q)
                            (declare (ignore grad))
                            (assert val nil
                                    "hmc: LOG-PDF-FN returned non-finite value at ~
                                     INITIAL-PARAMS. Ensure initial parameters are ~
                                     in the support of the distribution.")
                            val)))
```

Replace with:

```lisp
         ;; Timing for diagnostics
         (start-time (get-internal-real-time))
         ;; current-log-pdf is set after restart-case validation below
         (current-log-pdf nil))
    ;; Validate initial params; offer restarts on failure
    (block validate
      (loop
        (multiple-value-bind (val grad)
            (safe-gradient log-pdf-fn current-q)
          (declare (ignore grad))
          (when val
            (setf current-log-pdf val)
            (return-from validate))
          (restart-case
            (error 'invalid-initial-params-error
                   :params (copy-list current-q)
                   :message "LOG-PDF-FN returned non-finite value at INITIAL-PARAMS")
            (use-fallback-params (new-params)
              :report "Supply new initial params and retry"
              (setf current-q
                    (mapcar (lambda (x) (coerce x 'double-float)) new-params)))
            (return-empty-samples ()
              :report "Return empty sample list"
              (return-from hmc
                (values '() 0.0d0
                        (make-inference-diagnostics
                         :n-samples 0 :n-warmup n-warmup))))))))
```

Note: the closing `))` of the original `let*` moves up to close before the `(block validate ...)`. The let* now ends with `(current-log-pdf nil))` and the block/loop are in the let* body.

**Step 2: Update the return statement at the end of `hmc`**

Find:
```lisp
    (values (nreverse samples)
            (/ (coerce n-accepted 'double-float)
               (coerce n-samples 'double-float)))))
```

Replace with:
```lisp
    (let ((accept-rate (/ (coerce n-accepted 'double-float)
                          (coerce n-samples 'double-float))))
      (values (nreverse samples)
              accept-rate
              (make-inference-diagnostics
               :accept-rate accept-rate
               :n-divergences 0
               :final-step-size step-size
               :n-samples n-samples
               :n-warmup n-warmup
               :elapsed-seconds (/ (float (- (get-internal-real-time) start-time))
                                   internal-time-units-per-second))))))
```

**Step 3: Run tests**

```bash
rove cl-acorn.asd
```

Expected: `test-hmc-returns-diagnostics`, `test-hmc-diagnostics-step-size-set`,
`test-invalid-initial-params-error-signaled-hmc`, `test-use-fallback-params-restart`,
`test-return-empty-samples-restart`, `test-hmc-backward-compatible` all pass.

**Step 4: Commit**

```bash
git add src/inference/hmc.lisp
git commit -m "feat(hmc): add restart-case for initial params and return diagnostics"
```

---

### Task 4: Update `src/inference/nuts.lisp` — divergence count + warning + diagnostics

**Files:**
- Modify: `src/inference/nuts.lisp`

Four changes:
1. Replace `assert` initial-log-pdf check with `restart-case` (restructure let-binding)
2. Track `start-time` and `n-divergences`
3. Count per-iteration divergences during sampling phase
4. Signal `high-divergence-warning` after loop; return diagnostics

**Step 1: Add `+high-divergence-threshold+` variable near top of file, after `+max-delta-energy+`**

```lisp
(defvar +high-divergence-threshold+ 0.10d0
  "Warn when post-warmup divergence rate exceeds this fraction (default 10%).")
```

**Step 2: Replace the `assert` + inner `multiple-value-bind` block**

Currently `nuts` ends its `let*` with:
```lisp
         (n-accept-total 0))
    ;; Get initial log-pdf and gradient, validate finiteness
    (multiple-value-bind (init-val init-grad)
        (safe-gradient log-pdf-fn current-q)
      (assert init-val nil
              "nuts: LOG-PDF-FN returned non-finite value at INITIAL-PARAMS. ~
               Ensure initial parameters are in the support of the distribution.")
      (let ((current-log-pdf init-val)
            (current-grad init-grad))
        (with-float-traps-masked
        (dotimes (iter total-iterations)
          ...))
        ;; Compute average accept rate over sampling phase
        (let ((final-accept ...))
          (values (nreverse samples) final-accept))))))
```

Replace with:
```lisp
         (n-accept-total 0)
         (n-divergences 0)
         (start-time (get-internal-real-time))
         (current-log-pdf nil)
         (current-grad nil))
    ;; Validate initial params; offer restarts on failure
    (block validate
      (loop
        (multiple-value-bind (val grad)
            (safe-gradient log-pdf-fn current-q)
          (when val
            (setf current-log-pdf val current-grad grad)
            (return-from validate))
          (restart-case
            (error 'invalid-initial-params-error
                   :params (copy-list current-q)
                   :message "LOG-PDF-FN returned non-finite value at INITIAL-PARAMS")
            (use-fallback-params (new-params)
              :report "Supply new initial params and retry"
              (setf current-q
                    (mapcar (lambda (x) (coerce x 'double-float)) new-params)))
            (return-empty-samples ()
              :report "Return empty sample list"
              (return-from nuts
                (values '() 0.0d0
                        (make-inference-diagnostics
                         :n-samples 0 :n-warmup n-warmup))))))))
    (with-float-traps-masked
    (dotimes (iter total-iterations)
      ...EXISTING BODY (unchanged except divergence tracking below)...))
    ;; Warn if divergence rate exceeds threshold
    (when (and (> n-divergences 0)
               (> (/ n-divergences (float n-samples))
                  +high-divergence-threshold+))
      (restart-case
        (warn 'high-divergence-warning
              :n-divergences n-divergences
              :n-samples n-samples)
        (continue-with-warnings ()
          :report "Continue and return results despite high divergences"
          nil)))
    ;; Return
    (let ((final-accept (if (> n-accept-total 0)
                            (/ sum-accept (coerce n-accept-total 'double-float))
                            0.0d0)))
      (values (nreverse samples)
              final-accept
              (make-inference-diagnostics
               :accept-rate final-accept
               :n-divergences n-divergences
               :final-step-size step-size
               :n-samples n-samples
               :n-warmup n-warmup
               :elapsed-seconds (/ (float (- (get-internal-real-time) start-time))
                                   internal-time-units-per-second))))))
```

**Step 3: Add per-iteration divergence tracking inside `dotimes`**

Inside the `dotimes` body, find the `let` that creates `depth` and `keep-going`:

```lisp
            (let ((depth 0)
                  (keep-going t))
              (loop while (and keep-going (< depth max-tree-depth))
                    do (let* ((direction ...)
                              (tree ...))
                         ...
                         (cond
                           ;; Subtree diverged: stop growing
                           ((tree-state-diverging-p tree)
                            (setf keep-going nil))
```

Change to:

```lisp
            (let ((depth 0)
                  (keep-going t)
                  (iter-diverged-p nil))
              (loop while (and keep-going (< depth max-tree-depth))
                    do (let* ((direction ...)
                              (tree ...))
                         ...
                         (cond
                           ;; Subtree diverged: stop growing
                           ((tree-state-diverging-p tree)
                            (setf keep-going nil
                                  iter-diverged-p t))
```

Then after the loop (still inside `dotimes`, after `(incf sum-accept ...)` and `(incf n-accept-total ...)`), add:

```lisp
            ;; Count divergences in sampling phase only
            (when (and iter-diverged-p (>= iter n-warmup))
              (incf n-divergences))
```

**Step 4: Run tests**

```bash
rove cl-acorn.asd
```

Expected: all NUTS-related tests in `conditions-test` and `inference-diagnostics-test` pass.

**Step 5: Commit**

```bash
git add src/inference/nuts.lisp
git commit -m "feat(nuts): add restart-case, divergence count, high-divergence-warning, and diagnostics"
```

---

### Task 5: Update `src/inference/vi.lisp` — replace asserts + return diagnostics

**Files:**
- Modify: `src/inference/vi.lisp`

Two changes:
1. Replace `assert` validations with `error 'invalid-parameter-error`
2. Track `start-time`; return `inference-diagnostics` as 4th value

**Step 1: Replace all five `assert` calls in `vi`**

Current:
```lisp
  (assert (and (integerp n-params) (plusp n-params)) nil
          "vi: N-PARAMS must be a positive integer")
  (assert (and (integerp n-iterations) (plusp n-iterations)) nil
          "vi: N-ITERATIONS must be a positive integer")
  (assert (and (integerp n-elbo-samples) (plusp n-elbo-samples)) nil
          "vi: N-ELBO-SAMPLES must be a positive integer")
  (assert (> lr 0.0d0) nil "vi: LR must be a positive number")
```

Replace with:
```lisp
  (unless (and (integerp n-params) (plusp n-params))
    (error 'invalid-parameter-error
           :parameter :n-params :value n-params
           :message "vi: N-PARAMS must be a positive integer"))
  (unless (and (integerp n-iterations) (plusp n-iterations))
    (error 'invalid-parameter-error
           :parameter :n-iterations :value n-iterations
           :message "vi: N-ITERATIONS must be a positive integer"))
  (unless (and (integerp n-elbo-samples) (plusp n-elbo-samples))
    (error 'invalid-parameter-error
           :parameter :n-elbo-samples :value n-elbo-samples
           :message "vi: N-ELBO-SAMPLES must be a positive integer"))
  (unless (> lr 0.0d0)
    (error 'invalid-parameter-error
           :parameter :lr :value lr
           :message "vi: LR must be a positive number"))
```

**Step 2: Add `start-time` binding in `vi`'s `let*`**

Find:
```lisp
  (let* ((*log-pdf-error-warned-p* nil)
         (n-var-params (* 2 n-params))
```

Replace with:
```lisp
  (let* ((*log-pdf-error-warned-p* nil)
         (start-time (get-internal-real-time))
         (n-var-params (* 2 n-params))
```

**Step 3: Update the final return in `vi`**

Find:
```lisp
      (values mu-list sigma-list (nreverse elbo-history)))))
```

Replace with:
```lisp
      (values mu-list sigma-list (nreverse elbo-history)
              (make-inference-diagnostics
               :accept-rate 0.0d0       ; not applicable for VI
               :n-divergences 0
               :final-step-size (coerce lr 'double-float)
               :n-samples n-iterations
               :n-warmup 0
               :elapsed-seconds (/ (float (- (get-internal-real-time) start-time))
                                   internal-time-units-per-second))))))
```

**Step 4: Run full test suite**

```bash
rove cl-acorn.asd
```

Expected: all 170 existing tests pass + new conditions-test and inference-diagnostics-test pass.
Total should be ~185+ tests passing.

**Step 5: Commit**

```bash
git add src/inference/vi.lisp
git commit -m "feat(vi): replace asserts with conditions, return diagnostics as 4th value"
```

---

## Reference: Condition Hierarchy Summary

```
cl:error
└── infer:acorn-error                   (base, :message slot)
    ├── infer:model-error
    │   ├── infer:invalid-parameter-error   (:parameter, :value)
    │   └── infer:log-pdf-domain-error      (:distribution)
    └── infer:inference-error
        ├── infer:invalid-initial-params-error  (:params)
        └── infer:non-finite-gradient-error     (:params)

cl:warning
└── infer:high-divergence-warning       (:n-divergences, :n-samples)
```

## Reference: Restart Names

| Restart | Established in | Argument | Effect |
|---------|---------------|----------|--------|
| `use-fallback-params` | `hmc`, `nuts` | `new-params` (list) | Retry with new initial params |
| `return-empty-samples` | `hmc`, `nuts` | none | Return `(values '() 0.0d0 empty-diag)` |
| `continue-with-warnings` | `nuts` | none | Suppress high-divergence-warning |

## Reference: `inference-diagnostics` Fields

| Accessor | HMC | NUTS | VI |
|----------|-----|------|----|
| `diagnostics-accept-rate` | MH accept rate | Mean accept prob | 0.0 (N/A) |
| `diagnostics-n-divergences` | Always 0 | Post-warmup count | Always 0 |
| `diagnostics-final-step-size` | Adapted ε | Adapted ε | `lr` value |
| `diagnostics-n-samples` | `:n-samples` | `:n-samples` | `:n-iterations` |
| `diagnostics-n-warmup` | `:n-warmup` | `:n-warmup` | 0 |
| `diagnostics-elapsed-seconds` | Wall clock | Wall clock | Wall clock |
