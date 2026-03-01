# Forward-Mode AD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement forward-mode automatic differentiation using dual numbers with CLOS generic functions.

**Architecture:** CLOS `dual` class with double-float slots. Generic functions for binary ops (`+`,`-`,`*`,`/`) with 3 methods each (dual×dual, dual×number, number×dual) plus number×number fallback. Generic functions for unary transcendentals (sin, cos, tan, exp, log, sqrt, abs, expt). A `derivative` higher-order function seeds epsilon=1, evaluates, extracts parts.

**Tech Stack:** Common Lisp (SBCL), CLOS, ASDF, Rove (testing), cl-mcp (development tools)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `src/package.lisp`
- Create: `src/dual.lisp` (stub)
- Create: `src/arithmetic.lisp` (stub)
- Create: `src/transcendental.lisp` (stub)
- Create: `src/interface.lisp` (stub)
- Create: `tests/util.lisp`
- Create: `tests/dual-test.lisp` (stub)
- Create: `tests/arithmetic-test.lisp` (stub)
- Create: `tests/transcendental-test.lisp` (stub)
- Create: `tests/derivative-test.lisp` (stub)
- Modify: `cl-acorn.asd`
- Remove: `src/main.lisp`, `tests/main.lisp`

**Step 1: Create `src/package.lisp`**

```lisp
(defpackage #:cl-acorn.ad
  (:use #:cl)
  (:shadow #:+ #:- #:* #:/
           #:sin #:cos #:tan #:exp #:log #:expt #:sqrt #:abs)
  (:export
   ;; Dual number class and accessors
   #:dual
   #:make-dual
   #:dual-real
   #:dual-epsilon
   ;; Arithmetic
   #:+ #:- #:* #:/
   ;; Transcendental
   #:sin #:cos #:tan #:exp #:log #:expt #:sqrt #:abs
   ;; Interface
   #:derivative))
```

**Step 2: Create source stubs**

Each source file gets just `(in-package #:cl-acorn.ad)` as a placeholder.

`src/dual.lisp`:
```lisp
(in-package #:cl-acorn.ad)
```

`src/arithmetic.lisp`:
```lisp
(in-package #:cl-acorn.ad)
```

`src/transcendental.lisp`:
```lisp
(in-package #:cl-acorn.ad)
```

`src/interface.lisp`:
```lisp
(in-package #:cl-acorn.ad)
```

**Step 3: Create `tests/util.lisp`**

```lisp
(defpackage #:cl-acorn/tests/util
  (:use #:cl)
  (:export #:approx=))

(in-package #:cl-acorn/tests/util)

(defun approx= (a b &optional (tolerance 1d-10))
  "Check if two numbers are approximately equal within tolerance."
  (< (abs (- a b)) tolerance))
```

**Step 4: Create test stubs**

`tests/dual-test.lisp`:
```lisp
(defpackage #:cl-acorn/tests/dual-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/dual-test)
```

`tests/arithmetic-test.lisp`:
```lisp
(defpackage #:cl-acorn/tests/arithmetic-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/arithmetic-test)
```

`tests/transcendental-test.lisp`:
```lisp
(defpackage #:cl-acorn/tests/transcendental-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/transcendental-test)
```

`tests/derivative-test.lisp`:
```lisp
(defpackage #:cl-acorn/tests/derivative-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/derivative-test)
```

**Step 5: Update `cl-acorn.asd`**

```lisp
(defsystem "cl-acorn"
  :version "0.1.0"
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
                 (:file "interface"))))
  :description "Forward-mode automatic differentiation using dual numbers"
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
                 (:file "derivative-test"))))
  :description "Test system for cl-acorn"
  :perform (test-op (op c) (symbol-call :rove :run c)))
```

**Step 6: Remove old scaffold files**

```bash
rm src/main.lisp tests/main.lisp
```

**Step 7: Verify system loads**

Use `load-system` tool: `{"system": "cl-acorn", "force": true}`
Then: `{"system": "cl-acorn/tests", "force": true}`

Expected: Both load without errors.

**Step 8: Commit**

```bash
git add -A && git commit -m "Scaffold forward-mode AD project structure

Create package definition (cl-acorn.ad), source stubs, test stubs,
and shared test utility. Remove old placeholder files."
```

---

### Task 2: Dual Class (TDD)

**Files:**
- Write tests: `tests/dual-test.lisp`
- Implement: `src/dual.lisp`

**Step 1: Write failing tests in `tests/dual-test.lisp`**

```lisp
(defpackage #:cl-acorn/tests/dual-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/dual-test)

(deftest test-make-dual-defaults
  (testing "make-dual with default epsilon"
    (let ((d (ad:make-dual 3.0d0)))
      (ok (typep d 'ad:dual))
      (ok (approx= (ad:dual-real d) 3.0d0))
      (ok (approx= (ad:dual-epsilon d) 0.0d0)))))

(deftest test-make-dual-both-args
  (testing "make-dual with real and epsilon"
    (let ((d (ad:make-dual 2.0d0 5.0d0)))
      (ok (approx= (ad:dual-real d) 2.0d0))
      (ok (approx= (ad:dual-epsilon d) 5.0d0)))))

(deftest test-make-dual-coercion
  (testing "make-dual coerces integers to double-float"
    (let ((d (ad:make-dual 3 7)))
      (ok (typep (ad:dual-real d) 'double-float))
      (ok (typep (ad:dual-epsilon d) 'double-float))
      (ok (approx= (ad:dual-real d) 3.0d0))
      (ok (approx= (ad:dual-epsilon d) 7.0d0)))))

(deftest test-make-dual-rational-coercion
  (testing "make-dual coerces rationals to double-float"
    (let ((d (ad:make-dual 1/3 1)))
      (ok (typep (ad:dual-real d) 'double-float))
      (ok (approx= (ad:dual-real d) (coerce 1/3 'double-float))))))

(deftest test-dual-print
  (testing "print-object produces readable representation"
    (let ((output (format nil "~A" (ad:make-dual 1.0d0 2.0d0))))
      (ok (search "DUAL" output))
      (ok (search "1.0d0" output))
      (ok (search "2.0d0" output)))))
```

**Step 2: Verify tests fail**

Use `run-tests` tool: `{"system": "cl-acorn/tests"}`
Expected: FAIL — `make-dual` is undefined.

**Step 3: Implement `src/dual.lisp`**

```lisp
(in-package #:cl-acorn.ad)

(defclass dual ()
  ((real-part :initarg :real
              :reader dual-real
              :type double-float
              :documentation "The real (primal) component of the dual number.")
   (epsilon   :initarg :epsilon
              :reader dual-epsilon
              :type double-float
              :initform 0.0d0
              :documentation "The infinitesimal (tangent) component."))
  (:documentation "Dual number a + b*epsilon where epsilon^2 = 0.
Used for forward-mode automatic differentiation."))

(defun make-dual (real &optional (epsilon 0.0d0))
  "Construct a dual number, coercing inputs to double-float."
  (make-instance 'dual
                 :real (coerce real 'double-float)
                 :epsilon (coerce epsilon 'double-float)))

(defmethod print-object ((d dual) stream)
  (print-unreadable-object (d stream :type t)
    (format stream "~A + ~Aε" (dual-real d) (dual-epsilon d))))
```

**Step 4: Reload and verify tests pass**

Use `load-system`: `{"system": "cl-acorn", "force": true}`
Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: All dual-test tests PASS.

**Step 5: Commit**

```bash
git add src/dual.lisp tests/dual-test.lisp && git commit -m "Add dual number class with TDD

CLOS dual class with double-float slots, make-dual constructor
with type coercion, and print-object method."
```

---

### Task 3: Addition & Subtraction (TDD)

**Files:**
- Write tests: `tests/arithmetic-test.lisp`
- Implement: `src/arithmetic.lisp`

**Step 1: Write failing tests in `tests/arithmetic-test.lisp`**

```lisp
(defpackage #:cl-acorn/tests/arithmetic-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/arithmetic-test)

;;; Addition tests

(deftest test-add-dual-dual
  (testing "(1+2e) + (3+4e) = (4+6e)"
    (let ((result (ad:+ (ad:make-dual 1 2) (ad:make-dual 3 4))))
      (ok (typep result 'ad:dual))
      (ok (approx= (ad:dual-real result) 4.0d0))
      (ok (approx= (ad:dual-epsilon result) 6.0d0)))))

(deftest test-add-dual-number
  (testing "(1+2e) + 3 = (4+2e)"
    (let ((result (ad:+ (ad:make-dual 1 2) 3)))
      (ok (approx= (ad:dual-real result) 4.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-add-number-dual
  (testing "3 + (1+2e) = (4+2e)"
    (let ((result (ad:+ 3 (ad:make-dual 1 2))))
      (ok (approx= (ad:dual-real result) 4.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-add-number-number
  (testing "3 + 4 = 7 (falls through to cl:+)"
    (ok (= (ad:+ 3 4) 7))))

(deftest test-add-n-ary
  (testing "n-ary addition"
    (let ((result (ad:+ (ad:make-dual 1 1) (ad:make-dual 2 2) (ad:make-dual 3 3))))
      (ok (approx= (ad:dual-real result) 6.0d0))
      (ok (approx= (ad:dual-epsilon result) 6.0d0)))))

(deftest test-add-zero-args
  (testing "(ad:+) returns 0"
    (ok (= (ad:+) 0))))

(deftest test-add-one-arg
  (testing "(ad:+ x) returns x"
    (let ((d (ad:make-dual 5 3)))
      (ok (eq (ad:+ d) d)))))

;;; Subtraction tests

(deftest test-sub-dual-dual
  (testing "(5+3e) - (2+1e) = (3+2e)"
    (let ((result (ad:- (ad:make-dual 5 3) (ad:make-dual 2 1))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-sub-dual-number
  (testing "(5+3e) - 2 = (3+3e)"
    (let ((result (ad:- (ad:make-dual 5 3) 2)))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 3.0d0)))))

(deftest test-sub-number-dual
  (testing "5 - (2+3e) = (3 + -3e)"
    (let ((result (ad:- 5 (ad:make-dual 2 3))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) -3.0d0)))))

(deftest test-sub-negation
  (testing "(ad:- x) negates dual"
    (let ((result (ad:- (ad:make-dual 3 5))))
      (ok (approx= (ad:dual-real result) -3.0d0))
      (ok (approx= (ad:dual-epsilon result) -5.0d0)))))

(deftest test-sub-number-number
  (testing "5 - 3 = 2 (falls through to cl:-)"
    (ok (= (ad:- 5 3) 2))))
```

**Step 2: Verify tests fail**

Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: FAIL — `ad:+` and `ad:-` are undefined.

**Step 3: Implement addition and subtraction in `src/arithmetic.lisp`**

```lisp
(in-package #:cl-acorn.ad)

;;; --- Binary generic functions ---

(defgeneric binary-add (a b)
  (:documentation "Binary addition supporting dual numbers."))

(defmethod binary-add ((a dual) (b dual))
  (make-dual (cl:+ (dual-real a) (dual-real b))
             (cl:+ (dual-epsilon a) (dual-epsilon b))))

(defmethod binary-add ((a dual) (b number))
  (make-dual (cl:+ (dual-real a) (coerce b 'double-float))
             (dual-epsilon a)))

(defmethod binary-add ((a number) (b dual))
  (make-dual (cl:+ (coerce a 'double-float) (dual-real b))
             (dual-epsilon b)))

(defmethod binary-add ((a number) (b number))
  (cl:+ a b))

(defgeneric binary-sub (a b)
  (:documentation "Binary subtraction supporting dual numbers."))

(defmethod binary-sub ((a dual) (b dual))
  (make-dual (cl:- (dual-real a) (dual-real b))
             (cl:- (dual-epsilon a) (dual-epsilon b))))

(defmethod binary-sub ((a dual) (b number))
  (make-dual (cl:- (dual-real a) (coerce b 'double-float))
             (dual-epsilon a)))

(defmethod binary-sub ((a number) (b dual))
  (make-dual (cl:- (coerce a 'double-float) (dual-real b))
             (cl:- (dual-epsilon b))))

(defmethod binary-sub ((a number) (b number))
  (cl:- a b))

;;; --- N-ary wrappers ---

(defun + (&rest args)
  "Addition supporting dual numbers."
  (case (length args)
    (0 0)
    (1 (first args))
    (t (reduce #'binary-add args))))

(defun - (arg &rest more)
  "Subtraction/negation supporting dual numbers."
  (if more
      (reduce #'binary-sub more :initial-value arg)
      (binary-sub 0 arg)))
```

**Step 4: Reload and verify tests pass**

Use `load-system`: `{"system": "cl-acorn", "force": true}`
Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: All addition/subtraction tests PASS.

**Step 5: Commit**

```bash
git add src/arithmetic.lisp tests/arithmetic-test.lisp && git commit -m "Add dual number addition and subtraction

Binary generic functions for add/sub with dual×dual, dual×number,
number×dual, and number×number dispatch. N-ary wrappers with
reduce. Includes negation via (ad:- x)."
```

---

### Task 4: Multiplication & Division (TDD)

**Files:**
- Append tests: `tests/arithmetic-test.lisp`
- Append impl: `src/arithmetic.lisp`

**Step 1: Write failing tests — append to `tests/arithmetic-test.lisp`**

```lisp
;;; Multiplication tests

(deftest test-mul-dual-dual
  (testing "(2+3e) * (4+5e) = (8 + 22e)"
    ;; real: 2*4=8, eps: 2*5 + 3*4 = 10+12 = 22
    (let ((result (ad:* (ad:make-dual 2 3) (ad:make-dual 4 5))))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 22.0d0)))))

(deftest test-mul-dual-number
  (testing "(2+3e) * 4 = (8+12e)"
    (let ((result (ad:* (ad:make-dual 2 3) 4)))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 12.0d0)))))

(deftest test-mul-number-dual
  (testing "4 * (2+3e) = (8+12e)"
    (let ((result (ad:* 4 (ad:make-dual 2 3))))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 12.0d0)))))

(deftest test-mul-number-number
  (testing "3 * 4 = 12"
    (ok (= (ad:* 3 4) 12))))

(deftest test-mul-n-ary
  (testing "n-ary multiplication"
    ;; (1+1e) * (2+0e) * (3+0e) = first: (2+2e), then: (6+6e)
    (let ((result (ad:* (ad:make-dual 1 1) (ad:make-dual 2 0) (ad:make-dual 3 0))))
      (ok (approx= (ad:dual-real result) 6.0d0))
      (ok (approx= (ad:dual-epsilon result) 6.0d0)))))

(deftest test-mul-zero-args
  (testing "(ad:*) returns 1"
    (ok (= (ad:*) 1))))

;;; Division tests

(deftest test-div-dual-dual
  (testing "(6+4e) / (3+1e) = (2 + 2/9 * e)"
    ;; real: 6/3=2, eps: (4*3 - 6*1)/9 = (12-6)/9 = 6/9 = 2/3
    (let ((result (ad:/ (ad:make-dual 6 4) (ad:make-dual 3 1))))
      (ok (approx= (ad:dual-real result) 2.0d0))
      (ok (approx= (ad:dual-epsilon result) (/ 2.0d0 3.0d0))))))

(deftest test-div-dual-number
  (testing "(6+4e) / 2 = (3+2e)"
    (let ((result (ad:/ (ad:make-dual 6 4) 2)))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-div-number-dual
  (testing "1 / (2+1e) = (0.5 + -0.25e)"
    ;; real: 1/2=0.5, eps: (0*2 - 1*1)/4 = -1/4 = -0.25
    (let ((result (ad:/ 1 (ad:make-dual 2 1))))
      (ok (approx= (ad:dual-real result) 0.5d0))
      (ok (approx= (ad:dual-epsilon result) -0.25d0)))))

(deftest test-div-reciprocal
  (testing "(ad:/ x) computes reciprocal"
    ;; 1/(2+1e) = 0.5 - 0.25e
    (let ((result (ad:/ (ad:make-dual 2 1))))
      (ok (approx= (ad:dual-real result) 0.5d0))
      (ok (approx= (ad:dual-epsilon result) -0.25d0)))))

(deftest test-div-number-number
  (testing "6 / 3 = 2"
    (ok (= (ad:/ 6 3) 2))))
```

**Step 2: Verify new tests fail**

Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: Multiplication/division tests FAIL.

**Step 3: Implement mul/div — append to `src/arithmetic.lisp`**

```lisp
;;; --- Multiplication ---

(defgeneric binary-mul (a b)
  (:documentation "Binary multiplication supporting dual numbers."))

(defmethod binary-mul ((a dual) (b dual))
  (let ((ar (dual-real a))
        (ae (dual-epsilon a))
        (br (dual-real b))
        (be (dual-epsilon b)))
    (make-dual (cl:* ar br)
               (cl:+ (cl:* ar be) (cl:* ae br)))))

(defmethod binary-mul ((a dual) (b number))
  (let ((n (coerce b 'double-float)))
    (make-dual (cl:* (dual-real a) n)
               (cl:* (dual-epsilon a) n))))

(defmethod binary-mul ((a number) (b dual))
  (let ((n (coerce a 'double-float)))
    (make-dual (cl:* n (dual-real b))
               (cl:* n (dual-epsilon b)))))

(defmethod binary-mul ((a number) (b number))
  (cl:* a b))

;;; --- Division ---

(defgeneric binary-div (a b)
  (:documentation "Binary division supporting dual numbers."))

(defmethod binary-div ((a dual) (b dual))
  (let ((p (dual-real a))
        (q (dual-epsilon a))
        (c (dual-real b))
        (d (dual-epsilon b)))
    (let ((c2 (cl:* c c)))
      (make-dual (cl:/ p c)
                 (cl:/ (cl:- (cl:* q c) (cl:* p d))
                       c2)))))

(defmethod binary-div ((a dual) (b number))
  (let ((n (coerce b 'double-float)))
    (make-dual (cl:/ (dual-real a) n)
               (cl:/ (dual-epsilon a) n))))

(defmethod binary-div ((a number) (b dual))
  (let ((p (coerce a 'double-float))
        (c (dual-real b))
        (d (dual-epsilon b)))
    (make-dual (cl:/ p c)
               (cl:/ (cl:- (cl:* p d))
                     (cl:* c c)))))

(defmethod binary-div ((a number) (b number))
  (cl:/ a b))

;;; --- N-ary wrappers (mul/div) ---

(defun * (&rest args)
  "Multiplication supporting dual numbers."
  (case (length args)
    (0 1)
    (1 (first args))
    (t (reduce #'binary-mul args))))

(defun / (arg &rest more)
  "Division/reciprocal supporting dual numbers."
  (if more
      (reduce #'binary-div more :initial-value arg)
      (binary-div 1 arg)))
```

**Step 4: Reload and verify tests pass**

Use `load-system`: `{"system": "cl-acorn", "force": true}`
Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: All arithmetic tests PASS.

**Step 5: Commit**

```bash
git add src/arithmetic.lisp tests/arithmetic-test.lisp && git commit -m "Add dual number multiplication and division

Complete arithmetic: binary-mul, binary-div generics with all type
combinations. N-ary *, / wrappers. Reciprocal via (ad:/ x)."
```

---

### Task 5: Transcendental Functions (TDD)

**Files:**
- Write tests: `tests/transcendental-test.lisp`
- Implement: `src/transcendental.lisp`

**Step 1: Write failing tests in `tests/transcendental-test.lisp`**

```lisp
(defpackage #:cl-acorn/tests/transcendental-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/transcendental-test)

;;; sin

(deftest test-sin-dual
  (testing "sin(1+1e) = sin(1) + cos(1)*e"
    (let ((result (ad:sin (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (sin 1.0d0)))
      (ok (approx= (ad:dual-epsilon result) (cos 1.0d0))))))

(deftest test-sin-number
  (testing "sin delegates to cl:sin for numbers"
    (ok (= (ad:sin 0.0d0) (sin 0.0d0)))))

;;; cos

(deftest test-cos-dual
  (testing "cos(1+1e) = cos(1) + -sin(1)*e"
    (let ((result (ad:cos (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (cos 1.0d0)))
      (ok (approx= (ad:dual-epsilon result) (- (sin 1.0d0)))))))

(deftest test-cos-number
  (testing "cos delegates to cl:cos for numbers"
    (ok (= (ad:cos 0.0d0) (cos 0.0d0)))))

;;; tan

(deftest test-tan-dual
  (testing "tan(1+1e) = tan(1) + 1/cos^2(1)*e"
    (let ((result (ad:tan (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (tan 1.0d0)))
      (ok (approx= (ad:dual-epsilon result)
                    (/ 1.0d0 (expt (cos 1.0d0) 2)))))))

;;; exp

(deftest test-exp-dual
  (testing "exp(1+1e) = exp(1) + exp(1)*e"
    (let ((result (ad:exp (ad:make-dual 1.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (exp 1.0d0)))
      (ok (approx= (ad:dual-epsilon result) (exp 1.0d0))))))

(deftest test-exp-number
  (testing "exp delegates to cl:exp for numbers"
    (ok (= (ad:exp 1.0d0) (exp 1.0d0)))))

;;; log

(deftest test-log-dual
  (testing "log(2+1e) = log(2) + 1/2*e"
    (let ((result (ad:log (ad:make-dual 2.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) (log 2.0d0)))
      (ok (approx= (ad:dual-epsilon result) 0.5d0)))))

(deftest test-log-number
  (testing "log delegates to cl:log for numbers"
    (ok (= (ad:log 2.0d0) (log 2.0d0)))))

(deftest test-log-with-base
  (testing "log with base argument"
    ;; log_10(100) = 2, and with dual: log_10(100+1e) = log(100+1e)/log(10)
    (let ((result (ad:log (ad:make-dual 100.0d0 1.0d0) 10.0d0)))
      (ok (approx= (ad:dual-real result) (log 100.0d0 10.0d0)))
      ;; d/dx log_b(x) = 1/(x*ln(b))
      (ok (approx= (ad:dual-epsilon result)
                    (/ 1.0d0 (* 100.0d0 (log 10.0d0))))))))

;;; sqrt

(deftest test-sqrt-dual
  (testing "sqrt(4+1e) = sqrt(4) + 1/(2*sqrt(4))*e = 2 + 0.25e"
    (let ((result (ad:sqrt (ad:make-dual 4.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) 2.0d0))
      (ok (approx= (ad:dual-epsilon result) 0.25d0)))))

(deftest test-sqrt-number
  (testing "sqrt delegates to cl:sqrt for numbers"
    (ok (= (ad:sqrt 4.0d0) (sqrt 4.0d0)))))

;;; abs

(deftest test-abs-dual-positive
  (testing "abs of positive dual"
    (let ((result (ad:abs (ad:make-dual 3.0d0 2.0d0))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) 2.0d0)))))

(deftest test-abs-dual-negative
  (testing "abs of negative dual"
    (let ((result (ad:abs (ad:make-dual -3.0d0 2.0d0))))
      (ok (approx= (ad:dual-real result) 3.0d0))
      (ok (approx= (ad:dual-epsilon result) -2.0d0)))))

(deftest test-abs-number
  (testing "abs delegates to cl:abs for numbers"
    (ok (= (ad:abs -5) 5))))

;;; expt

(deftest test-expt-dual-number
  (testing "expt((2+1e), 3) = 8 + 12e (power rule: 3*2^2*1)"
    (let ((result (ad:expt (ad:make-dual 2.0d0 1.0d0) 3)))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) 12.0d0)))))

(deftest test-expt-number-dual
  (testing "expt(2, (3+1e)) = 2^3 + 2^3*ln(2)*e"
    (let ((result (ad:expt 2 (ad:make-dual 3.0d0 1.0d0))))
      (ok (approx= (ad:dual-real result) 8.0d0))
      (ok (approx= (ad:dual-epsilon result) (* 8.0d0 (log 2.0d0)))))))

(deftest test-expt-number-number
  (testing "expt delegates to cl:expt for numbers"
    (ok (= (ad:expt 2 3) 8))))

;;; Nonunit epsilon

(deftest test-sin-nonunit-epsilon
  (testing "sin with non-unit epsilon: sin(0.5+3e)"
    (let ((result (ad:sin (ad:make-dual 0.5d0 3.0d0))))
      (ok (approx= (ad:dual-real result) (sin 0.5d0)))
      (ok (approx= (ad:dual-epsilon result) (* (cos 0.5d0) 3.0d0))))))
```

**Step 2: Verify tests fail**

Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: Transcendental tests FAIL.

**Step 3: Implement `src/transcendental.lisp`**

```lisp
(in-package #:cl-acorn.ad)

;;; --- Unary generic functions ---

(defgeneric ad-sin (x)
  (:documentation "Sine supporting dual numbers."))

(defmethod ad-sin ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (make-dual (cl:sin a)
               (cl:* (cl:cos a) b))))

(defmethod ad-sin ((x number))
  (cl:sin x))

(defun sin (x)
  "Sine supporting dual numbers."
  (ad-sin x))

(defgeneric ad-cos (x)
  (:documentation "Cosine supporting dual numbers."))

(defmethod ad-cos ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (make-dual (cl:cos a)
               (cl:- (cl:* (cl:sin a) b)))))

(defmethod ad-cos ((x number))
  (cl:cos x))

(defun cos (x)
  "Cosine supporting dual numbers."
  (ad-cos x))

(defgeneric ad-tan (x)
  (:documentation "Tangent supporting dual numbers."))

(defmethod ad-tan ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (let ((cos-a (cl:cos a)))
      (make-dual (cl:tan a)
                 (cl:/ b (cl:* cos-a cos-a))))))

(defmethod ad-tan ((x number))
  (cl:tan x))

(defun tan (x)
  "Tangent supporting dual numbers."
  (ad-tan x))

(defgeneric ad-exp (x)
  (:documentation "Exponential supporting dual numbers."))

(defmethod ad-exp ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (let ((exp-a (cl:exp a)))
      (make-dual exp-a
                 (cl:* exp-a b)))))

(defmethod ad-exp ((x number))
  (cl:exp x))

(defun exp (x)
  "Exponential supporting dual numbers."
  (ad-exp x))

(defgeneric unary-log (x)
  (:documentation "Natural logarithm supporting dual numbers."))

(defmethod unary-log ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (make-dual (cl:log a)
               (cl:/ b a))))

(defmethod unary-log ((x number))
  (cl:log x))

(defun log (x &optional base)
  "Logarithm supporting dual numbers. Without base, returns natural log."
  (if base
      (binary-div (unary-log x) (unary-log base))
      (unary-log x)))

(defgeneric ad-sqrt (x)
  (:documentation "Square root supporting dual numbers."))

(defmethod ad-sqrt ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (let ((sqrt-a (cl:sqrt a)))
      (make-dual sqrt-a
                 (cl:/ b (cl:* 2.0d0 sqrt-a))))))

(defmethod ad-sqrt ((x number))
  (cl:sqrt x))

(defun sqrt (x)
  "Square root supporting dual numbers."
  (ad-sqrt x))

(defgeneric ad-abs (x)
  (:documentation "Absolute value supporting dual numbers."))

(defmethod ad-abs ((x dual))
  (let ((a (dual-real x))
        (b (dual-epsilon x)))
    (make-dual (cl:abs a)
               (cl:* (coerce (cl:signum a) 'double-float) b))))

(defmethod ad-abs ((x number))
  (cl:abs x))

(defun abs (x)
  "Absolute value supporting dual numbers."
  (ad-abs x))

;;; --- expt (binary) ---

(defgeneric binary-expt (base power)
  (:documentation "Exponentiation supporting dual numbers."))

(defmethod binary-expt ((base dual) (power number))
  (let ((a (dual-real base))
        (b (dual-epsilon base))
        (n (coerce power 'double-float)))
    (make-dual (cl:expt a n)
               (cl:* n (cl:expt a (cl:- n 1.0d0)) b))))

(defmethod binary-expt ((base number) (power dual))
  (let ((c (coerce base 'double-float))
        (a (dual-real power))
        (b (dual-epsilon power)))
    (let ((ca (cl:expt c a)))
      (make-dual ca
                 (cl:* ca (cl:log c) b)))))

(defmethod binary-expt ((base dual) (power dual))
  ;; x^y = exp(y * ln(x)) — reuses AD-aware exp, *, log
  (exp (* power (log base))))

(defmethod binary-expt ((base number) (power number))
  (cl:expt base power))

(defun expt (base power)
  "Exponentiation supporting dual numbers."
  (binary-expt base power))
```

**Step 4: Reload and verify tests pass**

Use `load-system`: `{"system": "cl-acorn", "force": true}`
Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: All transcendental tests PASS.

**Step 5: Commit**

```bash
git add src/transcendental.lisp tests/transcendental-test.lisp && git commit -m "Add transcendental functions for dual numbers

sin, cos, tan, exp, log (with optional base), sqrt, abs, expt.
Each uses chain rule for dual dispatch. expt(dual,dual) uses
exp(y*ln(x)) identity."
```

---

### Task 6: Derivative Interface (TDD)

**Files:**
- Write tests: `tests/derivative-test.lisp`
- Implement: `src/interface.lisp`

**Step 1: Write failing tests in `tests/derivative-test.lisp`**

```lisp
(defpackage #:cl-acorn/tests/derivative-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/derivative-test)

(deftest test-derivative-identity
  (testing "d/dx[x] = 1"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) x) 5.0d0)
      (ok (approx= val 5.0d0))
      (ok (approx= grad 1.0d0)))))

(deftest test-derivative-constant
  (testing "d/dx[42] = 0"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (declare (ignore x)) 42.0d0) 5.0d0)
      (ok (approx= val 42.0d0))
      (ok (approx= grad 0.0d0)))))

(deftest test-derivative-square
  (testing "d/dx[x^2] = 2x at x=3"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (ad:* x x)) 3.0d0)
      (ok (approx= val 9.0d0))
      (ok (approx= grad 6.0d0)))))

(deftest test-derivative-polynomial
  (testing "d/dx[3x^2 + 2x + 1] at x=2"
    ;; f(2) = 12+4+1 = 17, f'(2) = 6*2+2 = 14
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x) (ad:+ (ad:* 3 (ad:* x x)) (ad:+ (ad:* 2 x) 1)))
         2.0d0)
      (ok (approx= val 17.0d0))
      (ok (approx= grad 14.0d0)))))

(deftest test-derivative-sin
  (testing "d/dx[sin(x)] = cos(x) at x=1"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (ad:sin x)) 1.0d0)
      (ok (approx= val (sin 1.0d0)))
      (ok (approx= grad (cos 1.0d0))))))

(deftest test-derivative-composite-1
  (testing "d/dx[sin(x^2) + exp(2x)/x] at x=1"
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x)
           (ad:+ (ad:sin (ad:* x x))
                 (ad:/ (ad:exp (ad:* 2 x)) x)))
         1.0d0)
      ;; f(1) = sin(1) + exp(2)
      (let ((expected-val (+ (sin 1.0d0) (exp 2.0d0))))
        (ok (approx= val expected-val)))
      ;; f'(1) = 2*cos(1) + exp(2)*(2*1-1)/1^2
      (let ((expected-grad (+ (* 2.0d0 (cos 1.0d0))
                              (* (exp 2.0d0)
                                 (/ (- (* 2.0d0 1.0d0) 1.0d0)
                                    (* 1.0d0 1.0d0))))))
        (ok (approx= grad expected-grad))))))

(deftest test-derivative-composite-2
  (testing "d/dx[log(x) * sqrt(x)] at x=2"
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x) (ad:* (ad:log x) (ad:sqrt x)))
         2.0d0)
      ;; f(2) = ln(2) * sqrt(2)
      (ok (approx= val (* (log 2.0d0) (sqrt 2.0d0))))
      ;; f'(x) = (2 + ln(x)) / (2*sqrt(x))
      (let ((expected-grad (/ (+ 2.0d0 (log 2.0d0))
                              (* 2.0d0 (sqrt 2.0d0)))))
        (ok (approx= grad expected-grad))))))

(deftest test-derivative-exp-chain
  (testing "d/dx[exp(sin(x))] at x=1"
    (multiple-value-bind (val grad)
        (ad:derivative
         (lambda (x) (ad:exp (ad:sin x)))
         1.0d0)
      ;; f(1) = exp(sin(1))
      (ok (approx= val (exp (sin 1.0d0))))
      ;; f'(1) = exp(sin(1)) * cos(1)
      (ok (approx= grad (* (exp (sin 1.0d0)) (cos 1.0d0)))))))

(deftest test-derivative-integer-input
  (testing "derivative accepts integer input"
    (multiple-value-bind (val grad)
        (ad:derivative (lambda (x) (ad:* x x)) 3)
      (ok (approx= val 9.0d0))
      (ok (approx= grad 6.0d0)))))
```

**Step 2: Verify tests fail**

Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: FAIL — `derivative` is undefined.

**Step 3: Implement `src/interface.lisp`**

```lisp
(in-package #:cl-acorn.ad)

(defun derivative (fn x)
  "Compute f(x) and f'(x) using forward-mode automatic differentiation.
Returns (values f(x) f'(x)) as double-floats.
FN must be a function of one argument using cl-acorn.ad arithmetic."
  (let* ((x-dual (make-dual x 1.0d0))
         (result (funcall fn x-dual)))
    (etypecase result
      (dual   (values (dual-real result) (dual-epsilon result)))
      (number (values (coerce result 'double-float) 0.0d0)))))
```

**Step 4: Reload and verify tests pass**

Use `load-system`: `{"system": "cl-acorn", "force": true}`
Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: ALL tests PASS across all test files.

**Step 5: Commit**

```bash
git add src/interface.lisp tests/derivative-test.lisp && git commit -m "Add derivative interface with composite function tests

Higher-order derivative function: seeds epsilon=1, evaluates,
extracts (values f(x) f'(x)). Tests cover identity, constant,
polynomial, sin(x^2)+exp(2x)/x, log(x)*sqrt(x), and exp(sin(x))."
```

---

### Task 7: Final Validation

**Step 1: Full system compile check**

Use `repl-eval`:
```lisp
(asdf:compile-system :cl-acorn :force t)
```
Expected: No compiler warnings (optimization notes are OK).

**Step 2: Run complete test suite**

Use `run-tests`: `{"system": "cl-acorn/tests"}`
Expected: ALL tests pass, 0 failures.

**Step 3: Verify via REPL smoke test**

Use `repl-eval`:
```lisp
(let ((ad (find-package :cl-acorn.ad)))
  (format nil "Exported symbols: ~{~A~^, ~}"
          (let (syms)
            (do-external-symbols (s ad) (push (symbol-name s) syms))
            (sort syms #'string<))))
```
Expected: All 16 exported symbols listed.

**Step 4: Final commit**

```bash
git add -A && git commit -m "Complete phase 1: forward-mode AD via dual numbers

All tests passing. Exports: dual, make-dual, dual-real, dual-epsilon,
+, -, *, /, sin, cos, tan, exp, log, expt, sqrt, abs, derivative."
```
