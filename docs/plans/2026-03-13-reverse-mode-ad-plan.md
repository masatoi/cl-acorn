# Reverse-Mode AD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add tape-based reverse-mode AD to cl-acorn, providing `gradient`, `jacobian-vector-product`, and `hessian-vector-product` alongside the existing forward-mode `derivative`.

**Architecture:** Extend existing CLOS generic functions (`binary-add`, `ad-sin`, etc.) with `tape-node` methods. A dynamic tape (`*tape*` special variable) records computation graphs during forward evaluation. Backward pass traverses the tape in reverse to accumulate gradients. Forward-over-reverse composition enables Hessian-vector products.

**Tech Stack:** Common Lisp, CLOS (generic functions + methods), Rove (tests), ASDF (build system)

**Reference docs:**
- Design: `docs/plans/2026-03-13-reverse-mode-ad-design.md`
- Existing forward-mode: `src/dual.lisp`, `src/arithmetic.lisp`, `src/transcendental.lisp`
- Test patterns: `tests/arithmetic-test.lisp`, `tests/derivative-test.lisp`
- Test utility: `tests/util.lisp` (`approx=` helper, tolerance 1d-10)

**Important CL-MCP notes:**
- `lisp-edit-form` CST parser does NOT resolve package-local-nicknames (`ad:` prefix)
- For test files using `ad:` prefix, use the Write tool
- For implementation files (in `cl-acorn.ad` package, using `cl:` prefix), `lisp-edit-form` works well

---

### Task 1: Package exports and ASDF system definition

Add the new symbols to the package and register new source files in the ASDF system.

**Files:**
- Modify: `src/package.lisp`
- Modify: `cl-acorn.asd`

**Step 1: Add exports to package definition**

Use `lisp-edit-form` to replace the `defpackage` in `src/package.lisp`. Add these exports after the existing ones:

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
   ;; Interface (forward-mode)
   #:derivative
   ;; Tape node class and accessors (reverse-mode)
   #:tape-node
   #:node-value
   #:node-gradient
   ;; Interface (reverse-mode)
   #:gradient
   #:jacobian-vector-product
   #:hessian-vector-product))
```

**Step 2: Add new source files to ASDF system**

Use `lisp-edit-form` to replace the first `defsystem` in `cl-acorn.asd`. Add four new files after `interface`, update version and description:

```lisp
(defsystem "cl-acorn"
  :version "0.2.0"
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
                 (:file "gradient"))))
  :description "Automatic differentiation using dual numbers and reverse-mode tape"
  :in-order-to ((test-op (test-op "cl-acorn/tests"))))
```

**Step 3: Add test files to test system**

Use `lisp-edit-form` to replace the second `defsystem` in `cl-acorn.asd`. Add four test files:

```lisp
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
                 (:file "gradient-test"))))
  :description "Test system for cl-acorn"
  :perform (test-op (op c) (symbol-call :rove :run c)))
```

**Step 4: Commit**

```bash
git add src/package.lisp cl-acorn.asd
git commit -m "feat: add reverse-mode AD package exports and ASDF entries"
```

---

### Task 2: tape-node class, tape management, and backward pass

Create the core reverse-mode infrastructure.

**Files:**
- Create: `src/tape.lisp`
- Test: `tests/tape-test.lisp`

**Step 1: Write the failing test**

Create `tests/tape-test.lisp` using the Write tool (test files use `ad:` prefix which `lisp-edit-form` cannot parse):

```lisp
(defpackage #:cl-acorn/tests/tape-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/tape-test)

(deftest test-make-tape-node
  (testing "tape-node stores value"
    (let ((n (make-instance 'ad:tape-node :value 3.0d0)))
      (ok (approx= (ad:node-value n) 3.0d0))
      (ok (approx= (ad:node-gradient n) 0)))))

(deftest test-tape-recording
  (testing "*tape* records nodes during computation"
    (let ((cl-acorn.ad::*tape* '()))
      (let ((a (cl-acorn.ad::make-node 2.0d0 nil))
            (b (cl-acorn.ad::make-node 3.0d0 nil)))
        (declare (ignore a b))
        (ok (= (length cl-acorn.ad::*tape*) 2))))))

(deftest test-backward-simple-add
  (testing "backward propagates gradient through addition: z = a + b"
    (let ((cl-acorn.ad::*tape* '()))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             ;; z = a + b, dz/da = 1, dz/db = 1
             (z (cl-acorn.ad::make-node 5.0d0
                                        (list (cons a 1.0d0)
                                              (cons b 1.0d0)))))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient z) 1.0d0))
        (ok (approx= (ad:node-gradient a) 1.0d0))
        (ok (approx= (ad:node-gradient b) 1.0d0))))))

(deftest test-backward-simple-mul
  (testing "backward propagates gradient through multiplication: z = a * b"
    (let ((cl-acorn.ad::*tape* '()))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             ;; z = a * b, dz/da = b = 3, dz/db = a = 2
             (z (cl-acorn.ad::make-node 6.0d0
                                        (list (cons a 3.0d0)
                                              (cons b 2.0d0)))))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient a) 3.0d0))
        (ok (approx= (ad:node-gradient b) 2.0d0))))))

(deftest test-backward-chain
  (testing "backward propagates through chain: z = (a * b) + b"
    ;; dz/da = b = 3, dz/db = a + 1 = 3
    (let ((cl-acorn.ad::*tape* '()))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             (ab (cl-acorn.ad::make-node 6.0d0
                                         (list (cons a 3.0d0)
                                               (cons b 2.0d0))))
             (z (cl-acorn.ad::make-node 9.0d0
                                        (list (cons ab 1.0d0)
                                              (cons b 1.0d0)))))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient a) 3.0d0))
        (ok (approx= (ad:node-gradient b) 3.0d0))))))

(deftest test-backward-resets-gradients
  (testing "backward sets root gradient to 1 and accumulates correctly"
    (let ((cl-acorn.ad::*tape* '()))
      (let* ((a (cl-acorn.ad::make-node 5.0d0 nil))
             (z (cl-acorn.ad::make-node 5.0d0
                                        (list (cons a 1.0d0)))))
        (cl-acorn.ad::backward z)
        (ok (approx= (ad:node-gradient z) 1.0d0))
        (ok (approx= (ad:node-gradient a) 1.0d0))))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd
```

Expected: FAIL -- `tape-node` class does not exist yet.

**Step 3: Write minimal implementation**

Create `src/tape.lisp` using the Write tool for the initial file, then expand with `lisp-edit-form`:

```lisp
(in-package #:cl-acorn.ad)

(defvar *tape* nil
  "Dynamic computation graph for reverse-mode AD.
When non-nil, holds a list of tape-node instances in evaluation order
(most recent first). Bound dynamically by GRADIENT and related functions.")

(defclass tape-node ()
  ((value    :initarg :value
             :accessor node-value
             :documentation "Computed value at this node.")
   (gradient :initarg :gradient
             :accessor node-gradient
             :initform 0
             :documentation "Accumulated gradient during backward pass.")
   (children :initarg :children
             :accessor node-children
             :type list
             :initform nil
             :documentation "List of (child-node . local-partial-derivative) pairs."))
  (:documentation "A node in the computation graph for reverse-mode AD.
Each node records its value, its inputs (children), and the local partial
derivative with respect to each input. The backward pass uses these to
accumulate gradients via the chain rule."))

(defun make-node (value children)
  "Create a tape-node with VALUE and CHILDREN, pushing it onto *tape*.
CHILDREN is a list of (tape-node . local-partial) cons cells.
Returns the new node."
  (let ((node (make-instance 'tape-node
                             :value value
                             :children children)))
    (when *tape*
      (push node *tape*))
    node))

(defun backward (output-node)
  "Backpropagate gradients from OUTPUT-NODE through the tape.
Sets OUTPUT-NODE gradient to 1, then traverses *tape* in reverse
evaluation order (which is forward list order), accumulating gradients
via the chain rule.

All node gradients on the tape should be zero before calling this.
Uses generic addition and multiplication to support forward-over-reverse
composition (where gradients may be dual numbers)."
  (setf (node-gradient output-node) 1.0d0)
  (dolist (n *tape*)
    (let ((n-grad (node-gradient n)))
      (when (not (eql n-grad 0))
        (dolist (child-entry (node-children n))
          (let ((child (car child-entry))
                (local-grad (cdr child-entry)))
            (setf (node-gradient child)
                  (binary-add (node-gradient child)
                              (binary-mul n-grad local-grad)))))))))

(defmethod print-object ((n tape-node) stream)
  (print-unreadable-object (n stream :type t)
    (format stream "~A (grad: ~A)" (node-value n) (node-gradient n))))
```

Note: `backward` uses `binary-add` and `binary-mul` instead of `cl:+` and `cl:*`. This is essential for forward-over-reverse composition (Task 7), where gradients and local partials may be `dual` numbers.

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd
```

Expected: All tests pass (existing 60 + new tape tests).

**Step 5: Commit**

```bash
git add src/tape.lisp tests/tape-test.lisp
git commit -m "feat: add tape-node class, tape management, and backward pass"
```

---

### Task 3: Reverse-mode arithmetic methods

Add `tape-node` methods to the four binary arithmetic generic functions and the two unary generics.

**Files:**
- Create: `src/reverse-arithmetic.lisp`
- Test: `tests/reverse-arithmetic-test.lisp`

**Step 1: Write the failing test**

Create `tests/reverse-arithmetic-test.lisp` (Write tool -- uses `ad:` prefix):

```lisp
(defpackage #:cl-acorn/tests/reverse-arithmetic-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/reverse-arithmetic-test)

;;; Helper: compute reverse-mode gradient for f(a,b) and return (da db)
(defun reverse-grad-2 (fn a-val b-val)
  "Compute reverse-mode gradients of binary function FN at (A-VAL, B-VAL).
Returns (values f-val da db)."
  (let ((cl-acorn.ad::*tape* '()))
    (let* ((a (cl-acorn.ad::make-node (coerce a-val 'double-float) nil))
           (b (cl-acorn.ad::make-node (coerce b-val 'double-float) nil))
           (result (funcall fn a b)))
      (cl-acorn.ad::backward result)
      (values (ad:node-value result)
              (ad:node-gradient a)
              (ad:node-gradient b)))))

;;; Helper: compute reverse-mode gradient for f(x) and return dx
(defun reverse-grad-1 (fn x-val)
  "Compute reverse-mode gradient of unary function FN at X-VAL.
Returns (values f-val dx)."
  (let ((cl-acorn.ad::*tape* '()))
    (let* ((x (cl-acorn.ad::make-node (coerce x-val 'double-float) nil))
           (result (funcall fn x)))
      (cl-acorn.ad::backward result)
      (values (ad:node-value result)
              (ad:node-gradient x)))))

;;; --- Addition ---

(deftest test-reverse-add-tape-tape
  (testing "d/da(a+b)=1, d/db(a+b)=1"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:+ 2.0d0 3.0d0)
      (ok (approx= val 5.0d0))
      (ok (approx= da 1.0d0))
      (ok (approx= db 1.0d0)))))

(deftest test-reverse-add-tape-number
  (testing "d/da(a+5) = 1"
    (multiple-value-bind (val da)
        (reverse-grad-1 (lambda (a) (ad:+ a 5.0d0)) 3.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= da 1.0d0)))))

(deftest test-reverse-add-number-tape
  (testing "d/db(5+b) = 1"
    (multiple-value-bind (val db)
        (reverse-grad-1 (lambda (b) (ad:+ 5.0d0 b)) 3.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= db 1.0d0)))))

;;; --- Subtraction ---

(deftest test-reverse-sub-tape-tape
  (testing "d/da(a-b)=1, d/db(a-b)=-1"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:- 5.0d0 3.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= da 1.0d0))
      (ok (approx= db -1.0d0)))))

(deftest test-reverse-negate
  (testing "d/dx(-x) = -1"
    (multiple-value-bind (val dx)
        (reverse-grad-1 #'ad:- 3.0d0)
      (ok (approx= val -3.0d0))
      (ok (approx= dx -1.0d0)))))

;;; --- Multiplication ---

(deftest test-reverse-mul-tape-tape
  (testing "d/da(a*b)=b, d/db(a*b)=a"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:* 2.0d0 3.0d0)
      (ok (approx= val 6.0d0))
      (ok (approx= da 3.0d0))
      (ok (approx= db 2.0d0)))))

(deftest test-reverse-mul-tape-number
  (testing "d/da(a*5) = 5"
    (multiple-value-bind (val da)
        (reverse-grad-1 (lambda (a) (ad:* a 5.0d0)) 3.0d0)
      (ok (approx= val 15.0d0))
      (ok (approx= da 5.0d0)))))

(deftest test-reverse-mul-number-tape
  (testing "d/db(5*b) = 5"
    (multiple-value-bind (val db)
        (reverse-grad-1 (lambda (b) (ad:* 5.0d0 b)) 3.0d0)
      (ok (approx= val 15.0d0))
      (ok (approx= db 5.0d0)))))

;;; --- Division ---

(deftest test-reverse-div-tape-tape
  (testing "d/da(a/b)=1/b, d/db(a/b)=-a/b^2"
    (multiple-value-bind (val da db)
        (reverse-grad-2 #'ad:/ 6.0d0 3.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= da (/ 1.0d0 3.0d0)))
      (ok (approx= db (/ -6.0d0 9.0d0))))))

(deftest test-reverse-reciprocal
  (testing "d/dx(1/x) = -1/x^2"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:/ x)) 2.0d0)
      (ok (approx= val 0.5d0))
      (ok (approx= dx -0.25d0)))))

;;; --- N-ary wrappers ---

(deftest test-reverse-add-nary
  (testing "gradient through 3-arg addition"
    (let ((cl-acorn.ad::*tape* '()))
      (let* ((a (cl-acorn.ad::make-node 1.0d0 nil))
             (b (cl-acorn.ad::make-node 2.0d0 nil))
             (c (cl-acorn.ad::make-node 3.0d0 nil))
             (result (ad:+ a b c)))
        (cl-acorn.ad::backward result)
        (ok (approx= (ad:node-value result) 6.0d0))
        (ok (approx= (ad:node-gradient a) 1.0d0))
        (ok (approx= (ad:node-gradient b) 1.0d0))
        (ok (approx= (ad:node-gradient c) 1.0d0))))))

(deftest test-reverse-mul-nary
  (testing "gradient through 3-arg multiplication"
    (let ((cl-acorn.ad::*tape* '()))
      (let* ((a (cl-acorn.ad::make-node 2.0d0 nil))
             (b (cl-acorn.ad::make-node 3.0d0 nil))
             (c (cl-acorn.ad::make-node 4.0d0 nil))
             (result (ad:* a b c)))
        ;; d(abc)/da = bc = 12, d/db = ac = 8, d/dc = ab = 6
        (cl-acorn.ad::backward result)
        (ok (approx= (ad:node-value result) 24.0d0))
        (ok (approx= (ad:node-gradient a) 12.0d0))
        (ok (approx= (ad:node-gradient b) 8.0d0))
        (ok (approx= (ad:node-gradient c) 6.0d0))))))

;;; --- Cross-validation with forward-mode ---

(deftest test-reverse-matches-forward-polynomial
  (testing "reverse gradient of x^3-2x+5 matches forward derivative at x=3"
    (let ((fn (lambda (x) (ad:+ (ad:* x x x) (ad:+ (ad:* -2 x) 5)))))
      (multiple-value-bind (fwd-val fwd-grad)
          (ad:derivative fn 3.0d0)
        (multiple-value-bind (rev-val rev-grad)
            (reverse-grad-1 fn 3.0d0)
          (ok (approx= fwd-val rev-val))
          (ok (approx= fwd-grad rev-grad)))))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd
```

Expected: FAIL -- no `tape-node` methods for arithmetic.

**Step 3: Write minimal implementation**

Create `src/reverse-arithmetic.lisp`:

```lisp
(in-package #:cl-acorn.ad)

;;; --- Addition: tape-node methods ---

(defmethod binary-add ((a tape-node) (b tape-node))
  (make-node (cl:+ (node-value a) (node-value b))
             (list (cons a 1.0d0)
                   (cons b 1.0d0))))

(defmethod binary-add ((a tape-node) (b number))
  (make-node (cl:+ (node-value a) (coerce b 'double-float))
             (list (cons a 1.0d0))))

(defmethod binary-add ((a number) (b tape-node))
  (make-node (cl:+ (coerce a 'double-float) (node-value b))
             (list (cons b 1.0d0))))

;;; --- Subtraction: tape-node methods ---

(defmethod binary-sub ((a tape-node) (b tape-node))
  (make-node (cl:- (node-value a) (node-value b))
             (list (cons a 1.0d0)
                   (cons b -1.0d0))))

(defmethod binary-sub ((a tape-node) (b number))
  (make-node (cl:- (node-value a) (coerce b 'double-float))
             (list (cons a 1.0d0))))

(defmethod binary-sub ((a number) (b tape-node))
  (make-node (cl:- (coerce a 'double-float) (node-value b))
             (list (cons b -1.0d0))))

;;; --- Unary negation: tape-node method ---

(defmethod unary-negate ((a tape-node))
  (make-node (cl:- (node-value a))
             (list (cons a -1.0d0))))

;;; --- Multiplication: tape-node methods ---

(defmethod binary-mul ((a tape-node) (b tape-node))
  (make-node (cl:* (node-value a) (node-value b))
             (list (cons a (node-value b))
                   (cons b (node-value a)))))

(defmethod binary-mul ((a tape-node) (b number))
  (let ((bn (coerce b 'double-float)))
    (make-node (cl:* (node-value a) bn)
               (list (cons a bn)))))

(defmethod binary-mul ((a number) (b tape-node))
  (let ((an (coerce a 'double-float)))
    (make-node (cl:* an (node-value b))
               (list (cons b an)))))

;;; --- Division: tape-node methods ---

(defmethod binary-div ((a tape-node) (b tape-node))
  (let ((av (node-value a))
        (bv (node-value b)))
    (make-node (cl:/ av bv)
               (list (cons a (cl:/ 1.0d0 bv))
                     (cons b (cl:/ (cl:- av) (cl:* bv bv)))))))

(defmethod binary-div ((a tape-node) (b number))
  (let ((bn (coerce b 'double-float)))
    (make-node (cl:/ (node-value a) bn)
               (list (cons a (cl:/ 1.0d0 bn))))))

(defmethod binary-div ((a number) (b tape-node))
  (let ((an (coerce a 'double-float))
        (bv (node-value b)))
    (make-node (cl:/ an bv)
               (list (cons b (cl:/ (cl:- an) (cl:* bv bv)))))))

;;; --- Unary reciprocal: tape-node method ---

(defmethod unary-reciprocal ((a tape-node))
  (let ((v (node-value a)))
    (make-node (cl:/ v)
               (list (cons a (cl:/ -1.0d0 (cl:* v v)))))))
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd
```

Expected: All tests pass (existing 60 + tape tests + reverse arithmetic tests).

**Step 5: Commit**

```bash
git add src/reverse-arithmetic.lisp tests/reverse-arithmetic-test.lisp
git commit -m "feat: add reverse-mode tape-node methods for arithmetic"
```

---

### Task 4: Reverse-mode transcendental methods

Add `tape-node` methods for sin, cos, tan, exp, log, sqrt, abs, expt.

**Files:**
- Create: `src/reverse-transcendental.lisp`
- Test: `tests/reverse-transcendental-test.lisp`

**Step 1: Write the failing test**

Create `tests/reverse-transcendental-test.lisp` (Write tool):

```lisp
(defpackage #:cl-acorn/tests/reverse-transcendental-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/reverse-transcendental-test)

;;; Helper
(defun reverse-grad-1 (fn x-val)
  (let ((cl-acorn.ad::*tape* '()))
    (let* ((x (cl-acorn.ad::make-node (coerce x-val 'double-float) nil))
           (result (funcall fn x)))
      (cl-acorn.ad::backward result)
      (values (ad:node-value result)
              (ad:node-gradient x)))))

;;; Cross-validate each function: reverse gradient must match forward derivative

(deftest test-reverse-sin
  (testing "d/dx[sin(x)] = cos(x) at x=1"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:sin 1.0d0)
      (ok (approx= val (sin 1.0d0)))
      (ok (approx= dx (cos 1.0d0))))))

(deftest test-reverse-cos
  (testing "d/dx[cos(x)] = -sin(x) at x=1"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:cos 1.0d0)
      (ok (approx= val (cos 1.0d0)))
      (ok (approx= dx (- (sin 1.0d0)))))))

(deftest test-reverse-tan
  (testing "d/dx[tan(x)] = 1/cos^2(x) at x=0.5"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:tan 0.5d0)
      (ok (approx= val (tan 0.5d0)))
      (ok (approx= dx (/ 1.0d0 (expt (cos 0.5d0) 2)))))))

(deftest test-reverse-exp
  (testing "d/dx[exp(x)] = exp(x) at x=1"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:exp 1.0d0)
      (ok (approx= val (exp 1.0d0)))
      (ok (approx= dx (exp 1.0d0))))))

(deftest test-reverse-log
  (testing "d/dx[log(x)] = 1/x at x=2"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:log 2.0d0)
      (ok (approx= val (log 2.0d0)))
      (ok (approx= dx 0.5d0)))))

(deftest test-reverse-sqrt
  (testing "d/dx[sqrt(x)] = 1/(2*sqrt(x)) at x=4"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:sqrt 4.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= dx 0.25d0)))))

(deftest test-reverse-abs
  (testing "d/dx[abs(x)] = signum(x) at x=-3"
    (multiple-value-bind (val dx) (reverse-grad-1 #'ad:abs -3.0d0)
      (ok (approx= val 3.0d0))
      (ok (approx= dx -1.0d0)))))

(deftest test-reverse-expt-tape-number
  (testing "d/dx[x^3] = 3x^2 at x=2"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt x 3)) 2.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= dx 12.0d0)))))

(deftest test-reverse-expt-number-tape
  (testing "d/dx[2^x] = 2^x * ln(2) at x=3"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt 2 x)) 3.0d0)
      (ok (approx= val 8.0d0))
      (ok (approx= dx (* 8.0d0 (log 2.0d0)))))))

(deftest test-reverse-expt-tape-tape
  (testing "d/dx[x^x] at x=2 (uses exp(x*ln(x)) decomposition)"
    ;; f(x) = x^x, f'(x) = x^x * (1 + ln(x))
    ;; f(2) = 4, f'(2) = 4 * (1 + ln(2))
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:expt x x)) 2.0d0)
      (ok (approx= val 4.0d0))
      (ok (approx= dx (* 4.0d0 (+ 1.0d0 (log 2.0d0))))))))

(deftest test-reverse-log-with-base
  (testing "d/dx[log_10(x)] = 1/(x * ln(10)) at x=100"
    (multiple-value-bind (val dx)
        (reverse-grad-1 (lambda (x) (ad:log x 10)) 100.0d0)
      (ok (approx= val 2.0d0))
      (ok (approx= dx (/ 1.0d0 (* 100.0d0 (log 10.0d0))))))))

;;; Cross-validate composite function with forward-mode

(deftest test-reverse-composite-matches-forward
  (testing "reverse gradient of exp(sin(x)) matches forward derivative"
    (let ((fn (lambda (x) (ad:exp (ad:sin x)))))
      (multiple-value-bind (fwd-val fwd-grad) (ad:derivative fn 1.0d0)
        (multiple-value-bind (rev-val rev-grad) (reverse-grad-1 fn 1.0d0)
          (ok (approx= fwd-val rev-val))
          (ok (approx= fwd-grad rev-grad)))))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd
```

Expected: FAIL -- no `tape-node` methods for transcendental functions.

**Step 3: Write minimal implementation**

Create `src/reverse-transcendental.lisp`:

```lisp
(in-package #:cl-acorn.ad)

;;; --- sin ---

(defmethod ad-sin ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:sin v)
               (list (cons x (cl:cos v))))))

;;; --- cos ---

(defmethod ad-cos ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:cos v)
               (list (cons x (cl:- (cl:sin v)))))))

;;; --- tan ---

(defmethod ad-tan ((x tape-node))
  (let ((v (node-value x)))
    (let ((cos-v (cl:cos v)))
      (make-node (cl:tan v)
                 (list (cons x (cl:/ 1.0d0 (cl:* cos-v cos-v))))))))

;;; --- exp ---

(defmethod ad-exp ((x tape-node))
  (let ((exp-v (cl:exp (node-value x))))
    (make-node exp-v
               (list (cons x exp-v)))))

;;; --- log (unary) ---

(defmethod unary-log ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:log v)
               (list (cons x (cl:/ 1.0d0 v))))))

;;; --- sqrt ---

(defmethod ad-sqrt ((x tape-node))
  (let* ((v (node-value x))
         (sqrt-v (cl:sqrt v)))
    (make-node sqrt-v
               (list (cons x (cl:/ 1.0d0 (cl:* 2.0d0 sqrt-v)))))))

;;; --- abs ---

(defmethod ad-abs ((x tape-node))
  (let ((v (node-value x)))
    (make-node (cl:abs v)
               (list (cons x (coerce (cl:signum v) 'double-float))))))

;;; --- expt ---

(defmethod binary-expt ((base tape-node) (power number))
  (let ((bv (node-value base))
        (n (coerce power 'double-float)))
    (make-node (cl:expt bv n)
               (list (cons base (cl:* n (cl:expt bv (cl:- n 1.0d0))))))))

(defmethod binary-expt ((base number) (power tape-node))
  (let* ((c (coerce base 'double-float))
         (pv (node-value power))
         (cp (cl:expt c pv)))
    (make-node cp
               (list (cons power (cl:* cp (cl:log c)))))))

(defmethod binary-expt ((base tape-node) (power tape-node))
  ;; x^y = exp(y * ln(x)) -- reuses AD-aware exp, *, log
  (exp (* power (log base))))
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd
```

Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/reverse-transcendental.lisp tests/reverse-transcendental-test.lisp
git commit -m "feat: add reverse-mode tape-node methods for transcendental functions"
```

---

### Task 5: gradient function

The primary public API for reverse-mode AD.

**Files:**
- Create: `src/gradient.lisp`
- Test: `tests/gradient-test.lisp`

**Step 1: Write the failing test**

Create `tests/gradient-test.lisp` (Write tool):

```lisp
(defpackage #:cl-acorn/tests/gradient-test
  (:use #:cl #:rove #:cl-acorn/tests/util)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn/tests/gradient-test)

;;; --- gradient tests ---

(deftest test-gradient-sum
  (testing "gradient of f(x,y) = x + y is (1, 1)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:+ (first p) (second p)))
                     '(3.0d0 4.0d0))
      (ok (approx= val 7.0d0))
      (ok (approx= (first grad) 1.0d0))
      (ok (approx= (second grad) 1.0d0)))))

(deftest test-gradient-product
  (testing "gradient of f(x,y) = x * y is (y, x)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:* (first p) (second p)))
                     '(3.0d0 4.0d0))
      (ok (approx= val 12.0d0))
      (ok (approx= (first grad) 4.0d0))
      (ok (approx= (second grad) 3.0d0)))))

(deftest test-gradient-polynomial
  (testing "gradient of f(x,y) = x^2 + x*y at (3,4) is (2x+y, x) = (10, 3)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (let ((x (first p)) (y (second p)))
                         (ad:+ (ad:* x x) (ad:* x y))))
                     '(3.0d0 4.0d0))
      (ok (approx= val 21.0d0))
      (ok (approx= (first grad) 10.0d0))
      (ok (approx= (second grad) 3.0d0)))))

(deftest test-gradient-single-variable
  (testing "gradient of f(x) = x^2 at x=3 is (6)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p) (ad:* (first p) (first p)))
                     '(3.0d0))
      (ok (approx= val 9.0d0))
      (ok (approx= (first grad) 6.0d0)))))

(deftest test-gradient-three-variables
  (testing "gradient of f(x,y,z) = x*y*z at (2,3,4) is (yz,xz,xy)=(12,8,6)"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:* (first p) (second p) (third p)))
                     '(2.0d0 3.0d0 4.0d0))
      (ok (approx= val 24.0d0))
      (ok (approx= (first grad) 12.0d0))
      (ok (approx= (second grad) 8.0d0))
      (ok (approx= (third grad) 6.0d0)))))

(deftest test-gradient-transcendental
  (testing "gradient of f(x,y) = sin(x) * exp(y) at (1, 0)"
    ;; df/dx = cos(x)*exp(y) = cos(1)
    ;; df/dy = sin(x)*exp(y) = sin(1)
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (ad:* (ad:sin (first p)) (ad:exp (second p))))
                     '(1.0d0 0.0d0))
      (ok (approx= val (sin 1.0d0)))
      (ok (approx= (first grad) (cos 1.0d0)))
      (ok (approx= (second grad) (sin 1.0d0))))))

(deftest test-gradient-matches-derivative
  (testing "single-variable gradient matches forward-mode derivative"
    (let ((fn-grad (lambda (p) (ad:exp (ad:sin (first p)))))
          (fn-deriv (lambda (x) (ad:exp (ad:sin x)))))
      (multiple-value-bind (fwd-val fwd-grad) (ad:derivative fn-deriv 1.0d0)
        (multiple-value-bind (rev-val rev-grad) (ad:gradient fn-grad '(1.0d0))
          (ok (approx= fwd-val rev-val))
          (ok (approx= fwd-grad (first rev-grad))))))))

(deftest test-gradient-with-loop
  (testing "gradient propagates through iterative computation"
    ;; f(x,y) = sum_{i=0}^{9} (x*y) = 10*x*y
    ;; df/dx = 10y, df/dy = 10x
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p)
                       (let ((x (first p)) (y (second p))
                             (acc 0.0d0))
                         (dotimes (i 10)
                           (declare (ignore i))
                           (setf acc (ad:+ acc (ad:* x y))))
                         acc))
                     '(2.0d0 3.0d0))
      (ok (approx= val 60.0d0))
      (ok (approx= (first grad) 30.0d0))
      (ok (approx= (second grad) 20.0d0)))))

(deftest test-gradient-integer-params
  (testing "gradient accepts integer parameters"
    (multiple-value-bind (val grad)
        (ad:gradient (lambda (p) (ad:* (first p) (first p)))
                     '(3))
      (ok (approx= val 9.0d0))
      (ok (approx= (first grad) 6.0d0)))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd
```

Expected: FAIL -- `gradient` function does not exist.

**Step 3: Write minimal implementation**

Create `src/gradient.lisp`:

```lisp
(in-package #:cl-acorn.ad)

(defun gradient (fn params)
  "Compute the gradient of scalar function FN at PARAMS.
FN must accept a list of tape-node values and return a scalar tape-node
or number. PARAMS is a list of numbers.
Returns (values f(params) gradient-list) where gradient-list contains
df/dp_i for each parameter p_i, as double-floats."
  (let* ((*tape* (list t))  ; non-nil to enable recording; t is sentinel
         (input-nodes (mapcar (lambda (p)
                                (make-node (coerce p 'double-float) nil))
                              params))
         (output (funcall fn input-nodes)))
    (pop *tape*)  ; remove sentinel
    (etypecase output
      (tape-node
       (backward output)
       (values (node-value output)
               (mapcar (lambda (n) (coerce (node-gradient n) 'double-float))
                       input-nodes)))
      (number
       (values (coerce output 'double-float)
               (mapcar (constantly 0.0d0) input-nodes))))))
```

Note: `*tape*` is initialized with `(list t)` -- a non-nil list so that `make-node`'s `(when *tape* ...)` check activates recording. The sentinel `t` is popped before backward traversal. This avoids special-casing the tape initialization.

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd
```

Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/gradient.lisp tests/gradient-test.lisp
git commit -m "feat: add gradient function for reverse-mode AD"
```

---

### Task 6: Jacobian-vector product

Forward-mode based Jvp using multi-variable dual number seeding.

**Files:**
- Modify: `src/gradient.lisp`
- Modify: `tests/gradient-test.lisp`

**Step 1: Write the failing test**

Append to `tests/gradient-test.lisp`:

```lisp
;;; --- jacobian-vector-product tests ---

(deftest test-jvp-identity
  (testing "Jvp of f(x) = [x1, x2] with v = [1, 0] gives [1, 0]"
    (multiple-value-bind (val jvp)
        (ad:jacobian-vector-product
         (lambda (p) (list (first p) (second p)))
         '(3.0d0 4.0d0)
         '(1.0d0 0.0d0))
      (ok (approx= (first val) 3.0d0))
      (ok (approx= (second val) 4.0d0))
      (ok (approx= (first jvp) 1.0d0))
      (ok (approx= (second jvp) 0.0d0)))))

(deftest test-jvp-linear
  (testing "Jvp of f(x,y) = [2x+y, x-y] with v = [1,1]"
    ;; J = [[2, 1], [1, -1]], J*v = [3, 0]
    (multiple-value-bind (val jvp)
        (ad:jacobian-vector-product
         (lambda (p)
           (let ((x (first p)) (y (second p)))
             (list (ad:+ (ad:* 2 x) y)
                   (ad:- x y))))
         '(1.0d0 1.0d0)
         '(1.0d0 1.0d0))
      (ok (approx= (first val) 3.0d0))
      (ok (approx= (second val) 0.0d0))
      (ok (approx= (first jvp) 3.0d0))
      (ok (approx= (second jvp) 0.0d0)))))

(deftest test-jvp-scalar-function
  (testing "Jvp of scalar f(x,y) = x*y with v = [1,0] gives df/dx"
    (multiple-value-bind (val jvp)
        (ad:jacobian-vector-product
         (lambda (p) (list (ad:* (first p) (second p))))
         '(3.0d0 4.0d0)
         '(1.0d0 0.0d0))
      (ok (approx= (first val) 12.0d0))
      (ok (approx= (first jvp) 4.0d0)))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd
```

Expected: FAIL -- `jacobian-vector-product` not defined.

**Step 3: Write minimal implementation**

Add to `src/gradient.lisp` using `lisp-edit-form` with `insert_after` on the `gradient` defun:

```lisp
(defun jacobian-vector-product (fn params vector)
  "Compute J*v where J is the Jacobian of FN at PARAMS and v is VECTOR.
FN must accept a list of values and return a list of values (or a single value).
Uses forward-mode: seeds each parameter as dual(param, vector-component).
Returns (values f(params) J*v) where both are lists of double-floats."
  (let* ((dual-inputs (mapcar (lambda (p v)
                                (make-dual p v))
                              params vector))
         (outputs (funcall fn dual-inputs))
         (output-list (if (listp outputs) outputs (list outputs))))
    (values (mapcar (lambda (o)
                      (coerce (if (typep o 'dual) (dual-real o) o)
                              'double-float))
                    output-list)
            (mapcar (lambda (o)
                      (coerce (if (typep o 'dual) (dual-epsilon o) 0)
                              'double-float))
                    output-list))))
```

**Step 4: Run tests to verify they pass**

```bash
rove cl-acorn.asd
```

Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/gradient.lisp tests/gradient-test.lisp
git commit -m "feat: add jacobian-vector-product using forward-mode"
```

---

### Task 7: Hessian-vector product (forward-over-reverse)

The key compositional piece: run gradient computation with dual-number inputs.

**Files:**
- Modify: `src/tape.lisp` (generalize backward for dual arithmetic)
- Modify: `src/reverse-arithmetic.lisp` (accept dual values in node-value)
- Modify: `src/reverse-transcendental.lisp` (accept dual values in node-value)
- Modify: `src/gradient.lisp` (add hessian-vector-product)
- Modify: `tests/gradient-test.lisp` (add Hvp tests)

**Step 1: Write the failing test**

Append to `tests/gradient-test.lisp`:

```lisp
;;; --- hessian-vector-product tests ---

(deftest test-hvp-quadratic
  (testing "Hvp of f(x,y) = x^2 + x*y with v = [1, 0]"
    ;; H = [[2, 1], [1, 0]], H*[1,0] = [2, 1]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p)
           (let ((x (first p)) (y (second p)))
             (ad:+ (ad:* x x) (ad:* x y))))
         '(3.0d0 4.0d0)
         '(1.0d0 0.0d0))
      (ok (approx= (first grad) 10.0d0))   ; df/dx = 2x+y = 10
      (ok (approx= (second grad) 3.0d0))   ; df/dy = x = 3
      (ok (approx= (first hvp) 2.0d0))     ; d2f/dx2*1 + d2f/dxdy*0 = 2
      (ok (approx= (second hvp) 1.0d0))))) ; d2f/dydx*1 + d2f/dy2*0 = 1

(deftest test-hvp-cubic
  (testing "Hvp of f(x) = x^3 with v = [1]"
    ;; f'(x) = 3x^2, f''(x) = 6x
    ;; At x=2: grad = [12], Hvp = [12]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p) (ad:* (first p) (first p) (first p)))
         '(2.0d0)
         '(1.0d0))
      (ok (approx= (first grad) 12.0d0))
      (ok (approx= (first hvp) 12.0d0)))))

(deftest test-hvp-sin
  (testing "Hvp of f(x) = sin(x) with v = [1]"
    ;; f'(x) = cos(x), f''(x) = -sin(x)
    ;; At x=1: grad = [cos(1)], Hvp = [-sin(1)]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p) (ad:sin (first p)))
         '(1.0d0)
         '(1.0d0))
      (ok (approx= (first grad) (cos 1.0d0)))
      (ok (approx= (first hvp) (- (sin 1.0d0)))))))

(deftest test-hvp-multivariate
  (testing "Hvp of f(x,y) = sin(x)*y^2 with v = [1, 1]"
    ;; df/dx = cos(x)*y^2, df/dy = 2*sin(x)*y
    ;; H = [[−sin(x)*y^2, 2*cos(x)*y],
    ;;      [2*cos(x)*y,   2*sin(x)  ]]
    ;; At (1, 2): H = [[-sin(1)*4, 2*cos(1)*2], [2*cos(1)*2, 2*sin(1)]]
    ;; H*[1,1] = [-sin(1)*4+4*cos(1), 4*cos(1)+2*sin(1)]
    (multiple-value-bind (grad hvp)
        (ad:hessian-vector-product
         (lambda (p)
           (ad:* (ad:sin (first p)) (ad:* (second p) (second p))))
         '(1.0d0 2.0d0)
         '(1.0d0 1.0d0))
      (ok (approx= (first grad) (* (cos 1.0d0) 4.0d0)))
      (ok (approx= (second grad) (* 2.0d0 (sin 1.0d0) 2.0d0)))
      (ok (approx= (first hvp)
                    (+ (* -1.0d0 (sin 1.0d0) 4.0d0)
                       (* 4.0d0 (cos 1.0d0)))))
      (ok (approx= (second hvp)
                    (+ (* 4.0d0 (cos 1.0d0))
                       (* 2.0d0 (sin 1.0d0))))))))
```

**Step 2: Run test to verify it fails**

```bash
rove cl-acorn.asd
```

Expected: FAIL -- `hessian-vector-product` not defined.

**Step 3: Generalize reverse-mode arithmetic for dual values**

The key insight: for forward-over-reverse, `node-value` and local partials may be `dual` numbers instead of `double-float`. The existing tape methods use `cl:+`, `cl:*` etc. on node values, which won't work with duals.

**3a: Update `src/reverse-arithmetic.lisp`** -- Replace `cl:` calls on `node-value` with `binary-add`/`binary-mul` etc. so they dispatch correctly for both `number` and `dual`:

Use `lisp-edit-form` to replace each `tape-node` method. The pattern for each method:
- Replace `(cl:+ (node-value a) (node-value b))` with calls that handle both number and dual
- Replace `(cl:* ...)`, `(cl:/ ...)`, `(cl:- ...)` similarly
- Local partials computed from node values must also handle duals

Example -- `binary-mul ((a tape-node) (b tape-node))`:

```lisp
(defmethod binary-mul ((a tape-node) (b tape-node))
  (make-node (binary-mul (node-value a) (node-value b))
             (list (cons a (node-value b))
                   (cons b (node-value a)))))
```

This recursively calls `binary-mul` on the values, which are either `number` (dispatches to `(number number)` method) or `dual` (dispatches to dual method). Same pattern for all methods.

Apply this pattern to all 12 tape-node arithmetic methods:
- `binary-add` (3 methods): value computation uses `binary-add` on node-values
- `binary-sub` (3 methods): value computation uses `binary-sub` on node-values
- `unary-negate` (1 method): uses `unary-negate` on node-value
- `binary-mul` (3 methods): uses `binary-mul` on node-values; local partials are node-values directly
- `binary-div` (3 methods): uses `binary-div` on node-values; local partials use `binary-div`/`binary-mul`
- `unary-reciprocal` (1 method): uses `binary-div` for local partial

**3b: Update `src/reverse-transcendental.lisp`** -- Same pattern. Replace `cl:sin`, `cl:cos` etc. with `ad-sin`, `ad-cos` etc. on `node-value`:

Example -- `ad-sin ((x tape-node))`:

```lisp
(defmethod ad-sin ((x tape-node))
  (let ((v (node-value x)))
    (make-node (ad-sin v)
               (list (cons x (ad-cos v))))))
```

Apply to all transcendental tape-node methods (8 methods + 3 expt methods).

**3c: backward already uses `binary-add` and `binary-mul`** -- no changes needed (this was anticipated in Task 2).

**Step 4: Implement hessian-vector-product**

Add to `src/gradient.lisp` using `lisp-edit-form` with `insert_after` on `jacobian-vector-product`:

```lisp
(defun hessian-vector-product (fn params vector)
  "Compute H*v where H is the Hessian of scalar FN at PARAMS and v is VECTOR.
Uses forward-over-reverse: seeds dual numbers into reverse-mode gradient
computation. The epsilon components of the resulting gradient entries
give the Hessian-vector product.
Returns (values gradient-list hvp-list) as lists of double-floats."
  (let* ((dual-params (mapcar (lambda (p v)
                                (make-dual p v))
                              params vector))
         ;; Run gradient with dual-number parameters
         ;; This creates tape-nodes whose values are dual numbers
         (*tape* (list t))
         (input-nodes (mapcar (lambda (dp)
                                (make-node dp nil))
                              dual-params))
         (output (funcall fn input-nodes)))
    (pop *tape*)
    (etypecase output
      (tape-node
       (backward output)
       (values (mapcar (lambda (n)
                         (let ((g (node-gradient n)))
                           (coerce (if (typep g 'dual) (dual-real g) g)
                                   'double-float)))
                       input-nodes)
               (mapcar (lambda (n)
                         (let ((g (node-gradient n)))
                           (coerce (if (typep g 'dual) (dual-epsilon g) 0)
                                   'double-float)))
                       input-nodes)))
      (number
       (values (mapcar (constantly 0.0d0) params)
               (mapcar (constantly 0.0d0) params))))))
```

**Step 5: Run tests to verify they pass**

```bash
rove cl-acorn.asd
```

Expected: All tests pass. Pay special attention to:
- Existing forward-mode tests (60 tests) still pass
- Existing reverse-mode tests still pass
- New Hvp tests pass

**Step 6: Commit**

```bash
git add src/tape.lisp src/reverse-arithmetic.lisp src/reverse-transcendental.lisp src/gradient.lisp tests/gradient-test.lisp
git commit -m "feat: add hessian-vector-product via forward-over-reverse"
```

---

### Task 8: Reverse-mode neural network example

Rewrite the MLP example using `ad:gradient` to demonstrate the performance advantage.

**Files:**
- Create: `examples/08-reverse-neural-network/main.lisp`
- Create: `examples/08-reverse-neural-network/README.md`

**Step 1: Create the example**

Create `examples/08-reverse-neural-network/main.lisp`. This reuses the data and network architecture from example 02, but replaces the training loop:

**Old (forward-mode)**: 67 separate `ad:derivative` calls per epoch, each computing one partial derivative.

**New (reverse-mode)**: 1 `ad:gradient` call per epoch, computing all 67 partial derivatives at once.

Key changes from example 02:
- `train` function uses `ad:gradient` instead of per-parameter `ad:derivative`
- `total-loss` takes a parameter list (already tape-node aware via existing `ad:` ops)
- No need to `copy-list` params and swap one parameter at a time
- Can use the full 150-sample dataset since reverse-mode is much faster

The `forward-pass`, `sigmoid`, `softmax-cross-entropy`, `total-loss` functions work unchanged because they use `ad:+`, `ad:*` etc. which dispatch to tape-node methods automatically.

Write the full example file. The `train` function body should be roughly:

```lisp
(defun train (features labels &key (lr 0.5d0) (epochs 100))
  (let ((params (init-params)))
    (dotimes (epoch epochs)
      (multiple-value-bind (loss grads)
          (ad:gradient
           (lambda (p) (total-loss p features labels))
           params)
        ;; Update parameters
        (setf params
              (mapcar (lambda (p g) (- p (* lr g)))
                      params grads))
        ;; Report every 10 epochs
        (when (zerop (mod epoch 10))
          (format t "  Epoch ~3D  |  Loss ~8,4F  |  Accuracy ~5,1F%~%"
                  epoch loss
                  (* 100.0d0 (accuracy params features labels))))))
    params))
```

**Step 2: Create README.md**

Create `examples/08-reverse-neural-network/README.md` explaining the reverse-mode approach and performance comparison.

**Step 3: Test the example runs**

```bash
sbcl --load examples/08-reverse-neural-network/main.lisp
```

Expected: Training runs, loss decreases, accuracy increases. Should complete much faster than example 02.

**Step 4: Commit**

```bash
git add examples/08-reverse-neural-network/
git commit -m "feat: add reverse-mode neural network example"
```

---

### Task 9: Run full regression suite and update README

Final verification and documentation.

**Files:**
- Modify: `README.md`

**Step 1: Run all tests**

```bash
rove cl-acorn.asd
```

Expected: All tests pass (60 existing + ~40 new reverse-mode tests).

**Step 2: Run all examples to verify no regressions**

```bash
sbcl --load examples/01-curve-fitting/main.lisp
sbcl --load examples/03-newton-method/main.lisp
sbcl --load examples/08-reverse-neural-network/main.lisp
```

**Step 3: Update README.md**

Add reverse-mode AD section to README.md:
- Add `gradient`, `jacobian-vector-product`, `hessian-vector-product` to API reference
- Add `tape-node`, `node-value`, `node-gradient` to data structures
- Add usage example for `gradient`
- Add the new example to the examples table
- Update description from "Forward-mode" to "Forward and reverse-mode"

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for reverse-mode AD"
```

**Step 5: Push**

```bash
git push
```
