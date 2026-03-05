# cl-acorn Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 7 practical, E2E-runnable examples demonstrating cl-acorn's forward-mode AD across diverse domains.

**Architecture:** Each example lives in `examples/NN-name/` with a dedicated `defpackage` using `:local-nicknames (#:ad #:cl-acorn.ad)`. Every `main.lisp` is self-contained: load cl-acorn, define functions, run and print results. No ASDF system for examples.

**Tech Stack:** Common Lisp, cl-acorn (forward-mode AD), SBCL 2.x (package-local-nicknames), Rove (validation tests)

**Key pattern for all examples:** Each `main.lisp` starts with:
```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.NAME
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.NAME)
```

**Multi-variable differentiation pattern** (used in tasks 1, 2, 6, 7):
```lisp
;; Differentiate loss w.r.t. w, holding b fixed
(ad:derivative (lambda (w) (loss w b data)) current-w)
```

---

### Task 1: Newton-Raphson Root Finding (03-newton-method)

Start here because it's the simplest example — pure `derivative` usage, no optimization loop, no data.

**Files:**
- Create: `examples/03-newton-method/main.lisp`
- Create: `examples/03-newton-method/README.md`

**Step 1: Write main.lisp**

```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.newton-method
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.newton-method)

;;; Newton-Raphson Method Using Automatic Differentiation
;;;
;;; Given f(x) = 0, iterate: x_{n+1} = x_n - f(x_n) / f'(x_n)
;;; AD computes both f(x) and f'(x) in a single call to ad:derivative.

(defun newton-raphson (f x0 &key (tolerance 1d-12) (max-iter 50))
  "Find root of F starting from X0 using Newton-Raphson with AD.
Returns (values root iterations-used)."
  (let ((x x0))
    (format t "~&~3A  ~20A  ~20A  ~20A~%" "n" "x_n" "f(x_n)" "f'(x_n)")
    (format t "~A~%" (make-string 66 :initial-element #\-))
    (dotimes (i max-iter (values x max-iter))
      (multiple-value-bind (fx dfx) (ad:derivative f (coerce x 'double-float))
        (format t "~3D  ~20,15F  ~20,6E  ~20,6E~%" i x fx dfx)
        (when (< (abs fx) tolerance)
          (return (values x i)))
        (when (< (abs dfx) 1d-15)
          (error "Derivative near zero at x=~A, Newton step undefined." x))
        (setf x (- x (/ fx dfx)))))))

(defun run-examples ()
  ;; Problem 1: x^3 - 2x - 5 = 0  (classical, root ~ 2.0946)
  (format t "~&=== Problem 1: x^3 - 2x - 5 = 0 ===~%~%")
  (multiple-value-bind (root iters)
      (newton-raphson (lambda (x) (ad:- (ad:- (ad:expt x 3) (ad:* 2 x)) 5))
                      2.0d0)
    (format t "~%Root: ~,15F  (after ~D iterations)~%" root iters)
    (format t "Verification: f(root) = ~E~%~%"
            (- (expt root 3) (* 2 root) 5)))

  ;; Problem 2: cos(x) - x = 0  (transcendental, root ~ 0.7391)
  (format t "~&=== Problem 2: cos(x) - x = 0 ===~%~%")
  (multiple-value-bind (root iters)
      (newton-raphson (lambda (x) (ad:- (ad:cos x) x))
                      1.0d0)
    (format t "~%Root: ~,15F  (after ~D iterations)~%" root iters)
    (format t "Verification: cos(root) - root = ~E~%"
            (- (cos root) root))))

(run-examples)
```

**Step 2: Test by loading in REPL**

Run: `(load "examples/03-newton-method/main.lisp")` via `repl-eval`
Expected: Iteration tables showing quadratic convergence, roots matching known values (2.09455... and 0.73908...)

**Step 3: Write README.md**

```markdown
# Newton-Raphson Root Finding

## What This Demonstrates

The simplest use of automatic differentiation: computing f(x) and f'(x) simultaneously for Newton-Raphson iteration.

## Theory

Given equation f(x) = 0, Newton's method iterates:

    x_{n+1} = x_n - f(x_n) / f'(x_n)

Traditionally you must derive f'(x) by hand. With AD, `ad:derivative` returns both values:

```lisp
(multiple-value-bind (fx dfx) (ad:derivative f x)
  (- x (/ fx dfx)))
```

## Problems Solved

1. **x^3 - 2x - 5 = 0** — Classical polynomial (root ≈ 2.0946)
2. **cos(x) - x = 0** — Transcendental equation (root ≈ 0.7391)

## Running

```lisp
(load "examples/03-newton-method/main.lisp")
```

## Key Takeaway

AD eliminates manual derivative computation. Define f(x) once; get f'(x) for free.
```

**Step 4: Commit**

```bash
git add examples/03-newton-method/
git commit -m "Add Newton-Raphson root finding example"
```

---

### Task 2: Parameter Sensitivity Analysis (04-sensitivity)

Second simplest — direct `derivative` calls on physical models, no optimization loop.

**Files:**
- Create: `examples/04-sensitivity/main.lisp`
- Create: `examples/04-sensitivity/README.md`

**Step 1: Write main.lisp**

```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.sensitivity
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.sensitivity)

;;; Parameter Sensitivity Analysis Using Automatic Differentiation
;;;
;;; Sensitivity = dOutput/dParameter tells how much the output changes
;;; per unit change in a parameter. AD computes this exactly.

(defconstant +g+ 9.80665d0 "Gravitational acceleration (m/s^2).")

;;; --- Model 1: Simple Pendulum Period ---
;;; T(L) = 2*pi*sqrt(L/g)
;;; Analytical: dT/dL = pi/sqrt(L*g)

(defun pendulum-period (length)
  "Period of a simple pendulum as a function of arm length."
  (ad:* 2.0d0 (ad:* pi (ad:sqrt (ad:/ length +g+)))))

(defun analytical-dt-dl (length)
  "Analytical derivative dT/dL = pi / sqrt(L*g)."
  (/ pi (sqrt (* length +g+))))

(defun sensitivity-table (f analytical-df name param-name values)
  "Print sensitivity comparison table for function F over VALUES."
  (format t "~&=== ~A ===~%~%" name)
  (format t "~10A  ~15A  ~15A  ~15A  ~12A~%"
          param-name "f(x)" "df/dx (AD)" "df/dx (exact)" "error")
  (format t "~A~%" (make-string 70 :initial-element #\-))
  (dolist (x values)
    (multiple-value-bind (val grad) (ad:derivative f (coerce x 'double-float))
      (let ((exact (funcall analytical-df x)))
        (format t "~10,3F  ~15,8F  ~15,8E  ~15,8E  ~12,2E~%"
                x val grad exact (abs (- grad exact)))))))

;;; --- Model 2: Damped Oscillation ---
;;; x(t; gamma) = A * exp(-gamma*t) * cos(omega*t)
;;; Fix A=1, omega=2*pi, t=values; vary gamma

(defparameter *amplitude* 1.0d0)
(defparameter *omega* (* 2.0d0 pi))

(defun damped-oscillation (gamma time)
  "Displacement of damped oscillator at TIME for damping coefficient GAMMA."
  (ad:* *amplitude*
        (ad:* (ad:exp (ad:* (ad:- gamma) time))
              (ad:cos (ad:* *omega* time)))))

(defun analytical-dx-dgamma (gamma time)
  "Analytical: dx/dgamma = -t * A * exp(-gamma*t) * cos(omega*t)."
  (* (- time) *amplitude*
     (exp (- (* gamma time)))
     (cos (* *omega* time))))

(defun run-examples ()
  ;; Pendulum sensitivity
  (sensitivity-table
   #'pendulum-period #'analytical-dt-dl
   "Pendulum Period T(L) = 2*pi*sqrt(L/g)" "L (m)"
   '(0.25d0 0.5d0 1.0d0 1.5d0 2.0d0 3.0d0 5.0d0))

  (format t "~%")

  ;; Damped oscillation sensitivity to gamma at various times
  (let ((gamma 0.5d0))
    (format t "~&=== Damped Oscillation: dx/d(gamma) at gamma=~A ===~%~%" gamma)
    (format t "~10A  ~15A  ~15A  ~15A  ~12A~%"
            "t (s)" "x(t)" "dx/dg (AD)" "dx/dg (exact)" "error")
    (format t "~A~%" (make-string 70 :initial-element #\-))
    (dolist (time '(0.1d0 0.25d0 0.5d0 1.0d0 1.5d0 2.0d0 3.0d0))
      (multiple-value-bind (val grad)
          (ad:derivative (lambda (g) (damped-oscillation g time)) gamma)
        (let ((exact (analytical-dx-dgamma gamma time)))
          (format t "~10,3F  ~15,8F  ~15,8E  ~15,8E  ~12,2E~%"
                  time val grad exact (abs (- grad exact))))))))

(run-examples)
```

**Step 2: Test by loading in REPL**

Run: `(load "examples/04-sensitivity/main.lisp")` via `repl-eval`
Expected: Tables showing AD derivatives matching analytical values to machine precision (~1e-15 error)

**Step 3: Write README.md**

```markdown
# Parameter Sensitivity Analysis

## What This Demonstrates

Using AD to compute how sensitive a model's output is to changes in its parameters — the foundation of "what-if" analysis.

## Theory

Sensitivity of output y to parameter p is simply dy/dp. With AD:

```lisp
(ad:derivative (lambda (p) (model p other-params)) p-value)
;; Returns (values y dy/dp)
```

## Models

1. **Simple Pendulum**: T(L) = 2pi*sqrt(L/g) — How does period change with arm length?
2. **Damped Oscillation**: x(t;gamma) = A*exp(-gamma*t)*cos(omega*t) — How does displacement respond to damping?

Both have closed-form derivatives for verification.

## Running

```lisp
(load "examples/04-sensitivity/main.lisp")
```

## Key Takeaway

AD gives exact sensitivities for any differentiable model, even when analytical derivatives are tedious or unavailable.
```

**Step 4: Commit**

```bash
git add examples/04-sensitivity/
git commit -m "Add parameter sensitivity analysis example"
```

---

### Task 3: Black-Scholes Greeks (05-black-scholes)

Medium complexity. Introduces nested `derivative` for 2nd-order derivatives and a non-trivial utility (normal CDF).

**Files:**
- Create: `examples/05-black-scholes/main.lisp`
- Create: `examples/05-black-scholes/README.md`

**Step 1: Write main.lisp**

```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.black-scholes
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.black-scholes)

;;; Black-Scholes Option Greeks via Automatic Differentiation
;;;
;;; Greeks are partial derivatives of the option price w.r.t. market parameters.
;;; AD computes them without deriving closed-form formulas.

;;; --- Normal distribution utilities ---

(defun norm-pdf (x)
  "Standard normal PDF: (1/sqrt(2*pi)) * exp(-x^2/2)."
  (ad:* (ad:/ 1.0d0 (ad:sqrt (ad:* 2.0d0 pi)))
        (ad:exp (ad:* -0.5d0 (ad:* x x)))))

(defun norm-cdf (x)
  "Standard normal CDF via Abramowitz & Stegun rational approximation (7.1.26).
Accurate to ~1.5e-7. Works with dual numbers."
  (let* ((sign (if (typep x 'ad:dual)
                   (if (minusp (ad:dual-real x)) -1.0d0 1.0d0)
                   (if (minusp x) -1.0d0 1.0d0)))
         (x-abs (ad:abs x))
         (p  0.2316419d0)
         (b1 0.319381530d0)
         (b2 -0.356563782d0)
         (b3 1.781477937d0)
         (b4 -1.821255978d0)
         (b5 1.330274429d0)
         (t-val (ad:/ 1.0d0 (ad:+ 1.0d0 (ad:* p x-abs))))
         (poly (ad:* t-val
                     (ad:+ b1
                           (ad:* t-val
                                 (ad:+ b2
                                       (ad:* t-val
                                             (ad:+ b3
                                                   (ad:* t-val
                                                         (ad:+ b4
                                                               (ad:* b5 t-val))))))))))
         (cdf-positive (ad:- 1.0d0 (ad:* (norm-pdf x-abs) poly))))
    (if (= sign 1.0d0)
        cdf-positive
        (ad:- 1.0d0 cdf-positive))))

;;; --- Black-Scholes formula ---

(defun bs-call-price (spot strike rate vol time-to-expiry)
  "Black-Scholes European call option price.
All arguments may be dual numbers for AD."
  (let* ((d1 (ad:/ (ad:+ (ad:log (ad:/ spot strike))
                         (ad:* (ad:+ rate (ad:/ (ad:* vol vol) 2.0d0))
                               time-to-expiry))
                   (ad:* vol (ad:sqrt time-to-expiry))))
         (d2 (ad:- d1 (ad:* vol (ad:sqrt time-to-expiry)))))
    (ad:- (ad:* spot (norm-cdf d1))
          (ad:* (ad:* strike (ad:exp (ad:* (ad:- rate) time-to-expiry)))
                (norm-cdf d2)))))

;;; --- Analytical Greeks for validation ---

(defun bs-d1 (s k r v tt)
  (/ (+ (log (/ s k)) (* (+ r (/ (* v v) 2.0d0)) tt))
     (* v (sqrt tt))))

(defun bs-d2 (s k r v tt)
  (- (bs-d1 s k r v tt) (* v (sqrt tt))))

(defun standard-norm-pdf (x)
  (* (/ 1.0d0 (sqrt (* 2.0d0 pi))) (exp (* -0.5d0 x x))))

(defun analytical-delta (s k r v tt)
  "Analytical N(d1)."
  (let ((d1 (bs-d1 s k r v tt)))
    ;; Simple numerical CDF for validation
    (norm-cdf-numerical d1)))

(defun norm-cdf-numerical (x)
  "Numerical normal CDF for validation (non-AD)."
  (let* ((sign (if (minusp x) -1.0d0 1.0d0))
         (x-abs (abs x))
         (p 0.2316419d0)
         (b1 0.319381530d0) (b2 -0.356563782d0)
         (b3 1.781477937d0) (b4 -1.821255978d0) (b5 1.330274429d0)
         (t-val (/ 1.0d0 (+ 1.0d0 (* p x-abs))))
         (poly (* t-val (+ b1 (* t-val (+ b2 (* t-val (+ b3 (* t-val (+ b4 (* b5 t-val))))))))))
         (cdf-pos (- 1.0d0 (* (standard-norm-pdf x-abs) poly))))
    (if (= sign 1.0d0) cdf-pos (- 1.0d0 cdf-pos))))

(defun analytical-gamma-greek (s k r v tt)
  "Analytical Gamma = N'(d1) / (S * v * sqrt(T))."
  (let ((d1 (bs-d1 s k r v tt)))
    (/ (standard-norm-pdf d1) (* s v (sqrt tt)))))

(defun analytical-vega (s k r v tt)
  "Analytical Vega = S * N'(d1) * sqrt(T)."
  (let ((d1 (bs-d1 s k r v tt)))
    (* s (standard-norm-pdf d1) (sqrt tt))))

(defun analytical-rho (s k r v tt)
  "Analytical Rho = K * T * exp(-r*T) * N(d2)."
  (* k tt (exp (- (* r tt))) (norm-cdf-numerical (bs-d2 s k r v tt))))

;;; --- Compute Greeks via AD ---

(defun compute-greeks (s k r v tt)
  "Compute all Greeks via AD and compare with analytical values."
  (format t "~&=== Black-Scholes Greeks ===~%")
  (format t "S=~A  K=~A  r=~A  vol=~A  T=~A~%~%" s k r v tt)

  (let ((price (bs-call-price s k r v tt)))
    ;; price is a plain number here (no duals in input)
    (format t "Call Price: ~,6F~%~%" (if (typep price 'ad:dual) (ad:dual-real price) price)))

  (format t "~20A  ~15A  ~15A  ~12A~%"
          "Greek" "AD" "Analytical" "Error")
  (format t "~A~%" (make-string 65 :initial-element #\-))

  ;; Delta = dC/dS
  (multiple-value-bind (price delta)
      (ad:derivative (lambda (s) (bs-call-price s k r v tt)) s)
    (declare (ignore price))
    (let ((exact (analytical-delta s k r v tt)))
      (format t "~20A  ~15,8F  ~15,8F  ~12,2E~%"
              "Delta (dC/dS)" delta exact (abs (- delta exact)))))

  ;; Gamma = d2C/dS2 — nested derivative!
  (multiple-value-bind (delta gamma)
      (ad:derivative
       (lambda (s)
         ;; Inner derivative returns f'(s), which is delta
         ;; Outer derivative differentiates delta w.r.t. s = gamma
         (multiple-value-bind (price delta)
             (ad:derivative (lambda (s2) (bs-call-price s2 k r v tt)) s)
           (declare (ignore price))
           delta))
       s)
    (declare (ignore delta))
    (let ((exact (analytical-gamma-greek s k r v tt)))
      (format t "~20A  ~15,8F  ~15,8F  ~12,2E~%"
              "Gamma (d2C/dS2)" gamma exact (abs (- gamma exact)))))

  ;; Vega = dC/dvol
  (multiple-value-bind (price vega)
      (ad:derivative (lambda (v) (bs-call-price s k r v tt)) v)
    (declare (ignore price))
    (let ((exact (analytical-vega s k r v tt)))
      (format t "~20A  ~15,8F  ~15,8F  ~12,2E~%"
              "Vega (dC/dvol)" vega exact (abs (- vega exact)))))

  ;; Theta = -dC/dT  (negative by convention)
  (multiple-value-bind (price dc-dt)
      (ad:derivative (lambda (tt) (bs-call-price s k r v tt)) tt)
    (declare (ignore price))
    (format t "~20A  ~15,8F  ~15A  ~12A~%"
            "Theta (-dC/dT)" (- dc-dt) "—" "—"))

  ;; Rho = dC/dr
  (multiple-value-bind (price rho)
      (ad:derivative (lambda (r) (bs-call-price s k r v tt)) r)
    (declare (ignore price))
    (let ((exact (analytical-rho s k r v tt)))
      (format t "~20A  ~15,8F  ~15,8F  ~12,2E~%"
              "Rho (dC/dr)" rho exact (abs (- rho exact))))))

(defun run-examples ()
  ;; Standard parameters
  (compute-greeks 100.0d0 100.0d0 0.05d0 0.2d0 1.0d0)

  (format t "~%~%=== Delta vs Spot Price ===~%~%")
  (format t "~10A  ~15A~%" "Spot" "Delta")
  (format t "~A~%" (make-string 28 :initial-element #\-))
  (dolist (s '(80.0d0 85.0d0 90.0d0 95.0d0 100.0d0 105.0d0 110.0d0 115.0d0 120.0d0))
    (multiple-value-bind (price delta)
        (ad:derivative (lambda (s) (bs-call-price s 100.0d0 0.05d0 0.2d0 1.0d0)) s)
      (declare (ignore price))
      (format t "~10,2F  ~15,6F~%" s delta))))

(run-examples)
```

**Step 2: Test by loading in REPL**

Run: `(load "examples/05-black-scholes/main.lisp")` via `repl-eval`
Expected: Greeks table with AD vs analytical values matching within ~1e-5 (limited by CDF approximation accuracy). Delta approaching 0 for low spot, 1 for high spot.

**Step 3: Write README.md**

```markdown
# Black-Scholes Option Greeks

## What This Demonstrates

Computing option price sensitivities ("Greeks") using AD, including **nested derivatives** for second-order Greeks (Gamma).

## Theory

The Black-Scholes formula prices European call options. Greeks are partial derivatives:

| Greek | Definition | Order |
|-------|-----------|-------|
| Delta | dC/dS | 1st |
| Gamma | d2C/dS2 | 2nd |
| Vega | dC/dvol | 1st |
| Theta | -dC/dT | 1st |
| Rho | dC/dr | 1st |

With AD, each Greek is a single `ad:derivative` call. Gamma uses **nested** derivatives:

```lisp
;; Gamma = d/dS of Delta = d/dS of (d/dS of C)
(ad:derivative
 (lambda (s)
   (multiple-value-bind (price delta)
       (ad:derivative (lambda (s2) (bs-call-price s2 ...)) s)
     delta))
 s)
```

## Running

```lisp
(load "examples/05-black-scholes/main.lisp")
```

## Key Takeaway

AD handles first and higher-order derivatives uniformly. Nested `ad:derivative` gives second-order derivatives without deriving them by hand.
```

**Step 4: Commit**

```bash
git add examples/05-black-scholes/
git commit -m "Add Black-Scholes option Greeks example"
```

---

### Task 4: Linear Regression / Curve Fitting (01-curve-fitting)

First optimization-loop example. Introduces the multi-variable pattern and uses external data.

**Files:**
- Create: `examples/01-curve-fitting/main.lisp`
- Create: `examples/01-curve-fitting/data.lisp`
- Create: `examples/01-curve-fitting/README.md`

**Step 1: Write data.lisp**

Iris sepal-length → sepal-width (150 data points). Generate from the well-known UCI Iris dataset.

```lisp
(in-package #:cl-acorn.examples.curve-fitting)

;;; Iris Dataset: sepal-length (x) and sepal-width (y)
;;; Source: UCI Machine Learning Repository (Fisher, 1936)

(defparameter *iris-x*
  (make-array 150 :element-type 'double-float
              :initial-contents
              '(5.1d0 4.9d0 4.7d0 4.6d0 5.0d0 5.4d0 4.6d0 5.0d0 4.4d0 4.9d0
                5.4d0 4.8d0 4.8d0 4.3d0 5.8d0 5.7d0 5.4d0 5.1d0 5.7d0 5.1d0
                5.4d0 5.1d0 4.6d0 5.1d0 4.8d0 5.0d0 5.0d0 5.2d0 5.2d0 4.7d0
                4.8d0 5.4d0 5.2d0 5.5d0 4.9d0 5.0d0 5.5d0 4.9d0 4.4d0 5.1d0
                5.0d0 4.5d0 4.4d0 5.0d0 5.1d0 4.8d0 5.1d0 4.6d0 5.3d0 5.0d0
                7.0d0 6.4d0 6.9d0 5.5d0 6.5d0 5.7d0 6.3d0 4.9d0 6.6d0 5.2d0
                5.0d0 5.9d0 6.0d0 6.1d0 5.6d0 6.7d0 5.6d0 5.8d0 6.2d0 5.6d0
                5.9d0 6.1d0 6.3d0 6.1d0 6.4d0 6.6d0 6.8d0 6.7d0 6.0d0 5.7d0
                5.5d0 5.5d0 5.8d0 6.0d0 5.4d0 6.0d0 6.7d0 6.3d0 5.6d0 5.5d0
                5.5d0 6.1d0 5.8d0 5.0d0 5.6d0 5.7d0 5.7d0 6.2d0 5.1d0 5.7d0
                6.3d0 5.8d0 7.1d0 6.3d0 6.5d0 7.6d0 4.9d0 7.3d0 6.7d0 7.2d0
                6.5d0 6.4d0 6.8d0 5.7d0 5.8d0 6.4d0 6.5d0 7.7d0 7.7d0 6.0d0
                6.9d0 5.6d0 7.7d0 6.3d0 6.7d0 7.2d0 6.2d0 6.1d0 6.4d0 7.2d0
                7.4d0 7.9d0 6.4d0 6.3d0 6.1d0 7.7d0 6.3d0 6.4d0 6.0d0 6.9d0
                6.7d0 6.9d0 5.8d0 6.8d0 6.7d0 6.7d0 6.3d0 6.5d0 6.2d0 5.9d0))
  "Iris sepal length (cm).")

(defparameter *iris-y*
  (make-array 150 :element-type 'double-float
              :initial-contents
              '(3.5d0 3.0d0 3.2d0 3.1d0 3.6d0 3.9d0 3.4d0 3.4d0 2.9d0 3.1d0
                3.7d0 3.4d0 3.0d0 3.0d0 4.0d0 4.4d0 3.9d0 3.5d0 3.8d0 3.8d0
                3.4d0 3.7d0 3.6d0 3.3d0 3.4d0 3.0d0 3.4d0 3.5d0 3.4d0 3.2d0
                3.1d0 3.4d0 4.1d0 4.2d0 3.1d0 3.2d0 3.5d0 3.6d0 3.0d0 3.4d0
                3.5d0 2.3d0 3.2d0 3.5d0 3.8d0 3.0d0 3.8d0 3.2d0 3.7d0 3.3d0
                3.2d0 3.2d0 3.1d0 2.3d0 2.8d0 2.8d0 3.3d0 2.4d0 2.9d0 2.7d0
                2.0d0 3.0d0 2.2d0 2.9d0 2.9d0 3.1d0 3.0d0 2.7d0 2.2d0 2.5d0
                3.2d0 2.8d0 2.5d0 2.8d0 2.9d0 3.0d0 2.8d0 3.0d0 2.9d0 2.6d0
                2.4d0 2.4d0 2.7d0 2.7d0 3.0d0 3.4d0 3.1d0 2.3d0 3.0d0 2.5d0
                2.6d0 3.0d0 2.6d0 2.3d0 2.7d0 3.0d0 2.9d0 2.9d0 2.5d0 2.8d0
                3.3d0 2.7d0 3.0d0 2.9d0 3.0d0 3.0d0 2.5d0 2.9d0 2.5d0 3.6d0
                3.2d0 2.7d0 3.0d0 2.5d0 2.8d0 3.2d0 3.0d0 3.8d0 2.6d0 2.2d0
                3.2d0 2.8d0 2.8d0 2.7d0 3.3d0 3.2d0 2.8d0 3.0d0 2.8d0 3.0d0
                2.8d0 3.8d0 2.8d0 2.8d0 2.6d0 3.0d0 3.4d0 3.1d0 3.0d0 3.1d0
                3.1d0 3.1d0 2.7d0 3.2d0 3.3d0 3.0d0 2.5d0 3.0d0 3.4d0 3.0d0))
  "Iris sepal width (cm).")
```

**Step 2: Write main.lisp**

```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.curve-fitting
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.curve-fitting)

;;; Linear Regression via Gradient Descent with Automatic Differentiation
;;;
;;; Fit y = w*x + b to the Iris sepal-length → sepal-width data.
;;; AD computes dL/dw and dL/db without manual derivation.

;; Load data
(load (merge-pathnames "data.lisp" *load-pathname*))

(defun mse-loss (w b xs ys)
  "Mean squared error: (1/N) * sum((y_i - (w*x_i + b))^2)."
  (let ((n (length xs))
        (total 0.0d0))
    (dotimes (i n)
      (let* ((x (aref xs i))
             (y (aref ys i))
             (pred (ad:+ (ad:* w x) b))
             (residual (ad:- y pred)))
        (setf total (ad:+ total (ad:* residual residual)))))
    (ad:/ total (coerce n 'double-float))))

(defun gradient-descent (xs ys &key (lr 0.01d0) (epochs 200))
  "Train linear regression y = w*x + b using gradient descent with AD."
  (let ((w 0.0d0)
        (b 0.0d0))
    (format t "~&~5A  ~12A  ~12A  ~12A~%" "Epoch" "Loss" "w" "b")
    (format t "~A~%" (make-string 45 :initial-element #\-))
    (dotimes (epoch epochs)
      ;; Compute gradients via AD
      (multiple-value-bind (loss-w dw)
          (ad:derivative (lambda (w) (mse-loss w b xs ys)) w)
        (multiple-value-bind (loss-b db)
            (ad:derivative (lambda (b) (mse-loss w b xs ys)) b)
          (declare (ignore loss-b))
          (when (zerop (mod epoch 20))
            (format t "~5D  ~12,6F  ~12,6F  ~12,6F~%" epoch loss-w w b))
          ;; Update parameters
          (decf w (* lr dw))
          (decf b (* lr db)))))
    (values w b)))

(defun ols-solution (xs ys)
  "Closed-form ordinary least squares: w = cov(x,y)/var(x), b = mean(y) - w*mean(x)."
  (let* ((n (length xs))
         (nf (coerce n 'double-float))
         (sum-x (loop for i below n sum (aref xs i)))
         (sum-y (loop for i below n sum (aref ys i)))
         (mean-x (/ sum-x nf))
         (mean-y (/ sum-y nf))
         (cov-xy (loop for i below n
                       sum (* (- (aref xs i) mean-x) (- (aref ys i) mean-y))))
         (var-x (loop for i below n
                      sum (expt (- (aref xs i) mean-x) 2)))
         (w (/ cov-xy var-x))
         (b (- mean-y (* w mean-x))))
    (values w b)))

(defun run-examples ()
  (format t "~&=== Linear Regression: Iris sepal-length -> sepal-width ===~%")
  (format t "~&Training with gradient descent (AD-computed gradients)...~%~%")

  (multiple-value-bind (w-gd b-gd)
      (gradient-descent *iris-x* *iris-y* :lr 0.01d0 :epochs 200)
    (multiple-value-bind (w-ols b-ols)
        (ols-solution *iris-x* *iris-y*)
      (format t "~%~&=== Results ===~%~%")
      (format t "~25A  ~12A  ~12A~%" "" "w (slope)" "b (intercept)")
      (format t "~A~%" (make-string 52 :initial-element #\-))
      (format t "~25A  ~12,6F  ~12,6F~%" "Gradient Descent (AD)" w-gd b-gd)
      (format t "~25A  ~12,6F  ~12,6F~%" "OLS (closed-form)" w-ols b-ols)
      (format t "~%Note: GD may not fully converge in 200 epochs. Increase epochs or lr to match OLS.~%"))))

(run-examples)
```

**Step 3: Test by loading in REPL**

Run: `(load "examples/01-curve-fitting/main.lisp")` via `repl-eval`
Expected: Loss decreasing over epochs, GD solution approaching OLS values (w ≈ -0.06, b ≈ 3.39 for Iris)

**Step 4: Write README.md**

```markdown
# Linear Regression via Gradient Descent

## What This Demonstrates

Using AD-computed gradients to optimize parameters in a gradient descent loop — the core pattern of machine learning.

## Theory

Minimize MSE loss L(w,b) = (1/N) * sum((y_i - (w*x_i + b))^2).

Since cl-acorn provides univariate `derivative`, we differentiate w.r.t. one parameter at a time:

```lisp
;; Gradient w.r.t. w (holding b fixed)
(ad:derivative (lambda (w) (mse-loss w b xs ys)) current-w)
;; Gradient w.r.t. b (holding w fixed)
(ad:derivative (lambda (b) (mse-loss w b xs ys)) current-b)
```

This is the standard forward-mode pattern for multi-variable optimization.

## Data

Iris dataset (UCI ML Repository): 150 samples of sepal-length (x) vs sepal-width (y).

## Running

```lisp
(load "examples/01-curve-fitting/main.lisp")
```

## Key Takeaway

AD enables gradient-based optimization without hand-deriving gradients. The per-parameter `derivative` pattern works for any number of parameters.
```

**Step 5: Commit**

```bash
git add examples/01-curve-fitting/
git commit -m "Add linear regression curve fitting example with Iris data"
```

---

### Task 5: Neural Network (02-neural-network)

Highest complexity. Full MLP with forward pass, loss, and per-weight gradient descent on Iris classification.

**Files:**
- Create: `examples/02-neural-network/main.lisp`
- Create: `examples/02-neural-network/data.lisp`
- Create: `examples/02-neural-network/README.md`

**Step 1: Write data.lisp**

Full Iris dataset: 4 features, 3 classes (0/1/2), 150 samples.

This file is large (~200 lines of data literals). Structure:

```lisp
(in-package #:cl-acorn.examples.neural-network)

;;; Iris Dataset: 4 features, 3 classes
;;; Source: UCI Machine Learning Repository (Fisher, 1936)

(defparameter *iris-features*
  ;; 150 x 4 nested list: (sepal-length sepal-width petal-length petal-width)
  '((5.1d0 3.5d0 1.4d0 0.2d0)
    (4.9d0 3.0d0 1.4d0 0.2d0)
    ...  ;; all 150 samples
    (5.9d0 3.0d0 5.1d0 1.8d0)))

(defparameter *iris-labels*
  ;; 150 class labels: 0=setosa, 1=versicolor, 2=virginica
  '(0 0 0 ... 1 1 1 ... 2 2 2 ...))
```

**Step 2: Write main.lisp**

```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.neural-network
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.neural-network)

;;; Multi-Layer Perceptron on Iris Dataset
;;;
;;; Architecture: 4 -> 8 (sigmoid) -> 3 (softmax)
;;; Training: gradient descent with forward-mode AD
;;;
;;; Educational note: Forward-mode AD requires one derivative call per parameter.
;;; With ~59 parameters, each training step makes 59 derivative calls.
;;; Reverse-mode AD (future phase 2) would need just 1 call.

(load (merge-pathnames "data.lisp" *load-pathname*))

;;; --- Network structure ---
;;; Parameters stored as flat vector for easy AD indexing

(defconstant +input-size+ 4)
(defconstant +hidden-size+ 8)
(defconstant +output-size+ 3)
(defconstant +num-params+ (+ (* +input-size+ +hidden-size+)  ; W1: 32
                              +hidden-size+                    ; b1: 8
                              (* +hidden-size+ +output-size+)  ; W2: 24
                              +output-size+))                  ; b2: 3  = 67

(defun sigmoid (x)
  (ad:/ 1.0d0 (ad:+ 1.0d0 (ad:exp (ad:- x)))))

(defun get-param (params index)
  "Get parameter at INDEX from params vector (list)."
  (nth index params))

(defun forward-pass (params input)
  "Forward pass through the MLP. Returns list of 3 output logits.
PARAMS is a list of numbers/duals. INPUT is a list of 4 numbers."
  ;; Hidden layer: h = sigmoid(W1*x + b1)
  (let ((hidden (loop for j below +hidden-size+
                      collect
                      (let ((sum (get-param params (+ (* +input-size+ +hidden-size+) j)))) ; b1[j]
                        (dotimes (i +input-size+)
                          (setf sum (ad:+ sum
                                         (ad:* (get-param params (+ (* j +input-size+) i))
                                               (nth i input)))))
                        (sigmoid sum)))))
    ;; Output layer: o = W2*h + b2 (logits, no activation)
    (let ((offset (+ (* +input-size+ +hidden-size+) +hidden-size+)))
      (loop for k below +output-size+
            collect
            (let ((sum (get-param params (+ offset (* +hidden-size+ +output-size+) k)))) ; b2[k]
              (dotimes (j +hidden-size+)
                (setf sum (ad:+ sum
                               (ad:* (get-param params (+ offset (* k +hidden-size+) j))
                                     (nth j hidden)))))
              sum)))))

(defun softmax-cross-entropy (logits label)
  "Cross-entropy loss for a single sample. LABEL is class index (0,1,2)."
  ;; Numerically stable: subtract max
  (let* ((max-logit (reduce (lambda (a b) (if (> (if (typep a 'ad:dual) (ad:dual-real a) a)
                                                  (if (typep b 'ad:dual) (ad:dual-real b) b))
                                               a b))
                            logits))
         (shifted (mapcar (lambda (l) (ad:- l max-logit)) logits))
         (exp-sum (reduce #'ad:+ (mapcar #'ad:exp shifted)))
         (log-softmax (ad:- (nth label shifted) (ad:log exp-sum))))
    (ad:- log-softmax)))

(defun total-loss (params features labels)
  "Average cross-entropy loss over all samples."
  (let ((n (length labels))
        (total 0.0d0))
    (loop for input in features
          for label in labels
          do (let ((logits (forward-pass params input)))
               (setf total (ad:+ total (softmax-cross-entropy logits label)))))
    (ad:/ total (coerce n 'double-float))))

(defun predict (params input)
  "Return predicted class (argmax of logits)."
  (let* ((logits (forward-pass params input))
         (real-logits (mapcar (lambda (l) (if (typep l 'ad:dual) (ad:dual-real l) l)) logits)))
    (position (reduce #'max real-logits) real-logits)))

(defun accuracy (params features labels)
  "Classification accuracy (0.0 to 1.0)."
  (let ((correct 0))
    (loop for input in features
          for label in labels
          when (= (predict params input) label)
            do (incf correct))
    (/ (coerce correct 'double-float) (length labels))))

(defun init-params ()
  "Initialize parameters with small random values (Xavier-like)."
  (let ((params (make-list +num-params+)))
    (dotimes (i +num-params+)
      (setf (nth i params)
            (* (- (random 1.0d0) 0.5d0) 0.2d0)))  ; Uniform [-0.1, 0.1]
    params))

(defun train (features labels &key (lr 0.1d0) (epochs 50))
  "Train MLP using gradient descent with forward-mode AD."
  (let ((params (init-params)))
    (format t "~&~5A  ~12A  ~10A~%" "Epoch" "Loss" "Accuracy")
    (format t "~A~%" (make-string 30 :initial-element #\-))

    (dotimes (epoch epochs)
      ;; Compute gradient for each parameter via separate derivative calls
      (let ((grads (make-list +num-params+ :initial-element 0.0d0)))
        (dotimes (i +num-params+)
          (multiple-value-bind (loss grad)
              (ad:derivative
               (lambda (p)
                 (let ((ps (copy-list params)))
                   (setf (nth i ps) p)
                   (total-loss ps features labels)))
               (nth i params))
            (declare (ignore loss))
            (setf (nth i grads) grad)))
        ;; Update all parameters
        (dotimes (i +num-params+)
          (decf (nth i params) (* lr (nth i grads)))))

      (when (zerop (mod epoch 5))
        (let ((loss (total-loss params features labels))
              (acc (accuracy params features labels)))
          (format t "~5D  ~12,6F  ~10,1F%~%"
                  epoch
                  (if (typep loss 'ad:dual) (ad:dual-real loss) loss)
                  (* 100 acc)))))

    (format t "~%Final accuracy: ~,1F%~%" (* 100 (accuracy params features labels)))
    params))

(defun run-examples ()
  (format t "~&=== MLP on Iris Dataset ===~%")
  (format t "Architecture: ~D -> ~D (sigmoid) -> ~D (softmax)~%"
          +input-size+ +hidden-size+ +output-size+)
  (format t "Parameters: ~D~%" +num-params+)
  (format t "Note: Forward-mode AD requires ~D derivative calls per update step.~%"
          +num-params+)
  (format t "      (Reverse-mode would need only 1.)~%~%")
  (train *iris-features* *iris-labels* :lr 0.1d0 :epochs 50))

(run-examples)
```

**Step 3: Test by loading in REPL**

Run: `(load "examples/02-neural-network/main.lisp")` via `repl-eval`
Expected: Loss decreasing, accuracy improving toward 80-95% on training set. May be slow (~67 derivative calls per epoch x 50 epochs).

**Step 4: Write README.md**

```markdown
# Multi-Layer Perceptron (Neural Network)

## What This Demonstrates

Training a neural network using AD-computed gradients — no manual backpropagation needed.

## Architecture

Input(4) -> Hidden(8, sigmoid) -> Output(3, softmax), cross-entropy loss.
67 trainable parameters.

## Educational Value

Forward-mode AD differentiates through the entire computation graph (matrix ops, sigmoid, softmax, cross-entropy). However, it requires **one `derivative` call per parameter** — 67 calls per training step.

This motivates **reverse-mode AD** (backpropagation), which computes all 67 gradients in a single backward pass. That's planned for cl-acorn phase 2.

## Data

Iris dataset: 150 samples, 4 features, 3 classes.

## Running

```lisp
(load "examples/02-neural-network/main.lisp")
```

**Note:** This example is intentionally slow to illustrate forward-mode AD's O(n) cost for n parameters.

## Key Takeaway

AD eliminates manual gradient derivation for arbitrarily complex functions. Forward-mode's per-parameter cost motivates reverse-mode for high-dimensional optimization.
```

**Step 5: Commit**

```bash
git add examples/02-neural-network/
git commit -m "Add neural network MLP example with Iris classification"
```

---

### Task 6: PID Controller Tuning (06-pid-control)

Differentiating through a simulation loop. Medium complexity.

**Files:**
- Create: `examples/06-pid-control/main.lisp`
- Create: `examples/06-pid-control/README.md`

**Step 1: Write main.lisp**

```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.pid-control
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.pid-control)

;;; PID Controller Auto-Tuning via Automatic Differentiation
;;;
;;; Plant: first-order system G(s) = 1/(s+1), discretized via Euler method
;;; Controller: PID with gains (Kp, Ki, Kd)
;;; Objective: minimize ISE (Integral of Squared Error) for step response

(defconstant +dt+ 0.01d0 "Simulation time step (s).")
(defconstant +t-final+ 5.0d0 "Simulation duration (s).")
(defconstant +n-steps+ (round (/ +t-final+ +dt+)))
(defconstant +setpoint+ 1.0d0 "Step response target.")

(defun simulate-pid (kp ki kd)
  "Simulate PID-controlled step response. Returns ISE (Integral of Squared Error).
KP, KI, KD may be dual numbers for AD."
  (let ((y 0.0d0)       ; plant output
        (integral 0.0d0) ; integral of error
        (prev-error 0.0d0) ; previous error for derivative term
        (ise 0.0d0))     ; accumulated squared error
    (dotimes (step +n-steps+)
      (let* ((error (ad:- +setpoint+ y))
             (derivative-term (ad:/ (ad:- error prev-error) +dt+))
             (control (ad:+ (ad:* kp error)
                           (ad:+ (ad:* ki integral)
                                 (ad:* kd derivative-term))))
             ;; Plant dynamics: dy/dt = -y + u  =>  y += (-y + u) * dt
             (dy (ad:* (ad:+ (ad:- y) control) +dt+)))
        (setf y (ad:+ y dy))
        (setf integral (ad:+ integral (ad:* error +dt+)))
        (setf prev-error error)
        (setf ise (ad:+ ise (ad:* (ad:* error error) +dt+)))))
    ise))

(defun optimize-pid (&key (lr 0.5d0) (epochs 100)
                          (kp-init 1.0d0) (ki-init 0.1d0) (kd-init 0.0d0))
  "Optimize PID gains using gradient descent with AD."
  (let ((kp kp-init) (ki ki-init) (kd kd-init))
    (format t "~&~5A  ~12A  ~10A  ~10A  ~10A~%"
            "Epoch" "ISE" "Kp" "Ki" "Kd")
    (format t "~A~%" (make-string 50 :initial-element #\-))

    (dotimes (epoch epochs)
      ;; Compute gradients via AD
      (multiple-value-bind (ise dkp)
          (ad:derivative (lambda (kp) (simulate-pid kp ki kd)) kp)
        (declare (ignore ise))
        (multiple-value-bind (ise dki)
            (ad:derivative (lambda (ki) (simulate-pid kp ki kd)) ki)
          (declare (ignore ise))
          (multiple-value-bind (ise dkd)
              (ad:derivative (lambda (kd) (simulate-pid kp ki kd)) kd)

            (when (zerop (mod epoch 10))
              (let ((ise-val (if (typep ise 'ad:dual) (ad:dual-real ise) ise)))
                (format t "~5D  ~12,6F  ~10,4F  ~10,4F  ~10,4F~%"
                        epoch ise-val kp ki kd)))

            ;; Gradient descent update
            (decf kp (* lr dkp))
            (decf ki (* lr dki))
            (decf kd (* lr dkd))))))

    (format t "~%Final gains: Kp=~,4F  Ki=~,4F  Kd=~,4F~%" kp ki kd)
    (values kp ki kd)))

(defun print-step-response (kp ki kd label)
  "Print step response at selected time points."
  (format t "~&--- Step Response (~A) ---~%" label)
  (format t "~8A  ~12A  ~12A~%" "Time" "Output" "Error")
  (let ((y 0.0d0) (integral 0.0d0) (prev-error 0.0d0))
    (dotimes (step +n-steps+)
      (let* ((time (* step +dt+))
             (error (- +setpoint+ y))
             (derivative-term (/ (- error prev-error) +dt+))
             (control (+ (* kp error) (* ki integral) (* kd derivative-term)))
             (dy (* (+ (- y) control) +dt+)))
        (setf y (+ y dy))
        (setf integral (+ integral (* error +dt+)))
        (setf prev-error error)
        (when (or (< time 0.005d0)
                  (zerop (mod step 50))
                  (= step (1- +n-steps+)))
          (format t "~8,2F  ~12,6F  ~12,6F~%" time y error))))))

(defun run-examples ()
  (format t "~&=== PID Controller Auto-Tuning ===~%")
  (format t "Plant: G(s) = 1/(s+1), step response to setpoint=~A~%"
          +setpoint+)
  (format t "Simulation: dt=~As, T=~As (~D steps)~%~%"
          +dt+ +t-final+ +n-steps+)

  ;; Show initial response
  (let ((kp0 1.0d0) (ki0 0.1d0) (kd0 0.0d0))
    (print-step-response kp0 ki0 kd0
                         (format nil "initial: Kp=~A Ki=~A Kd=~A" kp0 ki0 kd0))
    (format t "~%")

    ;; Optimize
    (multiple-value-bind (kp ki kd)
        (optimize-pid :kp-init kp0 :ki-init ki0 :kd-init kd0
                      :lr 0.5d0 :epochs 100)
      (format t "~%")
      (print-step-response kp ki kd
                           (format nil "optimized: Kp=~,3F Ki=~,3F Kd=~,3F"
                                   kp ki kd)))))

(run-examples)
```

**Step 2: Test by loading in REPL**

Run: `(load "examples/06-pid-control/main.lisp")` via `repl-eval`
Expected: ISE decreasing over epochs, optimized response settling faster with less overshoot.

**Step 3: Write README.md**

```markdown
# PID Controller Auto-Tuning

## What This Demonstrates

Differentiating through an entire simulation loop — AD computes gradients of an integral objective function w.r.t. controller parameters.

## Theory

A PID controller outputs u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt.

The objective is to minimize ISE (Integral of Squared Error) over a step response simulation. AD propagates derivatives through every timestep of the Euler discretization.

## Model

- **Plant**: First-order system G(s) = 1/(s+1)
- **Controller**: PID with gains (Kp, Ki, Kd)
- **Simulation**: 500 Euler steps, dt=0.01s, T=5s

## Running

```lisp
(load "examples/06-pid-control/main.lisp")
```

## Key Takeaway

AD can differentiate through iterative simulation loops, not just closed-form expressions. This enables gradient-based optimization of any differentiable simulation.
```

**Step 4: Commit**

```bash
git add examples/06-pid-control/
git commit -m "Add PID controller auto-tuning example"
```

---

### Task 7: FIR Filter Optimization (07-signal-processing)

Differentiating through a signal processing pipeline with array operations.

**Files:**
- Create: `examples/07-signal-processing/main.lisp`
- Create: `examples/07-signal-processing/README.md`

**Step 1: Write main.lisp**

```lisp
(asdf:load-system :cl-acorn)

(defpackage #:cl-acorn.examples.signal-processing
  (:use #:cl)
  (:local-nicknames (#:ad #:cl-acorn.ad)))

(in-package #:cl-acorn.examples.signal-processing)

;;; FIR Filter Coefficient Optimization via Automatic Differentiation
;;;
;;; Optimize a 3-tap FIR filter to remove noise from a sinusoidal signal.
;;; AD differentiates through the convolution operation.

(defconstant +n-samples+ 200 "Number of signal samples.")
(defconstant +freq+ 5.0d0 "Signal frequency (Hz).")
(defconstant +sample-rate+ 100.0d0 "Sample rate (Hz).")
(defconstant +noise-sigma+ 0.3d0 "Noise standard deviation.")
(defconstant +n-taps+ 5 "FIR filter taps.")

;;; --- Noise generation (Box-Muller) ---

(defvar *rng-state* 42)

(defun lcg-random ()
  "Simple linear congruential generator returning float in (0, 1)."
  (setf *rng-state* (mod (+ (* 1103515245 *rng-state*) 12345) (expt 2 31)))
  (/ (coerce *rng-state* 'double-float) (coerce (expt 2 31) 'double-float)))

(defun box-muller ()
  "Generate a standard normal random number via Box-Muller transform."
  (let ((u1 (max 1d-10 (lcg-random)))
        (u2 (lcg-random)))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

;;; --- Signal generation ---

(defun generate-signals ()
  "Generate clean signal, noise, and noisy signal.
Returns (values clean-signal noisy-signal) as vectors."
  (setf *rng-state* 42) ; reproducible
  (let ((clean (make-array +n-samples+ :element-type 'double-float))
        (noisy (make-array +n-samples+ :element-type 'double-float)))
    (dotimes (i +n-samples+)
      (let* ((t-val (/ (coerce i 'double-float) +sample-rate+))
             (s (sin (* 2.0d0 pi +freq+ t-val)))
             (noise (* +noise-sigma+ (box-muller))))
        (setf (aref clean i) s)
        (setf (aref noisy i) (+ s noise))))
    (values clean noisy)))

;;; --- FIR filter ---

(defun fir-filter (coeffs noisy-signal)
  "Apply FIR filter with COEFFS (list of N dual/numbers) to NOISY-SIGNAL (vector).
Returns list of filtered values (starting from index N-1)."
  (let* ((n-taps (length coeffs))
         (n-out (- +n-samples+ (1- n-taps)))
         (output '()))
    (loop for i from (1- n-taps) below +n-samples+
          do (let ((sum 0.0d0))
               (dotimes (j n-taps)
                 (setf sum (ad:+ sum
                                 (ad:* (nth j coeffs)
                                       (aref noisy-signal (- i j))))))
               (push sum output)))
    (nreverse output)))

(defun filter-mse (coeffs noisy-signal clean-signal)
  "MSE between FIR-filtered noisy signal and clean reference."
  (let* ((n-taps (length coeffs))
         (filtered (fir-filter coeffs noisy-signal))
         (n (length filtered))
         (total 0.0d0))
    (loop for f-val in filtered
          for i from (1- n-taps)
          do (let ((residual (ad:- f-val (aref clean-signal i))))
               (setf total (ad:+ total (ad:* residual residual)))))
    (ad:/ total (coerce n 'double-float))))

(defun optimize-filter (clean noisy &key (lr 0.01d0) (epochs 200))
  "Optimize FIR filter coefficients using gradient descent with AD."
  ;; Initialize: simple averaging filter
  (let ((coeffs (make-list +n-taps+ :initial-element (/ 1.0d0 +n-taps+))))
    (format t "~&~5A  ~12A  ~A~%"
            "Epoch" "MSE" "Coefficients")
    (format t "~A~%" (make-string 60 :initial-element #\-))

    (dotimes (epoch epochs)
      ;; Compute gradient for each coefficient
      (let ((grads (make-list +n-taps+ :initial-element 0.0d0)))
        (dotimes (i +n-taps+)
          (multiple-value-bind (mse grad)
              (ad:derivative
               (lambda (ci)
                 (let ((cs (copy-list coeffs)))
                   (setf (nth i cs) ci)
                   (filter-mse cs noisy clean)))
               (nth i coeffs))
            (declare (ignore mse))
            (setf (nth i grads) grad)))
        ;; Update
        (dotimes (i +n-taps+)
          (decf (nth i coeffs) (* lr (nth i grads)))))

      (when (zerop (mod epoch 20))
        (let ((mse (filter-mse coeffs noisy clean)))
          (format t "~5D  ~12,6F  ~{~,4F~^ ~}~%"
                  epoch
                  (if (typep mse 'ad:dual) (ad:dual-real mse) mse)
                  coeffs))))

    (format t "~%Optimized coefficients: ~{~,6F~^ ~}~%" coeffs)
    coeffs))

(defun run-examples ()
  (format t "~&=== FIR Filter Optimization ===~%")
  (format t "Signal: ~AHz sine at ~AHz sample rate~%" +freq+ +sample-rate+)
  (format t "Noise: Gaussian, sigma=~A~%" +noise-sigma+)
  (format t "Filter: ~D-tap FIR~%~%" +n-taps+)

  (multiple-value-bind (clean noisy) (generate-signals)
    ;; Initial MSE (no filtering = identity)
    (let ((initial-mse (filter-mse (list 1.0d0 0.0d0 0.0d0 0.0d0 0.0d0) noisy clean)))
      (format t "Initial MSE (passthrough): ~,6F~%~%"
              (if (typep initial-mse 'ad:dual) (ad:dual-real initial-mse) initial-mse)))

    ;; Optimize
    (let ((coeffs (optimize-filter clean noisy :lr 0.01d0 :epochs 200)))
      ;; Final MSE
      (let ((final-mse (filter-mse coeffs noisy clean)))
        (format t "~%Final MSE: ~,6F~%"
                (if (typep final-mse 'ad:dual) (ad:dual-real final-mse) final-mse)))
      ;; Compare with simple averaging
      (let ((avg-coeffs (make-list +n-taps+ :initial-element (/ 1.0d0 +n-taps+))))
        (let ((avg-mse (filter-mse avg-coeffs noisy clean)))
          (format t "Averaging filter MSE: ~,6F~%"
                  (if (typep avg-mse 'ad:dual) (ad:dual-real avg-mse) avg-mse)))))))

(run-examples)
```

**Step 2: Test by loading in REPL**

Run: `(load "examples/07-signal-processing/main.lisp")` via `repl-eval`
Expected: MSE decreasing, optimized filter outperforming simple averaging filter.

**Step 3: Write README.md**

```markdown
# FIR Filter Optimization

## What This Demonstrates

Optimizing signal processing filter coefficients by differentiating through the entire filtering pipeline with AD.

## Theory

A FIR filter computes y[n] = sum(a_k * x[n-k]) for k=0..N-1.

We minimize MSE between filtered noisy signal and the clean reference. AD differentiates through the convolution:

```lisp
(ad:derivative
 (lambda (a0)
   (let ((coeffs (list a0 a1 a2 a3 a4)))
     (filter-mse coeffs noisy-signal clean-signal)))
 current-a0)
```

## Signal

- Clean: 5Hz sinusoid sampled at 100Hz
- Noise: Gaussian (sigma=0.3) via Box-Muller
- Filter: 5-tap FIR

## Running

```lisp
(load "examples/07-signal-processing/main.lisp")
```

## Key Takeaway

AD works through array-indexed operations and convolution — any differentiable computation graph, not just mathematical formulas.
```

**Step 4: Commit**

```bash
git add examples/07-signal-processing/
git commit -m "Add FIR filter optimization example"
```

---

### Task 8: Validation and Final Commit

Verify all examples load and run correctly, then make a final integration commit.

**Files:**
- Verify: all `examples/*/main.lisp`

**Step 1: Run each example via REPL**

```lisp
;; Run each in sequence, verify no errors
(dolist (example '("examples/03-newton-method/main.lisp"
                   "examples/04-sensitivity/main.lisp"
                   "examples/05-black-scholes/main.lisp"
                   "examples/01-curve-fitting/main.lisp"
                   "examples/06-pid-control/main.lisp"
                   "examples/07-signal-processing/main.lisp"))
  (format t "~%~%========== Loading ~A ==========~%~%" example)
  (load example))
;; Run neural network separately (may be slow)
(load "examples/02-neural-network/main.lisp")
```

Expected: All 7 examples produce output without errors.

**Step 2: Fix any issues found**

Debug and fix any runtime errors, numerical issues, or formatting problems.

**Step 3: Verify git status and final commit**

```bash
git status
# If any fixes were made:
git add -A examples/
git commit -m "Fix example issues found during validation"
```
