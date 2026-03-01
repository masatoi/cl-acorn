# Forward-Mode Automatic Differentiation Design

**Date:** 2026-03-02
**Phase:** 1 of PPL project (cl-acorn)
**Status:** Approved

## Summary

Implement forward-mode automatic differentiation using dual numbers and CLOS generic functions. This is the foundational phase for a probabilistic programming language built on Common Lisp.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API style | Prefixed functions (`ad:+`, `ad:sin`) | No shadowing conflicts; safe integration with existing CL code |
| Precision | Double-float internal storage | Best balance of precision and performance for scientific computing |
| Return style | Multiple values `(values f(x) f'(x))` | Idiomatic CL; easy to ignore derivative when only value needed |
| Dispatch | CLOS generic functions | Extensible for phase 2 (reverse-mode); idiomatic; method combination |

## Mathematical Foundation

Dual numbers: `a + b*epsilon` where `epsilon^2 = 0`.

For any differentiable function `f`, evaluating `f(x + 1*epsilon)` yields `f(x) + f'(x)*epsilon`, giving both the function value and its derivative in a single forward pass.

## Package Structure

```
cl-acorn/
├── cl-acorn.asd
├── src/
│   ├── package.lisp           # Package definitions (cl-acorn.ad)
│   ├── dual.lisp              # CLOS dual class, constructors, print-object
│   ├── arithmetic.lisp        # Generic binary ops: +, -, *, /
│   ├── transcendental.lisp    # Generic unary ops: sin, cos, tan, exp, log, expt, sqrt, abs
│   └── interface.lisp         # derivative function
└── tests/
    ├── package.lisp           # Test package definition
    ├── dual-test.lisp         # Dual construction & accessor tests
    ├── arithmetic-test.lisp   # Binary op tests including mixed-type
    ├── transcendental-test.lisp  # Unary math function tests
    └── derivative-test.lisp   # End-to-end composite function derivative tests
```

### Packages

- **`cl-acorn.ad`**: Public API. Exports `dual`, `make-dual`, `dual-real`, `dual-epsilon`, `derivative`, and all arithmetic/transcendental function names (`+`, `-`, `*`, `/`, `sin`, `cos`, `tan`, `exp`, `log`, `expt`, `sqrt`, `abs`).
- Users access via `cl-acorn.ad:+` or local nickname `(:local-nicknames (:ad :cl-acorn.ad))`.

## Core Data Structure

```lisp
(defclass dual ()
  ((real-part :initarg :real
              :reader dual-real
              :type double-float)
   (epsilon   :initarg :epsilon
              :reader dual-epsilon
              :type double-float
              :initform 0.0d0)))
```

- Slots typed `double-float`, coercion in constructor `make-dual`
- Immutable by convention (reader accessors only)
- `print-object` shows `#<DUAL a + b*epsilon>`

## Arithmetic Operations (Binary)

Each binary op requires 3 CLOS methods: `dual x dual`, `dual x number`, `number x dual`.

**Rules (where `(a + b*e)` and `(c + d*e)` are duals):**

| Op | Real part | Epsilon part |
|----|-----------|-------------|
| `+` | `a + c` | `b + d` |
| `-` | `a - c` | `b - d` |
| `*` | `a * c` | `a*d + b*c` |
| `/` | `a / c` | `(b*c - a*d) / c^2` |

N-ary: `ad:+` and `ad:*` accept `&rest` args and reduce pairwise, falling through to CL ops when no duals present.

## Transcendental Operations (Unary)

Apply chain rule: `f(a + b*e) = f(a) + f'(a)*b*e`.

| Function | Value | Derivative coefficient |
|----------|-------|----------------------|
| `sin` | `sin(a)` | `cos(a) * b` |
| `cos` | `cos(a)` | `-sin(a) * b` |
| `tan` | `tan(a)` | `b / cos^2(a)` |
| `exp` | `exp(a)` | `exp(a) * b` |
| `log` | `log(a)` | `b / a` |
| `sqrt` | `sqrt(a)` | `b / (2 * sqrt(a))` |
| `expt(d,n)` | `a^n` | `n * a^(n-1) * b` |
| `abs` | `abs(a)` | `sign(a) * b` |

For `number` arguments, delegate to corresponding `cl:` functions.

## Derivative Interface

```lisp
(defun derivative (fn x)
  "Compute f(x) and f'(x) via forward-mode AD.
Returns (values f(x) f'(x)) as double-floats."
  (let* ((x-dual (make-dual x 1.0d0))
         (result (funcall fn x-dual)))
    (etypecase result
      (dual   (values (dual-real result) (dual-epsilon result)))
      (number (values (coerce result 'double-float) 0.0d0)))))
```

Seed with `epsilon = 1`, evaluate, extract parts.

## Test Strategy

All tests use Rove. Tolerance: `1d-10` for floating-point comparisons.

### Test Layers

1. **dual-test.lisp**: Construction, accessors, print-object, type coercion from integers/rationals
2. **arithmetic-test.lisp**: Each binary op with dual*dual, dual*number, number*dual; edge cases (zero, negative, identity)
3. **transcendental-test.lisp**: Each unary op against known derivatives
4. **derivative-test.lisp**: Composite functions:
   - `f(x) = x^2` -> `f'(x) = 2x`
   - `f(x) = sin(x^2) + exp(2x)/x` -> analytically computed derivative
   - `f(x) = log(x) * sqrt(x)` -> product rule verification
   - Constant function -> derivative = 0
   - Identity -> derivative = 1

## Future Extensibility (Phase 2+)

The CLOS generic function approach allows phase 2 (reverse-mode AD) to add methods for a `tape-node` class without modifying existing code. The `cl-acorn.ad` package can be extended with:
- `gradient` for multivariate functions
- `tape-node` class for reverse-mode
- Higher-order derivatives via nested dual numbers
