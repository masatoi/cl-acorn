# Reverse-Mode AD Design

Phase 2 of cl-acorn: tape-based reverse-mode automatic differentiation with gradient, Jacobian-vector product, and Hessian-vector product.

## Goals

1. Reverse-mode AD via dynamic tape (Wengert list) -- O(1) gradient computation regardless of parameter count
2. Unified API -- same `ad:+`, `ad:sin` etc. work with both `dual` (forward) and `tape-node` (reverse)
3. `gradient` function for multi-variable optimization
4. Forward-over-reverse for Hessian-vector product (HMC prerequisite)
5. Neural network example rewritten with reverse-mode for performance comparison

## Non-Goals

- Probability distributions and sampling (Phase 3)
- MCMC / HMC inference algorithms (Phase 3)
- Static graph compilation or JIT optimization
- GPU acceleration

## Data Structure: `tape-node`

```lisp
(defclass tape-node ()
  ((value    :initarg :value    :accessor node-value)
   (gradient :initarg :gradient :accessor node-gradient
             :initform 0)
   (children :initarg :children :accessor node-children :type list
             :initform nil))
  (:documentation "A node in the computation graph for reverse-mode AD."))
```

- `value`: The computed value at this node. Normally `double-float`; during forward-over-reverse, may hold a `dual`.
- `gradient`: Accumulated gradient during backward pass. Same type flexibility as `value`.
- `children`: List of `(child-tape-node . local-partial-derivative)` cons cells. Encodes the chain rule factors needed for backpropagation.

### Tape Management

```lisp
(defvar *tape* nil "Dynamic computation graph. nil when not recording.")
```

- `*tape*` is a list; each `make-node` call pushes the new node onto it.
- Since nodes are pushed in evaluation order, the list head is the most recent operation -- iterating the list gives reverse topological order, which is exactly what backward pass needs.
- When `*tape*` is nil, no tape-node methods are active (forward-mode has zero overhead).

## Arithmetic Methods

Existing generic functions (`binary-add`, `binary-sub`, `binary-mul`, `binary-div`, `ad-sin`, `ad-cos`, etc.) gain `tape-node` methods. Each method:

1. Computes the forward value using `cl:` arithmetic on `node-value`
2. Records local partial derivatives in the `children` list
3. Pushes the new node onto `*tape*`

### Pattern (binary operations, 4 methods each)

```
(tape-node, tape-node) -> make-node(op(a.val, b.val), [(a, d_op/da), (b, d_op/db)])
(tape-node, number)    -> make-node(op(a.val, b),     [(a, d_op/da)])
(number, tape-node)    -> make-node(op(a, b.val),     [(b, d_op/db)])
```

The `(number, number)` case is already handled by existing methods.

### Local Partial Derivatives

| Operation | d/da | d/db |
|-----------|------|------|
| a + b | 1 | 1 |
| a - b | 1 | -1 |
| a * b | b | a |
| a / b | 1/b | -a/b^2 |
| sin(a) | cos(a) | -- |
| cos(a) | -sin(a) | -- |
| tan(a) | 1/cos^2(a) | -- |
| exp(a) | exp(a) | -- |
| log(a) | 1/a | -- |
| sqrt(a) | 1/(2*sqrt(a)) | -- |
| abs(a) | sign(a) | -- |
| a^b (base=tape) | b*a^(b-1) | a^b*log(a) |

## Backward Pass

```lisp
(defun backward (node)
  "Backpropagate gradients from NODE through the tape."
  (setf (node-gradient node) 1.0d0)
  (dolist (n *tape*)
    (dolist (child-entry (node-children n))
      (let ((child (car child-entry))
            (local-grad (cdr child-entry)))
        (incf (node-gradient child)
              (cl:* (node-gradient n) local-grad))))))
```

Note: `incf` with `cl:*` will need generalization for forward-over-reverse (where gradient and local-grad may be `dual`). Use `ad:+` and `ad:*` internally, or provide a helper that dispatches appropriately.

## Public Interface

### `gradient`

```lisp
(defun gradient (fn params)
  "Compute the gradient of scalar function FN at PARAMS.
FN: function taking a list of tape-nodes, returning a scalar tape-node.
PARAMS: list of numbers.
Returns (values f(params) gradient-list)."
  ...)
```

Usage:
```lisp
(ad:gradient (lambda (p)
               (let ((x (first p)) (y (second p)))
                 (ad:+ (ad:* x x) (ad:* x y))))
             '(3.0d0 4.0d0))
;; => 13.0d0, (10.0d0 3.0d0)
```

### `jacobian-vector-product`

```lisp
(defun jacobian-vector-product (fn params vector)
  "Compute J*v where J is the Jacobian of FN at PARAMS.
Uses forward-mode: seeds each parameter as dual(param, vector-component).
FN: function taking a list of values, returning a list of values.
Returns (values f(params) J*v)."
  ...)
```

### `hessian-vector-product`

```lisp
(defun hessian-vector-product (fn params vector)
  "Compute H*v where H is the Hessian of scalar FN at PARAMS.
Uses forward-over-reverse: runs gradient computation with dual-number inputs.
Returns (values gradient H*v)."
  ...)
```

This requires `tape-node` value/gradient slots to accept `dual` numbers, and backward pass arithmetic to use AD-aware operations.

## File Structure

| File | Contents |
|------|----------|
| `src/tape.lisp` | `tape-node` class, `*tape*`, `make-node`, `backward` |
| `src/reverse-arithmetic.lisp` | `tape-node` methods for `binary-add/sub/mul/div`, `unary-negate/reciprocal` |
| `src/reverse-transcendental.lisp` | `tape-node` methods for `ad-sin/cos/tan/exp`, `unary-log`, `ad-sqrt/abs`, `binary-expt` |
| `src/gradient.lisp` | `gradient`, `jacobian-vector-product`, `hessian-vector-product` |

### Package Changes

Add to `cl-acorn.ad` exports:
```lisp
#:tape-node #:node-value #:node-gradient
#:gradient #:jacobian-vector-product #:hessian-vector-product
```

### ASDF Changes

Add new files to `cl-acorn.asd` components (after `interface.lisp`):
```lisp
(:file "tape")
(:file "reverse-arithmetic")
(:file "reverse-transcendental")
(:file "gradient")
```

## Test Strategy

| Test File | What It Covers |
|-----------|---------------|
| `tests/tape-test.lisp` | `tape-node` creation, `backward` correctness, tape lifecycle |
| `tests/reverse-arithmetic-test.lisp` | Each arithmetic op's gradient vs. analytical/forward-mode |
| `tests/reverse-transcendental-test.lisp` | Each transcendental op's gradient vs. analytical/forward-mode |
| `tests/gradient-test.lisp` | `gradient`, `jacobian-vector-product`, `hessian-vector-product` |

Validation approach: For every operation, verify that reverse-mode gradient matches forward-mode `derivative` to within `approx=` tolerance (1e-10).

Existing 60 forward-mode tests must continue to pass (regression).

## Example

`examples/08-reverse-neural-network/`: Rewrite the MLP classifier (example 02) using `ad:gradient` instead of per-parameter `ad:derivative` calls. Compare performance (67 forward passes vs. 1 backward pass per epoch).

## Design Decisions

1. **Dynamic tape over static graph**: Matches concept document's Define-by-Run philosophy. Loops/conditionals naturally supported.
2. **Unified API over separate packages**: CLOS dispatch selects forward vs. reverse based on input type. User code is mode-agnostic.
3. **Type-flexible slots**: `value` and `gradient` slots accept both `number` and `dual` to enable forward-over-reverse composition.
4. **`*tape*` as special variable**: Conventional Lisp idiom. `gradient` binds it dynamically; nested calls are safe.
