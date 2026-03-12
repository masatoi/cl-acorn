# cl-acorn

Automatic differentiation for Common Lisp -- forward-mode (dual numbers) and reverse-mode (tape-based backpropagation).

```lisp
(ql:quickload :cl-acorn)

;; Forward-mode: derivative of a single-variable function
(ad:derivative (lambda (x) (ad:+ (ad:* x x x) (ad:* -2 x) 5)) 3.0d0)
;; => 26.0d0, 25.0d0  (f(3) = 26, f'(3) = 25)

;; Reverse-mode: gradient of a multi-variable function
(ad:gradient (lambda (p)
               (let ((x (first p)) (y (second p)))
                 (ad:+ (ad:* x x) (ad:* x y))))
             '(3.0d0 4.0d0))
;; => 21.0d0, (10.0d0 3.0d0)  (f = 21, df/dx = 10, df/dy = 3)
```

## How It Works

cl-acorn provides two AD modes:

**Forward-mode** via [dual numbers](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation): a dual number `a + b*epsilon` (where `epsilon^2 = 0`) propagates derivatives through the chain rule. Efficient for functions with few inputs and many outputs.

**Reverse-mode** via a dynamic tape (Wengert list): records operations during a forward pass, then backpropagates gradients in a single backward pass. Efficient for functions with many inputs and few outputs (e.g., neural network training).

Both modes use the same arithmetic operators (`ad:+`, `ad:sin`, etc.) -- CLOS dispatch selects the correct method based on input type (`dual` or `tape-node`).

## Installation

cl-acorn has no dependencies beyond ANSI Common Lisp. Clone this repository to a location visible to ASDF/Quicklisp:

```bash
cd ~/common-lisp/  # or ~/quicklisp/local-projects/
git clone https://github.com/masatoi/cl-acorn.git
```

```lisp
(ql:quickload :cl-acorn)
```

## API Reference

All symbols are exported from the `cl-acorn.ad` package (nickname: `ad`).

### Dual Numbers (Forward-Mode)

| Symbol | Description |
|--------|-------------|
| `dual` | CLOS class representing a dual number |
| `(make-dual real &optional epsilon)` | Construct a dual number (inputs coerced to `double-float`) |
| `(dual-real d)` | Extract the real part |
| `(dual-epsilon d)` | Extract the epsilon (derivative) part |

### Tape Nodes (Reverse-Mode)

| Symbol | Description |
|--------|-------------|
| `tape-node` | CLOS class representing a node in the computation graph |
| `(node-value n)` | Extract the computed value |
| `(node-gradient n)` | Extract the accumulated gradient (after backward pass) |

### Arithmetic

| Symbol | Description |
|--------|-------------|
| `ad:+`, `ad:-`, `ad:*`, `ad:/` | N-ary arithmetic, accepting any mix of dual numbers, tape nodes, and plain numbers |

### Transcendental Functions

| Symbol | Description |
|--------|-------------|
| `ad:sin`, `ad:cos`, `ad:tan` | Trigonometric functions |
| `ad:exp`, `ad:log` | Exponential and natural logarithm (`ad:log` accepts optional base) |
| `ad:sqrt`, `ad:abs`, `ad:expt` | Square root, absolute value, exponentiation |

### Differentiation

```lisp
(ad:derivative fn x) => f(x), f'(x)
```

Computes `f(x)` and `f'(x)` using forward-mode AD. `fn` must be a function of one argument that uses `ad:` arithmetic. Returns two `double-float` values.

```lisp
(ad:gradient fn params) => f(params), (df/dp1 df/dp2 ...)
```

Computes the gradient of scalar function `fn` at `params` using reverse-mode AD. `fn` must accept a list of tape-node values and return a scalar. `params` is a list of numbers. Returns `f(params)` and a list of partial derivatives.

```lisp
(ad:jacobian-vector-product fn params vector) => f(params), J*v
```

Computes `J*v` where `J` is the Jacobian of `fn` at `params`. Uses forward-mode seeding. `fn` must accept and return lists of values. Returns both as lists of `double-float`.

```lisp
(ad:hessian-vector-product fn params vector) => gradient, H*v
```

Computes `H*v` where `H` is the Hessian of scalar `fn` at `params`. Uses forward-over-reverse composition. Returns the gradient and Hessian-vector product as lists of `double-float`.

## Usage Patterns

### Basic Differentiation (Forward-Mode)

```lisp
;; d/dx sin(x) = cos(x)
(ad:derivative #'ad:sin 0.0d0)
;; => 0.0d0, 1.0d0

;; d/dx e^x = e^x
(ad:derivative #'ad:exp 1.0d0)
;; => 2.718281828459045d0, 2.718281828459045d0
```

### Multi-Variable Gradient (Reverse-Mode)

```lisp
;; Gradient of f(x,y) = sin(x) * exp(y)
(ad:gradient (lambda (p)
               (ad:* (ad:sin (first p)) (ad:exp (second p))))
             '(1.0d0 0.0d0))
;; => 0.8414..., (0.5403... 0.8414...)
;; df/dx = cos(x)*exp(y), df/dy = sin(x)*exp(y)
```

### Composing Functions

```lisp
;; d/dx sin(x^2) = 2x*cos(x^2)
(ad:derivative (lambda (x) (ad:sin (ad:* x x))) 1.0d0)
;; => 0.8414709848078965d0, 1.0806046117362795d0
```

### Differentiating Through Loops

AD propagates through arbitrary program structures -- loops, accumulators, conditionals:

```lisp
(defun simulate (param)
  "Run a 100-step simulation parameterized by param."
  (let ((state (ad:* param 0.1d0)))
    (dotimes (i 100)
      (setf state (ad:+ state (ad:* 0.01d0 (ad:sin state)))))
    state))

(ad:derivative #'simulate 1.0d0)
;; => exact derivative of the entire simulation w.r.t. param
```

### Hessian-Vector Product (Forward-over-Reverse)

```lisp
;; H*v for f(x,y) = x^2 + x*y
;; Hessian = [[2, 1], [1, 0]]
(ad:hessian-vector-product
 (lambda (p)
   (let ((x (first p)) (y (second p)))
     (ad:+ (ad:* x x) (ad:* x y))))
 '(3.0d0 4.0d0)
 '(1.0d0 0.0d0))
;; => (10.0d0 3.0d0), (2.0d0 1.0d0)
;; gradient = (10, 3), H*v = (2, 1)
```

## Examples

The `examples/` directory contains complete, runnable demonstrations:

| Example | Description |
|---------|-------------|
| [01-curve-fitting](examples/01-curve-fitting/) | Linear regression on Iris data via gradient descent |
| [02-neural-network](examples/02-neural-network/) | MLP classifier (4-8-3) trained with forward-mode AD |
| [03-newton-method](examples/03-newton-method/) | Newton-Raphson root finding with AD-derived Jacobians |
| [04-sensitivity](examples/04-sensitivity/) | Parameter sensitivity analysis for physics models |
| [05-black-scholes](examples/05-black-scholes/) | Option Greeks (Delta, Gamma, Vega, Theta, Rho) via AD |
| [06-pid-control](examples/06-pid-control/) | PID controller auto-tuning by differentiating through simulation |
| [07-signal-processing](examples/07-signal-processing/) | FIR filter coefficient optimization |
| [08-reverse-neural-network](examples/08-reverse-neural-network/) | MLP classifier trained with reverse-mode AD (1 backward pass vs 67 forward passes) |

Run any example:

```lisp
(load "examples/01-curve-fitting/main.lisp")
```

## Running Tests

cl-acorn uses [Rove](https://github.com/fukamachi/rove) for testing (107 tests).

```bash
rove cl-acorn.asd
```

Or from the REPL:

```lisp
(asdf:test-system :cl-acorn)
```

## License

MIT
