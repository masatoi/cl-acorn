# cl-acorn

Forward-mode automatic differentiation for Common Lisp using dual numbers.

```lisp
(ql:quickload :cl-acorn)

(defun f (x)
  (ad:+ (ad:* x x x) (ad:* -2 x) 5))  ; x^3 - 2x + 5

(ad:derivative #'f 3.0d0)
;; => 26.0d0, 25.0d0
;;    f(3) = 26, f'(3) = 25
```

`ad:derivative` returns two values: `f(x)` and `f'(x)`, computed exactly with no numerical approximation.

## How It Works

cl-acorn implements [forward-mode AD](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation) via dual numbers. A dual number carries a value and its derivative simultaneously: `a + b*epsilon` where `epsilon^2 = 0`. Arithmetic on dual numbers propagates derivatives automatically through the chain rule.

To differentiate `f` at `x`, cl-acorn seeds `x` as the dual number `x + 1*epsilon`, evaluates `f`, and reads off the epsilon component as the derivative.

## Installation

cl-acorn has no dependencies beyond ANSI Common Lisp. Clone this repository to a location visible to ASDF/Quicklisp:

```bash
cd ~/common-lisp/  # or ~/quicklisp/local-projects/
git clone https://github.com/wiz/cl-acorn.git
```

```lisp
(ql:quickload :cl-acorn)
```

## API Reference

All symbols are exported from the `cl-acorn.ad` package (nickname: `ad`).

### Dual Numbers

| Symbol | Description |
|--------|-------------|
| `dual` | CLOS class representing a dual number |
| `(make-dual real &optional epsilon)` | Construct a dual number (inputs coerced to `double-float`) |
| `(dual-real d)` | Extract the real part |
| `(dual-epsilon d)` | Extract the epsilon (derivative) part |

### Arithmetic

| Symbol | Description |
|--------|-------------|
| `ad:+`, `ad:-`, `ad:*`, `ad:/` | N-ary arithmetic, accepting any mix of dual numbers and plain numbers |

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

## Usage Patterns

### Basic Differentiation

```lisp
;; d/dx sin(x) = cos(x)
(ad:derivative #'ad:sin 0.0d0)
;; => 0.0d0, 1.0d0

;; d/dx e^x = e^x
(ad:derivative #'ad:exp 1.0d0)
;; => 2.718281828459045d0, 2.718281828459045d0
```

### Composing Functions

```lisp
;; d/dx sin(x^2) = 2x*cos(x^2)
(ad:derivative (lambda (x) (ad:sin (ad:* x x))) 1.0d0)
;; => 0.8414709848078965d0, 1.0806046117362795d0
```

### Multi-parameter Optimization via Closures

Forward-mode AD differentiates with respect to one variable at a time. For multi-parameter optimization, close over the other parameters:

```lisp
(defun loss (w b data)
  "MSE loss as a function of weight w and bias b."
  (ad:/ (reduce #'ad:+ data
                :key (lambda (xy)
                       (let ((err (ad:- (cdr xy) (ad:+ (ad:* w (car xy)) b))))
                         (ad:* err err))))
        (length data)))

;; Gradient with respect to w (b fixed):
(ad:derivative (lambda (w) (loss w current-b data)) current-w)

;; Gradient with respect to b (w fixed):
(ad:derivative (lambda (b) (loss current-w b data)) current-b)
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

## Examples

The `examples/` directory contains complete, runnable demonstrations:

| Example | Description |
|---------|-------------|
| [01-curve-fitting](examples/01-curve-fitting/) | Linear regression on Iris data via gradient descent |
| [02-neural-network](examples/02-neural-network/) | MLP classifier (4-8-3) trained with AD-computed gradients |
| [03-newton-method](examples/03-newton-method/) | Newton-Raphson root finding with AD-derived Jacobians |
| [04-sensitivity](examples/04-sensitivity/) | Parameter sensitivity analysis for physics models |
| [05-black-scholes](examples/05-black-scholes/) | Option Greeks (Delta, Gamma, Vega, Theta, Rho) via AD |
| [06-pid-control](examples/06-pid-control/) | PID controller auto-tuning by differentiating through simulation |
| [07-signal-processing](examples/07-signal-processing/) | FIR filter coefficient optimization |

Run any example:

```lisp
(load "examples/01-curve-fitting/main.lisp")
```

## Running Tests

cl-acorn uses [Rove](https://github.com/fukamachi/rove) for testing (60 tests).

```bash
rove cl-acorn.asd
```

Or from the REPL:

```lisp
(asdf:test-system :cl-acorn)
```

## License

MIT
