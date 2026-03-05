# Newton-Raphson Root Finding with Automatic Differentiation

## What This Demonstrates

Newton-Raphson iteration requires the derivative f'(x) at every step. Traditionally you either derive it by hand or approximate it with finite differences. With cl-acorn's forward-mode AD, the exact derivative comes for free — you define f(x) once and `ad:derivative` gives you both f(x) and f'(x).

## Theory

Newton's method updates an estimate x_n toward a root of f:

    x_{n+1} = x_n - f(x_n) / f'(x_n)

Convergence is quadratic near simple roots: each iteration roughly doubles the number of correct digits. The iteration tables printed by this example make that rate visible.

## Problems Solved

1. **x^3 - 2x - 5 = 0** starting from x0 = 2.0 (root near 2.09455)
2. **cos(x) - x = 0** starting from x0 = 1.0 (Dottie number, near 0.73909)

## How to Run

```
sbcl --load examples/03-newton-method/main.lisp
```

Or from a running REPL:

```lisp
(load "examples/03-newton-method/main.lisp")
```

## Key Takeaway

AD eliminates an entire class of bugs — wrong derivatives, stale derivatives after refactoring, and truncation error from finite differences — by computing exact derivatives mechanically from the function definition.
