# Curve Fitting with Automatic Differentiation

Linear regression on the Iris dataset using gradient descent with
cl-acorn's forward-mode automatic differentiation.

## What This Demonstrates

- **Multi-variable optimization with single-variable AD**: cl-acorn's
  `ad:derivative` differentiates functions of one argument. To optimize
  over multiple parameters (w and b), we create closures that fix all
  parameters except the one being differentiated. This "close over the
  rest" pattern is the standard approach for forward-mode AD.

- **AD-aware loss function**: The `mse-loss` function uses `ad:+`, `ad:*`,
  `ad:/`, and `ad:-` throughout, so it works transparently with both plain
  numbers and dual numbers.

- **Gradient descent loop**: Each iteration makes two `ad:derivative` calls
  (one per parameter), then updates both parameters with standard arithmetic.

## Data Source

150 observations from the UCI Iris dataset (Fisher, 1936): sepal-length (x)
versus sepal-width (y). The relationship is weak (r = -0.11), so the optimal
slope is near zero.

## How to Run

```lisp
(load "examples/01-curve-fitting/main.lisp")
```

Or from the project root:

```bash
sbcl --load examples/01-curve-fitting/main.lisp
```

## Key Takeaway

Forward-mode AD computes exact derivatives with no numerical approximation.
The gradients are always correct; convergence speed depends only on the
optimizer's learning rate and iteration count. The example also computes
the closed-form OLS solution for comparison, showing the target that
gradient descent is approaching.
