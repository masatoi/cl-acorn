# Parameter Sensitivity Analysis with Automatic Differentiation

## What This Demonstrates

Parameter sensitivity analysis answers the question: how much does a model's output change when an input parameter changes slightly? This example uses cl-acorn's forward-mode AD to compute exact sensitivities (partial derivatives) for two physics models, and validates them against hand-derived analytical formulas.

## Theory

Given a model f(p) that depends on parameter p, the sensitivity is df/dp. Finite differences approximate this as [f(p+h) - f(p)] / h, but the result depends on the step size h and suffers from truncation and cancellation errors. AD computes the exact derivative to machine precision by propagating dual numbers through the model, with no step-size tuning required.

## Models

1. **Simple Pendulum Period** -- T(L) = 2 pi sqrt(L/g). Sensitivity dT/dL tells you how much the period changes per unit change in string length. Analytical: dT/dL = pi / sqrt(L g).

2. **Damped Oscillation** -- x(t; gamma) = A exp(-gamma t) cos(omega t). Sensitivity dx/dgamma tells you how the displacement responds to changes in the damping coefficient. Analytical: dx/dgamma = -t A exp(-gamma t) cos(omega t).

## How to Run

```
sbcl --load examples/04-sensitivity/main.lisp
```

Or from a running REPL:

```lisp
(load "examples/04-sensitivity/main.lisp")
```

## Key Takeaway

AD delivers exact parameter sensitivities -- matching analytical derivatives to within machine epsilon (~1e-16) -- without deriving or coding the derivative formula by hand. This makes sensitivity analysis reliable and easy to maintain as models evolve.
