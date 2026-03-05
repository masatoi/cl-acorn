# FIR Filter Coefficient Optimization via Automatic Differentiation

## What It Demonstrates

This example uses `cl-acorn`'s forward-mode automatic differentiation to optimize the coefficients of a 5-tap FIR (Finite Impulse Response) digital filter. Given a noisy sinusoidal signal, gradient descent minimizes the mean squared error between the filtered output and the original clean signal -- with AD computing the exact gradient automatically.

## Theory

A FIR filter computes each output sample as a weighted sum of recent input samples:

    y[n] = a_0 * x[n] + a_1 * x[n-1] + ... + a_K * x[n-K]

The coefficients `a_0, ..., a_K` determine the filter's frequency response. To find optimal coefficients, we minimize the MSE loss:

    L(a) = (1/N) * sum( (y[n] - clean[n])^2 )

For each coefficient `a_i`, `ad:derivative` computes `dL/da_i` exactly by seeding that coefficient as a dual number with epsilon=1. This gives us the full gradient in 5 passes (one per tap), enabling standard gradient descent without ever deriving the gradient by hand.

## How to Run

```lisp
(load "examples/07-signal-processing/main.lisp")
```

Or from the shell:

```bash
sbcl --load examples/07-signal-processing/main.lisp
```

## Key Takeaway

AD turns any differentiable computation into an optimization target. Here the "model" is a digital filter and the "parameters" are its tap weights, but the same pattern -- wrap parameters with `ad:derivative`, compute loss, extract gradients -- applies to any parameterized signal processing pipeline. The optimized filter achieves over 50% MSE reduction compared to the unfiltered signal, and substantially outperforms a naive uniform averaging filter.
