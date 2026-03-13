# Bayesian Linear Regression with NUTS

Demonstrates using cl-acorn's No-U-Turn Sampler (NUTS) to estimate the posterior
distribution of parameters in a Bayesian linear regression model.

## The Model

    y_i = w · x_i + b + ε_i,   ε_i ~ N(0, 0.5²)

Priors:
- w ~ N(0, 1)    — slope
- b ~ N(0, 10)   — intercept

## NUTS vs HMC

HMC requires tuning the number of leapfrog steps (`:n-leapfrog`). NUTS eliminates
this by building a balanced binary tree of leapfrog steps and stopping when it
detects a U-turn in trajectory direction. This makes NUTS more robust and usually
achieves higher effective sample size per gradient evaluation.

Key parameters:
- `:adapt-step-size` — dual averaging step-size adaptation during warmup (default: t)
- `:max-tree-depth` — upper bound on tree depth (default: 10, i.e. max 1024 steps)

## Usage

```lisp
(load "examples/10-nuts-bayes-regression/main.lisp")
```

## Expected Output

```
=== NUTS Bayesian Linear Regression ===
Model: y = w*x + b,  sigma=0.5
True values: w=2.0, b=1.0
Data: 20 observations

Running NUTS (1000 samples, 500 warmup)...

Posterior statistics (1000 samples after 500 warmup):
  w:  mean= 2.008  std= 0.083   [true: 2.0]
  b:  mean= 1.002  std= 0.115   [true: 1.0]
Accept rate: 0.921
```

Compare with example 09 (HMC) — NUTS typically achieves a higher effective sample
rate without requiring manual tuning of leapfrog steps.
