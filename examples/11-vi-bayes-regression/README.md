# Bayesian Linear Regression with VI (ADVI)

Demonstrates using cl-acorn's mean-field Automatic Differentiation Variational
Inference (ADVI) to approximate the posterior distribution of parameters in a
Bayesian linear regression model.

## The Model

    y_i = w · x_i + b + ε_i,   ε_i ~ N(0, 0.5²)

Priors:
- w ~ N(0, 1)    — slope
- b ~ N(0, 10)   — intercept

## Variational Inference vs MCMC

Instead of drawing samples from the posterior (as HMC/NUTS do), VI finds the
best Gaussian approximation q(w, b) = N(μ_w, σ_w) · N(μ_b, σ_b) by maximizing
the Evidence Lower BOund (ELBO). This is faster than MCMC but approximate —
the mean-field assumption (independent Gaussians) misses posterior correlations.

Returns directly: posterior means and standard deviations (no samples needed).

Key parameters:
- `:n-iterations` — Adam optimizer steps
- `:n-elbo-samples` — Monte Carlo samples per ELBO gradient estimate
- `:lr` — Adam learning rate

## Usage

```lisp
(load "examples/11-vi-bayes-regression/main.lisp")
```

## Expected Output

```
=== VI (ADVI) Bayesian Linear Regression ===
Model: y = w*x + b,  sigma=0.5
True values: w=2.0, b=1.0
Data: 20 observations

Running VI (2000 iterations)...

Variational posterior (mean-field Gaussian):
  w:  mean= 2.009  std= 0.088   [true: 2.0]
  b:  mean= 1.001  std= 0.121   [true: 1.0]
Final ELBO: -42.317  (higher is better)
```

Compare with examples 09 (HMC) and 10 (NUTS): VI is faster but gives an
approximate posterior. For this well-conditioned model the difference is small.
