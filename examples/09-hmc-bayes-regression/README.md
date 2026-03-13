# Bayesian Linear Regression with HMC

Demonstrates using cl-acorn's Hamiltonian Monte Carlo (HMC) sampler to estimate
the posterior distribution of parameters in a Bayesian linear regression model.

## The Model

We fit a linear relationship y = w·x + b to observed data with Gaussian noise:

    y_i = w · x_i + b + ε_i,   ε_i ~ N(0, 0.5²)

With priors:
- w ~ N(0, 1)    — slope
- b ~ N(0, 10)   — intercept

Unlike maximum likelihood estimation, Bayesian inference gives the full posterior
distribution p(w, b | data), capturing parameter uncertainty.

## HMC

Hamiltonian Monte Carlo uses gradient information to propose efficient moves in
parameter space. It simulates Hamiltonian dynamics (like a ball rolling in a
potential energy landscape) to explore the posterior with low autocorrelation.

Key parameters:
- `:step-size` — leapfrog integration step size (adapted during warmup)
- `:n-leapfrog` — number of leapfrog steps per proposal
- `:adapt-step-size` — enable dual averaging adaptation during warmup

## Usage

```lisp
(load "examples/09-hmc-bayes-regression/main.lisp")
```

## Expected Output

```
=== HMC Bayesian Linear Regression ===
Model: y = w*x + b,  sigma=0.5
True values: w=2.0, b=1.0
Data: 20 observations

Running HMC (1000 samples, 500 warmup)...

Posterior statistics (1000 samples after 500 warmup):
  w:  mean= 2.012  std= 0.085   [true: 2.0]
  b:  mean= 0.997  std= 0.118   [true: 1.0]
Accept rate: 0.832
```

Results vary due to sampling randomness; posterior means should be close to the true values.
