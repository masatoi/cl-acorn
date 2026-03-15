# cl-acorn

Automatic differentiation, probability distributions, and Bayesian inference building blocks for Common Lisp.

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

The ASDF system has no external library dependencies. Clone this repository to a location visible to ASDF/Quicklisp:

```bash
cd ~/common-lisp/  # or ~/quicklisp/local-projects/
git clone https://github.com/masatoi/cl-acorn.git
```

```lisp
(ql:quickload :cl-acorn)
```

The inference implementation uses SBCL floating-point utilities (`sb-ext:float-nan-p`, `sb-ext:float-infinity-p`, `sb-int:with-float-traps-masked`) and **requires SBCL**.

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

### Probability Distributions

All symbols are exported from `cl-acorn.distributions` (nickname: `dist`). Most log-PDF parameters are AD-transparent (accept dual numbers and tape-nodes). **Exception**: `gamma-log-pdf :shape` and `beta-log-pdf :alpha`/`:beta` use `log-gammaln` internally, which is not AD-transparent; differentiate through `x` (the observation variable) instead.

| Utility | Description |
|---------|-------------|
| `dist:log-gammaln` | Log gamma function helper used by several distributions |
| `dist:+log-pdf-sentinel+` | Large negative finite sentinel returned by some out-of-support log-PDF paths |

| Distribution | Log-PDF | Sample |
|-------------|---------|--------|
| Normal | `(dist:normal-log-pdf x :mu 0 :sigma 1)` | `(dist:normal-sample ...)` |
| Gamma | `(dist:gamma-log-pdf x :shape 2 :rate 1)` | `(dist:gamma-sample ...)` |
| Beta | `(dist:beta-log-pdf x :alpha 2 :beta 3)` | `(dist:beta-sample ...)` |
| Uniform | `(dist:uniform-log-pdf x :low 0 :high 1)` | `(dist:uniform-sample ...)` |
| Bernoulli | `(dist:bernoulli-log-pdf x :prob 0.7)` | `(dist:bernoulli-sample ...)` |
| Poisson | `(dist:poisson-log-pdf k :rate 5)` | `(dist:poisson-sample ...)` |

### Optimizers

All symbols are exported from `cl-acorn.optimizers` (nickname: `opt`).

| Optimizer | Usage |
|----------|-------|
| SGD | `(opt:sgd-step params grads :lr 0.01)` |
| Adam | `(opt:adam-step params grads state :lr 0.001)` |

```lisp
;; Adam requires a state object
(defvar *state* (opt:make-adam-state n-params))
(opt:adam-step params grads *state* :lr 0.001d0)
```

`opt:adam-step` updates the Adam state in place and returns a new parameter list.

### Bayesian Inference

All symbols are exported from `cl-acorn.inference` (nickname: `infer`).

```lisp
(infer:hmc log-pdf-fn initial-params
  :n-samples 1000 :n-warmup 500
  :step-size 0.01d0 :n-leapfrog 10)
;; => (values samples accept-rate diagnostics)
```

```lisp
(infer:nuts log-pdf-fn initial-params
  :n-samples 1000 :n-warmup 500
  :step-size 0.01d0 :max-tree-depth 10)
;; => (values samples accept-rate diagnostics)
```

```lisp
(infer:vi log-pdf-fn initial-params
  :n-iterations 1000 :n-elbo-samples 10 :lr 0.01d0)
;; => (values mu-list sigma-list elbo-history diagnostics)
```

`infer:hmc` uses a **fixed step size by default** (`:adapt-step-size nil`); pass `:adapt-step-size t` to enable Nesterov dual averaging during warmup. `infer:nuts` adapts step size by default. Both return an `infer:inference-diagnostics` struct as their third value. `infer:vi` takes a list of initial parameter values and returns posterior means, posterior standard deviations, ELBO history, and diagnostics.

Common exported condition and restart APIs include:

| Category | Symbols |
|----------|---------|
| Base conditions | `infer:acorn-error`, `infer:acorn-error-message`, `infer:model-error`, `infer:inference-error` |
| Errors | `infer:invalid-parameter-error`, `infer:log-pdf-domain-error`, `infer:invalid-initial-params-error` |
| Informational signals | `infer:non-finite-gradient-error`, `infer:non-finite-gradient-error-params`, `infer:non-finite-gradient-error-message` |
| Warnings | `infer:high-divergence-warning` |
| Restarts | `infer:use-fallback-params`, `infer:return-empty-samples`, `infer:continue-with-warnings` |
| Diagnostics | `infer:inference-diagnostics`, `infer:inference-diagnostics-p`, `infer:diagnostics-accept-rate`, `infer:diagnostics-n-divergences`, `infer:diagnostics-final-step-size`, `infer:diagnostics-n-samples`, `infer:diagnostics-n-warmup`, `infer:diagnostics-elapsed-seconds` |

`infer:acorn-error` is the root condition for all library errors. Use `(handler-case ... (infer:acorn-error (c) ...))` to catch any error raised by cl-acorn inference functions. `infer:model-error` and `infer:inference-error` are subclasses for model-definition and sampler failures respectively. `infer:acorn-error-message` retrieves the human-readable message from any `acorn-error` subtype.

`infer:non-finite-gradient-error` is a **plain condition** (not an error subtype) that is `signal`-ed — not `error`-ed — when a gradient computation yields non-finite values. Because it does not inherit from `error`, a `handler-case` `error` clause will never observe it; use `handler-bind` instead. The sampler automatically skips non-finite-gradient steps and continues sampling; user handlers may observe but should not perform non-local exits during `run-chains`.

### MCMC Diagnostics

All symbols are exported from `cl-acorn.diagnostics` (nickname: `diag`).

**Multi-Chain Execution:**

```lisp
(diag:run-chains my-log-posterior '(0.0d0 0.0d0)
                 :n-chains 4 :n-samples 1000 :n-warmup 500)
;; => chain-result with R-hat, ESS, accept-rates, n-divergences
```

`run-chains` accepts `:sampler :nuts` (default) or `:sampler :hmc`, and returns a `chain-result` struct with fields:

| Accessor | Description |
|----------|-------------|
| `diag:chain-result-samples` | List of per-chain sample lists |
| `diag:chain-result-n-chains` | Number of chains |
| `diag:chain-result-n-samples` | Post-warmup samples per chain |
| `diag:chain-result-n-warmup` | Warmup samples per chain |
| `diag:chain-result-r-hat` | Per-parameter R-hat values |
| `diag:chain-result-bulk-ess` | Per-parameter bulk ESS |
| `diag:chain-result-tail-ess` | Per-parameter tail ESS |
| `diag:chain-result-accept-rates` | Per-chain acceptance rates |
| `diag:chain-result-n-divergences` | Total divergent transitions |
| `diag:chain-result-elapsed-seconds` | Wall-clock time |

**Convergence Diagnostics:**

```lisp
(diag:print-convergence-summary result)
;; Convergence diagnostics  (4 chains x 1000 samples)
;; ====================================================
;; Param | R-hat  | Bulk-ESS | Tail-ESS | Status
;; ...
```

Individual diagnostic functions (each takes a list of per-chain sample lists):

```lisp
(diag:r-hat chains)     ;; => list of R-hat values (one per parameter)
(diag:bulk-ess chains)  ;; => list of bulk ESS values
(diag:tail-ess chains)  ;; => list of tail ESS values
```

**Model Comparison:**

```lisp
(diag:waic result log-lik-fn data)
;; => (values waic p-waic lppd)

(diag:loo result log-lik-fn data)
;; => (values loo p-loo k-hats)

(diag:print-model-comparison
  "model-a" result-a log-lik-a data
  "model-b" result-b log-lik-b data)
;; Model comparison
;; ====================================================
;; Model    | WAIC   | p_waic | LOO    | p_loo
;; ...
;; Lower is better.
```

`log-lik-fn` is a function `(lambda (params data-point) -> log-likelihood)` using plain arithmetic (not AD-transparent).

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

### Bayesian Inference with HMC

```lisp
;; Infer parameters of a normal distribution from data
(defvar *data* '(2.1d0 1.8d0 2.3d0 1.9d0 2.0d0))

(defun model-log-pdf (params)
  (let ((mu (first params))
        (log-sigma (second params)))
    (let ((sigma (ad:exp log-sigma))
          (ll 0.0d0))
      (dolist (x *data*)
        (setf ll (ad:+ ll (dist:normal-log-pdf x :mu mu :sigma sigma))))
      ;; Add priors
      (ad:+ ll
            (dist:normal-log-pdf mu :mu 0.0d0 :sigma 10.0d0)
            (dist:normal-log-pdf log-sigma :mu 0.0d0 :sigma 2.0d0)))))

(multiple-value-bind (samples accept-rate diagnostics)
    (infer:hmc #'model-log-pdf '(0.0d0 0.0d0)
      :n-samples 2000 :n-warmup 500
      :step-size 0.01d0 :n-leapfrog 20)
  ;; samples: list of (mu, log-sigma) parameter vectors
  ;; accept-rate: fraction of accepted proposals
  ;; diagnostics: inference summary struct
  )
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
| [09-hmc-bayes-regression](examples/09-hmc-bayes-regression/) | Bayesian linear regression with Hamiltonian Monte Carlo |
| [10-nuts-bayes-regression](examples/10-nuts-bayes-regression/) | Bayesian linear regression with the No-U-Turn Sampler |
| [11-vi-bayes-regression](examples/11-vi-bayes-regression/) | Bayesian linear regression with mean-field variational inference |
| [12-football-goals](examples/12-football-goals/) | Baseline Poisson model for international football goal counts with multi-chain NUTS and convergence diagnostics |
| [13-football-hierarchy](examples/13-football-hierarchy/) | Hierarchical Poisson model for national team strengths with non-centered parameterization, partial pooling, and flat vs hierarchical WAIC comparison |

Run any example:

```lisp
(load "examples/01-curve-fitting/main.lisp")
```

## Running Tests

cl-acorn uses [Rove](https://github.com/fukamachi/rove) for testing (currently 238 tests).

```bash
rove cl-acorn.asd
```

Or from the REPL:

```lisp
(asdf:test-system :cl-acorn)
```

## License

MIT
