# cl-acorn Examples Design

## Goal

Create 7 practical, E2E-runnable examples demonstrating cl-acorn's forward-mode AD across diverse application domains and AD usage patterns. Target audience: both library users wanting API examples and learners studying automatic differentiation.

## Approach

**Domain-axis organization**: each example represents a distinct application domain with its own directory. AD usage patterns emerge naturally from each domain's needs.

## Directory Structure

```
examples/
  01-curve-fitting/
    README.md
    main.lisp
    data.lisp
  02-neural-network/
    README.md
    main.lisp
    data.lisp
  03-newton-method/
    README.md
    main.lisp
  04-sensitivity/
    README.md
    main.lisp
  05-black-scholes/
    README.md
    main.lisp
  06-pid-control/
    README.md
    main.lisp
  07-signal-processing/
    README.md
    main.lisp
```

## Example Specifications

### 01-curve-fitting/ — Statistics, Gradient Descent

**Domain**: Statistics / regression analysis
**AD pattern**: Gradient descent optimization loop with per-parameter `derivative` calls
**Data**: Iris dataset (sepal-length → sepal-width, 150 points), embedded as Lisp literals in `data.lisp`

**Theory**: Linear regression minimizing MSE loss `L(w,b) = (1/N) Σ (y_i - (w*x_i + b))²`. Each parameter (w, b) is updated via `derivative` holding the other fixed.

**Output**: Iteration log showing loss convergence, final w/b values, comparison with closed-form OLS solution.

### 02-neural-network/ — Machine Learning, MLP Training

**Domain**: Machine learning / classification
**AD pattern**: Per-weight `derivative` calls through a composite function (forward pass + loss)
**Data**: Iris dataset (4 features, 3 classes, 150 samples), embedded in `data.lisp`

**Architecture**: Input(4) → Hidden(8, sigmoid) → Output(3, softmax). Cross-entropy loss.

**Educational value**: Demonstrates that forward-mode AD requires O(num_params) derivative calls — motivating reverse-mode AD (future cl-acorn phase 2). Shows AD's ability to differentiate through arbitrary compositions without manual gradient derivation.

**Output**: Training loss per epoch, final classification accuracy on training set.

### 03-newton-method/ — Numerical Analysis, Root Finding

**Domain**: Numerical analysis
**AD pattern**: Direct use of `derivative` returning both f(x) and f'(x) in Newton-Raphson iteration
**Data**: Hardcoded equations (no external data)

**Problems**:
1. Polynomial: `x³ - 2x - 5 = 0` (classical example, root ≈ 2.0946)
2. Transcendental: `cos(x) = x` (root ≈ 0.7391)

**Educational value**: Simplest possible AD usage. Shows `(multiple-value-bind (fx dfx) (ad:derivative f x) ...)` pattern directly.

**Output**: Iteration table (n, x_n, f(x_n), f'(x_n)) showing quadratic convergence.

### 04-sensitivity/ — Physics, Parameter Sensitivity Analysis

**Domain**: Physics / simulation
**AD pattern**: Parameter sensitivity — `derivative` applied to a physical model to quantify output sensitivity to input parameters
**Data**: Hardcoded parameter values

**Models**:
1. Simple pendulum period: `T(L) = 2π√(L/g)` — sensitivity dT/dL, compared with analytical `π/√(Lg)`
2. Damped oscillation: `x(t;γ) = A·exp(-γt)·cos(ωt)` — sensitivity to damping coefficient γ at various time points

**Educational value**: Shows AD for "what-if" analysis. Hand-computable derivatives allow verification.

**Output**: Parameter-sensitivity tables, AD vs analytical comparison.

### 05-black-scholes/ — Finance, Option Greeks

**Domain**: Quantitative finance
**AD pattern**: First and second-order derivatives (Greeks). Nested `derivative` for Gamma (2nd order).
**Data**: Hardcoded option parameters (S=100, K=100, r=0.05, σ=0.2, T=1.0)

**Implementation**: Black-Scholes call price formula. Normal CDF via rational approximation (Abramowitz & Stegun).

**Greeks computed**:
- Delta (∂C/∂S) — first derivative w.r.t. spot price
- Gamma (∂²C/∂S²) — nested `derivative` for second order
- Vega (∂C/∂σ) — sensitivity to volatility
- Theta (∂C/∂T) — time decay
- Rho (∂C/∂r) — interest rate sensitivity

**Educational value**: Real-world finance application. Demonstrates nested `derivative` for higher-order derivatives. Analytical Greeks available for validation.

**Output**: Greeks table, AD vs analytical comparison, sensitivity surface as parameters vary.

### 06-pid-control/ — Control Engineering, Gain Tuning

**Domain**: Control engineering
**AD pattern**: Differentiating through a simulation loop — gradient of an integral objective w.r.t. controller parameters
**Data**: Simulation-generated (step response of first-order plant)

**Model**: Plant `G(s) = 1/(s+1)` discretized via Euler method. PID controller with gains (Kp, Ki, Kd). Objective: minimize ISE (Integral of Squared Error) over step response.

**Optimization**: Gradient descent on ISE w.r.t. each PID gain using `derivative`. Shows AD propagating through the entire discrete-time simulation.

**Output**: Before/after step response comparison, ISE convergence, optimized gains.

### 07-signal-processing/ — Signal Processing, FIR Filter Optimization

**Domain**: Signal processing / DSP
**AD pattern**: Optimizing filter coefficients by differentiating a signal processing pipeline
**Data**: Synthetic — clean sinusoid + Gaussian noise (Box-Muller generation)

**Model**: 3-tap FIR filter `y[n] = a₀x[n] + a₁x[n-1] + a₂x[n-2]`. Objective: minimize MSE between filtered noisy signal and clean reference.

**Educational value**: Shows AD working through array-indexed convolution operations. Practical DSP application.

**Output**: Before/after signal quality (MSE), optimized filter coefficients, comparison with known optimal (Wiener filter).

## Common Design Principles

### Execution Model
- Each `main.lisp` runs in `cl-user` with package-local-nickname `ad` for `cl-acorn.ad`
- Loads cl-acorn via `(asdf:load-system :cl-acorn)` at top
- Self-contained: `(load "examples/NN-name/main.lisp")` produces complete output
- No ASDF system definition for examples (explicit `load` only)

### Output Style
- Results printed via `format` to `*standard-output*`
- Readable tables with aligned columns
- Theory/analytical values alongside AD-computed values for verification

### Data Strategy
- Iris data: Lisp literals in `data.lisp` (loaded via `load` from `main.lisp`)
- Synthetic data: generated in-place (Box-Muller for noise, simulation for control)
- Hardcoded parameters: inline in `main.lisp`

### Dependencies
- cl-acorn only. No external libraries required.
- Normal CDF approximation, Box-Muller RNG, etc. implemented locally in each example.

### Multi-variable Differentiation Pattern
Since cl-acorn currently provides only univariate `derivative`, examples requiring gradients use the pattern:
```lisp
;; Differentiate loss w.r.t. parameter w, holding b fixed
(derivative (lambda (w) (loss w b data)) current-w)
;; Then w.r.t. b, holding w fixed
(derivative (lambda (b) (loss w b data)) current-b)
```
This pattern is documented in each relevant README as a natural consequence of forward-mode AD.

## Summary Table

| # | Domain | AD Pattern | Data Source | Complexity |
|---|--------|-----------|-------------|------------|
| 01 | Statistics | Gradient descent | Iris (embedded) | Low |
| 02 | ML | Per-weight AD | Iris (embedded) | High |
| 03 | Numerical analysis | Newton-Raphson | Hardcoded | Low |
| 04 | Physics | Sensitivity analysis | Hardcoded | Low |
| 05 | Finance | Greeks (1st+2nd order) | Hardcoded | Medium |
| 06 | Control | Simulation differentiation | Sim-generated | Medium |
| 07 | Signal processing | Pipeline differentiation | Synthetic | Medium |
