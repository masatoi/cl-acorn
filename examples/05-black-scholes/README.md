# Black-Scholes Option Greeks via Automatic Differentiation

## What It Demonstrates

This example computes the five major **option Greeks** (Delta, Gamma, Vega, Theta, Rho) for a European call option under the Black-Scholes model, using `cl-acorn`'s forward-mode automatic differentiation.

Each Greek is a partial derivative of the option price with respect to a different input parameter. Rather than deriving and implementing each closed-form formula separately, AD computes them all from a single pricing function by differentiating with respect to the desired variable.

## Theory

The Black-Scholes call price is:

    C = S * N(d1) - K * exp(-r*T) * N(d2)

where `d1 = [log(S/K) + (r + vol^2/2)*T] / (vol*sqrt(T))` and `d2 = d1 - vol*sqrt(T)`.

The **Greeks** measure sensitivity to each parameter:

| Greek | Definition | Measures |
|-------|-----------|----------|
| Delta | dC/dS     | Spot price sensitivity |
| Gamma | d^2C/dS^2 | Convexity / Delta sensitivity |
| Vega  | dC/dvol   | Volatility sensitivity |
| Theta | -dC/dT    | Time decay |
| Rho   | dC/dr     | Interest rate sensitivity |

Delta, Vega, Theta, and Rho are computed directly via `ad:derivative`. Gamma (the second derivative) uses a central difference on the AD-computed Delta, since the library does not yet support nested dual numbers (hyper-duals).

The normal CDF uses the Abramowitz & Stegun rational approximation (7.1.26), which introduces errors at the 1e-7 level -- well within practical tolerance for option pricing.

## How to Run

```lisp
(load "examples/05-black-scholes/main.lisp")
```

Or from the shell:

```bash
sbcl --load examples/05-black-scholes/main.lisp
```

## Key Takeaway

Automatic differentiation lets you compute exact derivatives of arbitrarily complex pricing models without manual calculus. Write the pricing function once using `ad:` arithmetic, and `ad:derivative` gives you any first-order Greek for free. This approach scales to models far more complex than Black-Scholes, where closed-form Greeks may not exist.
