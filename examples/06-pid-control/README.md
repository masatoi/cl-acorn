# PID Controller Auto-Tuning via Automatic Differentiation

## What It Demonstrates

This example uses `cl-acorn`'s forward-mode automatic differentiation to auto-tune a PID controller by gradient descent. The key insight is that AD can differentiate through an entire simulation loop -- 500 discrete time steps of plant dynamics and controller logic -- to compute exact gradients of a cost function with respect to controller gains.

## Theory

A **PID controller** computes a control signal as:

    u = Kp * e + Ki * integral(e) + Kd * (de/dt)

where `e = setpoint - y` is the tracking error. The plant is a first-order system `G(s) = 1/(s+1)`, discretized via Euler integration:

    y[n+1] = y[n] + (-y[n] + u[n]) * dt

The **Integral of Squared Error** (ISE) measures controller performance:

    ISE = sum(e[n]^2 * dt)

To minimize ISE, we need the gradients `dISE/dKp`, `dISE/dKi`, `dISE/dKd`. Rather than deriving these analytically (which would require differentiating through the entire recurrence), we pass each gain as a dual number through `ad:derivative`. The dual arithmetic automatically propagates derivatives through all 500 loop iterations, yielding exact gradients.

Gradient descent then updates each gain: `K -= lr * dISE/dK`.

## How to Run

```lisp
(load "examples/06-pid-control/main.lisp")
```

Or from the shell:

```bash
sbcl --load examples/06-pid-control/main.lisp
```

## Key Takeaway

Automatic differentiation is not limited to closed-form mathematical expressions. It works through arbitrary program structures -- loops, accumulators, conditionals -- as long as the computation uses AD-aware arithmetic. This makes AD a practical tool for optimizing parameters of simulations, control systems, and other iterative algorithms where analytical derivatives would be impractical to derive.
