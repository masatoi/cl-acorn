"""
JAX automatic differentiation benchmarks for comparison with cl-acorn.

Target function: f(x) = sum(x_i^2), analytic gradient: 2*x
All computations use float64 to match cl-acorn's double-float precision.

Run: python bench_ad_jax.py
"""

import time
import os

# Force float64 before importing jax
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def make_f(n):
    """Return f(x) = sum(x_i^2) for an n-dimensional input."""
    def f(x):
        return jnp.sum(x ** 2)
    return f


def time_task(fn, n_runs, n_trials=5):
    """
    Run fn() n_runs times per trial, n_trials trials total.
    Returns (mean_us, min_us, max_us) across trials.
    """
    trial_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for _ in range(n_runs):
            result = fn()
            # Block until JAX computation completes
            result.block_until_ready()
        elapsed = time.perf_counter() - start
        trial_times.append(elapsed / n_runs * 1e6)  # convert to microseconds
    mean_us = sum(trial_times) / len(trial_times)
    min_us = min(trial_times)
    max_us = max(trial_times)
    return mean_us, min_us, max_us


def print_table_header():
    print(f"{'Task':<30} | {'Mean (us)':>10} | {'Min (us)':>9} | {'Max (us)':>9}")
    print(f"{'-' * 30}-|-{'-' * 10}-|-{'-' * 9}-|-{'-' * 9}")


def print_table_row(task, mean_us, min_us, max_us):
    print(f"{task:<30} | {mean_us:>10.1f} | {min_us:>9.1f} | {max_us:>9.1f}")


def run_ad_benchmarks():
    results = []

    # --- derivative-1d: forward-mode scalar derivative ---
    # JAX jvp is the forward-mode analogue of cl-acorn's derivative
    f1 = make_f(1)
    x1 = jnp.ones(1, dtype=jnp.float64)
    v1 = jnp.ones(1, dtype=jnp.float64)

    jit_jvp_1d = jax.jit(lambda x, v: jax.jvp(f1, (x,), (v,)))
    # Warmup to trigger JIT compilation
    for _ in range(200):
        primals, tangents = jit_jvp_1d(x1, v1)
        tangents.block_until_ready()

    def task_derivative_1d():
        _, tangents = jit_jvp_1d(x1, v1)
        return tangents

    mean_us, min_us, max_us = time_task(task_derivative_1d, n_runs=10000)
    results.append(("derivative-1d", mean_us, min_us, max_us))

    # --- gradient-1d ---
    grad_f1 = jax.jit(jax.grad(f1))
    for _ in range(200):
        grad_f1(x1).block_until_ready()

    def task_gradient_1d():
        return grad_f1(x1)

    mean_us, min_us, max_us = time_task(task_gradient_1d, n_runs=10000)
    results.append(("gradient-1d", mean_us, min_us, max_us))

    # --- gradient-10d ---
    f10 = make_f(10)
    x10 = jnp.ones(10, dtype=jnp.float64)
    grad_f10 = jax.jit(jax.grad(f10))
    for _ in range(200):
        grad_f10(x10).block_until_ready()

    def task_gradient_10d():
        return grad_f10(x10)

    mean_us, min_us, max_us = time_task(task_gradient_10d, n_runs=1000)
    results.append(("gradient-10d", mean_us, min_us, max_us))

    # --- gradient-100d ---
    f100 = make_f(100)
    x100 = jnp.ones(100, dtype=jnp.float64)
    grad_f100 = jax.jit(jax.grad(f100))
    for _ in range(200):
        grad_f100(x100).block_until_ready()

    def task_gradient_100d():
        return grad_f100(x100)

    mean_us, min_us, max_us = time_task(task_gradient_100d, n_runs=500)
    results.append(("gradient-100d", mean_us, min_us, max_us))

    # --- gradient-1000d ---
    f1000 = make_f(1000)
    x1000 = jnp.ones(1000, dtype=jnp.float64)
    grad_f1000 = jax.jit(jax.grad(f1000))
    for _ in range(100):
        grad_f1000(x1000).block_until_ready()

    def task_gradient_1000d():
        return grad_f1000(x1000)

    mean_us, min_us, max_us = time_task(task_gradient_1000d, n_runs=100)
    results.append(("gradient-1000d", mean_us, min_us, max_us))

    # --- hessian-vector-product-10d ---
    # HVP: jvp(grad(f), x, v) — forward-over-reverse mode
    grad_f10_plain = jax.grad(f10)
    jit_hvp_10d = jax.jit(lambda x, v: jax.jvp(grad_f10_plain, (x,), (v,))[1])
    v10 = jnp.ones(10, dtype=jnp.float64)
    for _ in range(200):
        jit_hvp_10d(x10, v10).block_until_ready()

    def task_hvp_10d():
        return jit_hvp_10d(x10, v10)

    mean_us, min_us, max_us = time_task(task_hvp_10d, n_runs=500)
    results.append(("hessian-vector-product-10d", mean_us, min_us, max_us))

    return results


def run_distribution_benchmarks():
    results = []

    x_scalar = jnp.float64(1.0)
    zero = jnp.float64(0.0)
    one = jnp.float64(1.0)
    two = jnp.float64(2.0)
    half = jnp.float64(0.5)

    # --- normal-log-pdf ---
    jit_normal = jax.jit(
        lambda x: jax.scipy.stats.norm.logpdf(x, loc=zero, scale=one)
    )
    for _ in range(200):
        jit_normal(x_scalar).block_until_ready()

    def task_normal():
        return jit_normal(x_scalar)

    mean_us, min_us, max_us = time_task(task_normal, n_runs=10000)
    results.append(("normal-log-pdf", mean_us, min_us, max_us))

    # --- gamma-log-pdf (shape=2, rate=1 => scale=1) ---
    # jax.scipy.stats.gamma uses (a, loc, scale) parameterization
    jit_gamma = jax.jit(
        lambda x: jax.scipy.stats.gamma.logpdf(x, a=two, loc=zero, scale=one)
    )
    for _ in range(200):
        jit_gamma(x_scalar).block_until_ready()

    def task_gamma():
        return jit_gamma(x_scalar)

    mean_us, min_us, max_us = time_task(task_gamma, n_runs=10000)
    results.append(("gamma-log-pdf", mean_us, min_us, max_us))

    # --- beta-log-pdf (alpha=2, beta=2) ---
    jit_beta = jax.jit(
        lambda x: jax.scipy.stats.beta.logpdf(x, a=two, b=two)
    )
    x_beta = jnp.float64(0.5)
    for _ in range(200):
        jit_beta(x_beta).block_until_ready()

    def task_beta():
        return jit_beta(x_beta)

    mean_us, min_us, max_us = time_task(task_beta, n_runs=10000)
    results.append(("beta-log-pdf", mean_us, min_us, max_us))

    # --- poisson-log-pdf (rate=3, k=2) ---
    jit_poisson = jax.jit(
        lambda k: jax.scipy.stats.poisson.logpmf(k, mu=jnp.float64(3.0))
    )
    k_val = jnp.int32(2)
    for _ in range(200):
        jit_poisson(k_val).block_until_ready()

    def task_poisson():
        return jit_poisson(k_val)

    mean_us, min_us, max_us = time_task(task_poisson, n_runs=10000)
    results.append(("poisson-log-pdf", mean_us, min_us, max_us))

    return results


def main():
    print("JAX benchmark results")
    print("=" * 62)
    print_table_header()

    print("-- AD benchmarks --")
    for task, mean_us, min_us, max_us in run_ad_benchmarks():
        print_table_row(task, mean_us, min_us, max_us)

    print("-- Distribution benchmarks --")
    for task, mean_us, min_us, max_us in run_distribution_benchmarks():
        print_table_row(task, mean_us, min_us, max_us)


if __name__ == "__main__":
    main()
