"""
PyTorch automatic differentiation benchmarks for comparison with cl-acorn.

Target function: f(x) = sum(x_i^2), analytic gradient: 2*x
All computations use float64 (torch.double) to match cl-acorn's double-float precision.

Timing covers the combined forward + backward pass because the two are inseparable
in typical PyTorch usage: you must call forward before backward, and the backward
populates .grad on the leaf tensors.  The table notes this explicitly.

Run: python bench_ad_torch.py
"""

import time
import torch
from torch.autograd import forward_ad


def f(x):
    """Target function: sum(x_i^2)."""
    return (x ** 2).sum()


def make_input(n, requires_grad=True):
    """Create a float64 ones tensor of dimension n."""
    return torch.ones(n, dtype=torch.float64, requires_grad=requires_grad)


def time_task(fn, n_runs, n_trials=5):
    """
    Run fn() n_runs times per trial, n_trials trials total.
    Returns (mean_us, min_us, max_us) in microseconds across trials.
    """
    trial_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for _ in range(n_runs):
            fn()
        elapsed = time.perf_counter() - start
        trial_times.append(elapsed / n_runs * 1e6)
    mean_us = sum(trial_times) / len(trial_times)
    min_us = min(trial_times)
    max_us = max(trial_times)
    return mean_us, min_us, max_us


def print_table_header():
    print(f"{'Task':<30} | {'Mean (us)':>10} | {'Min (us)':>9} | {'Max (us)':>9}")
    print(f"{'-' * 30}-|-{'-' * 10}-|-{'-' * 9}-|-{'-' * 9}")


def print_table_row(task, mean_us, min_us, max_us):
    print(f"{task:<30} | {mean_us:>10.1f} | {min_us:>9.1f} | {max_us:>9.1f}")


def grad_task(n):
    """Return a closure that times forward + backward for dimension n."""
    def run():
        x = make_input(n, requires_grad=True)
        loss = f(x)
        loss.backward()
    return run


def warmup(task_fn, n_warmup):
    for _ in range(n_warmup):
        task_fn()


def derivative_1d_task():
    """Return a closure that computes df/dx via forward-mode AD (dim=1)."""
    x = torch.tensor(1.0, dtype=torch.float64)
    v = torch.tensor(1.0, dtype=torch.float64)

    def run():
        with forward_ad.dual_level():
            dual_x = forward_ad.make_dual(x, v)
            dual_out = dual_x * dual_x
            forward_ad.unpack_dual(dual_out)

    return run


def run_ad_benchmarks():
    results = []

    # --- derivative-1d (forward-mode, dim=1) ---
    task_deriv_1d = derivative_1d_task()
    # Warmup amortizes Python overhead; PyTorch eager mode doesn't JIT-compile
    warmup(task_deriv_1d, 200)
    mean_us, min_us, max_us = time_task(task_deriv_1d, n_runs=10000)
    results.append(("derivative-1d (forward-mode)", mean_us, min_us, max_us))

    # --- gradient-1d (forward + backward, dim=1) ---
    task_1d = grad_task(1)
    # Warmup amortizes Python overhead; PyTorch eager mode doesn't JIT-compile
    warmup(task_1d, 200)
    mean_us, min_us, max_us = time_task(task_1d, n_runs=10000)
    results.append(("gradient-1d (fwd+bwd)", mean_us, min_us, max_us))

    # --- gradient-10d ---
    task_10d = grad_task(10)
    warmup(task_10d, 200)
    mean_us, min_us, max_us = time_task(task_10d, n_runs=1000)
    results.append(("gradient-10d (fwd+bwd)", mean_us, min_us, max_us))

    # --- gradient-100d ---
    task_100d = grad_task(100)
    warmup(task_100d, 200)
    mean_us, min_us, max_us = time_task(task_100d, n_runs=500)
    results.append(("gradient-100d (fwd+bwd)", mean_us, min_us, max_us))

    # --- gradient-1000d ---
    task_1000d = grad_task(1000)
    warmup(task_1000d, 100)
    mean_us, min_us, max_us = time_task(task_1000d, n_runs=100)
    results.append(("gradient-1000d (fwd+bwd)", mean_us, min_us, max_us))

    # --- hessian-vector-product-10d ---
    # PyTorch forward-over-reverse HVP via functional API
    x10 = torch.ones(10, dtype=torch.float64)
    v10 = torch.ones(10, dtype=torch.float64)

    def hvp_10d():
        torch.autograd.functional.jvp(
            lambda x: torch.autograd.grad(f(x), x, create_graph=True)[0],
            (x10,),
            (v10,),
        )

    warmup(hvp_10d, 200)
    mean_us, min_us, max_us = time_task(hvp_10d, n_runs=500)
    results.append(("hessian-vector-product-10d", mean_us, min_us, max_us))

    return results


def main():
    print("PyTorch benchmark results")
    print("=" * 62)
    print("Note: gradient tasks time the combined forward + backward pass.")
    print()
    print_table_header()

    for task, mean_us, min_us, max_us in run_ad_benchmarks():
        print_table_row(task, mean_us, min_us, max_us)


if __name__ == "__main__":
    main()
