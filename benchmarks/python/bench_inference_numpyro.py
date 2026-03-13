"""
NumPyro inference benchmarks for comparison with cl-acorn.

Model: 2D standard normal, log p(x) = -0.5 * (x1^2 + x2^2)
Matches the cl-acorn inference benchmark configuration:
  - HMC:  500 samples, 200 warmup
  - NUTS: 500 samples, 200 warmup
  - VI:   1000 SVI steps with AutoNormal guide (mean-field ADVI equivalent)

Results are reported as samples/sec (or iterations/sec for VI).

Run: python bench_inference_numpyro.py
"""

import time
import os

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

jax.config.update("jax_enable_x64", True)

# Match cl-acorn benchmark configuration
N_SAMPLES = 500
N_WARMUP = 200
N_VI_STEPS = 1000
N_ELBO_SAMPLES = 10
N_TRIALS = 5


def standard_normal_2d_model():
    """2D standard normal: log p(x) = -0.5 * (x1^2 + x2^2)."""
    numpyro.sample("x1", dist.Normal(0.0, 1.0))
    numpyro.sample("x2", dist.Normal(0.0, 1.0))


def run_mcmc_benchmark(kernel_class, kernel_kwargs, label):
    """
    Run MCMC with the given kernel class and measure wall-clock time.
    Returns samples/sec.
    """
    trial_rates = []

    for _ in range(N_TRIALS):
        key = jax.random.PRNGKey(42)
        kernel = kernel_class(**kernel_kwargs)
        mcmc = MCMC(
            kernel,
            num_warmup=N_WARMUP,
            num_samples=N_SAMPLES,
            progress_bar=False,
        )

        # Warmup run (compilation + adaptation) — not timed
        # NumPyro MCMC.run() blocks internally; no extra barrier needed for warmup
        mcmc.run(key, standard_normal_2d_model)

        # Timed run
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        mcmc.run(subkey, standard_normal_2d_model)
        # Block until JAX async ops finish
        jax.effects_barrier()
        elapsed = time.perf_counter() - start

        rate = N_SAMPLES / elapsed
        trial_rates.append(rate)

    mean_rate = sum(trial_rates) / len(trial_rates)
    min_rate = min(trial_rates)
    max_rate = max(trial_rates)
    return mean_rate, min_rate, max_rate


def run_vi_benchmark():
    """
    Run SVI (mean-field ADVI via AutoNormal) and measure iterations/sec.
    """
    trial_rates = []

    for _ in range(N_TRIALS):
        key = jax.random.PRNGKey(42)
        guide = AutoNormal(standard_normal_2d_model)
        optimizer = numpyro.optim.Adam(step_size=0.01)
        svi = SVI(
            standard_normal_2d_model,
            guide,
            optimizer,
            loss=Trace_ELBO(num_particles=N_ELBO_SAMPLES),
        )

        # Warmup: one step to trigger JIT compilation
        svi_state = svi.init(key)
        svi_state, _ = svi.update(svi_state)
        jax.effects_barrier()

        # Timed run
        start = time.perf_counter()
        for _ in range(N_VI_STEPS):
            svi_state, loss = svi.update(svi_state)
        jax.effects_barrier()
        elapsed = time.perf_counter() - start

        rate = N_VI_STEPS / elapsed
        trial_rates.append(rate)

    mean_rate = sum(trial_rates) / len(trial_rates)
    min_rate = min(trial_rates)
    max_rate = max(trial_rates)
    return mean_rate, min_rate, max_rate


def print_table_header():
    print(
        f"{'Task':<35} | {'samples/sec':>12} | {'min s/s':>10} | {'max s/s':>10}"
    )
    print(f"{'-' * 35}-|-{'-' * 12}-|-{'-' * 10}-|-{'-' * 10}")


def print_table_row(task, mean_rate, min_rate, max_rate):
    print(
        f"{task:<35} | {mean_rate:>12.1f} | {min_rate:>10.1f} | {max_rate:>10.1f}"
    )


def main():
    print("NumPyro inference benchmark results")
    print("=" * 72)
    print(
        f"Config: {N_SAMPLES} samples, {N_WARMUP} warmup (MCMC); "
        f"{N_VI_STEPS} steps, {N_ELBO_SAMPLES} ELBO samples (VI)"
    )
    print(f"Trials: {N_TRIALS} — reporting mean/min/max samples-per-second")
    print()
    print_table_header()

    # NUTS
    nuts_mean, nuts_min, nuts_max = run_mcmc_benchmark(
        NUTS,
        {"target_accept_prob": 0.8},
        "nuts-2d-standard-normal",
    )
    print_table_row("nuts-2d-standard-normal", nuts_mean, nuts_min, nuts_max)

    # HMC (step_size and num_steps left at defaults to match cl-acorn defaults)
    hmc_mean, hmc_min, hmc_max = run_mcmc_benchmark(
        HMC,
        {"step_size": 0.1, "num_steps": 10, "target_accept_prob": 0.8},
        "hmc-2d-standard-normal",
    )
    print_table_row("hmc-2d-standard-normal", hmc_mean, hmc_min, hmc_max)

    # VI (SVI with AutoNormal — mean-field ADVI equivalent)
    vi_mean, vi_min, vi_max = run_vi_benchmark()
    # VI reports iterations/sec rather than samples/sec; annotate the label
    print_table_row("vi-2d-standard-normal (iters/s)", vi_mean, vi_min, vi_max)


if __name__ == "__main__":
    main()
