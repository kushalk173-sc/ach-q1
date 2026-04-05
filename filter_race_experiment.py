"""Minimal experiment for bounded-resource multiscale latent-state inference.

This script operationalizes the scoped comparison described in
`admissible_classes.md`:

- Environment:
  - slow latent context `theta_t in {-1, +1}` with switch rate `eps`
  - fast latent event `z_t in {-1, +1}` with switch rate `rho`
  - shared observation stream
    `y_t = a * theta_t + b * z_t + eta_t`, `eta_t ~ N(0, sigma^2)`
- Task:
  - online filtering of both latents from `y_{1:t}`
- Competitors:
  - dual-timescale code: a filter with separate effective transition rates for the
    slow and fast latent
  - single-timescale rival: the same 4-state architecture constrained to one
    common transition rate `tau`, selected by sweep

The experiment is intentionally scoped. It does *not* prove a universal
impossibility theorem against every conceivable single-timescale competitor.
Instead, it measures the advantage of a matched dual-timescale representation
over an equal-access, single-timescale rival class on the minimal 2-factor
latent filtering problem.

This is closest in spirit to a tiny FHMM-style setting: two simultaneous hidden
causes jointly generate one observation stream. HHMMs and HSMMs are natural
secondary enrichments, but this script is the minimal computational testbed for
the conjecture

    inf_{X in C1(R)} L(X) > inf_{(X^s, X^f) in C2(R)} L(X^s, X^f)

in a regime with `eps << rho`, under an explicit filtering loss and matched
resource interpretation.

`best_single_filter` τ sweeps are vectorized (one pass, all τ at once) — much
faster than per-τ loops. Optional Numba JIT still accelerates single-sequence
`dual_filter` / `single_filter` / `single_filter_with_beliefs` when installed.

Parallelism: use `--workers` > 1 to run independent seeds in separate processes
(Windows-safe). Each worker sets BLAS/OpenMP to 1 thread to avoid oversubscription.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit

    _HAVE_NUMBA = True

    @njit(cache=True)
    def _build_transition_core(eps, rho):
        th_s = np.array([-1.0, -1.0, 1.0, 1.0])
        z_s = np.array([-1.0, 1.0, -1.0, 1.0])
        transition = np.empty((4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                p_th = eps if th_s[j] != th_s[i] else 1.0 - eps
                p_z = rho if z_s[j] != z_s[i] else 1.0 - rho
                transition[j, i] = p_th * p_z
        return transition

    @njit(cache=True)
    def _filter_forward_core(y, transition, state_means, sigma, theta_vals, z_vals):
        """Same math as the NumPy reference loop; float64 throughout."""
        T = y.shape[0]
        n = 4
        belief = np.empty(n, dtype=np.float64)
        for k in range(n):
            belief[k] = 1.0 / n
        theta_est = np.empty(T, dtype=np.float64)
        z_est = np.empty(T, dtype=np.float64)
        inv_sig = 1.0 / sigma
        for t in range(T):
            new_b = np.empty(n, dtype=np.float64)
            for j in range(n):
                s = 0.0
                for i in range(n):
                    s += transition[j, i] * belief[i]
                new_b[j] = s
            belief = new_b

            obs = y[t]
            mxl = -1.0e300
            for k in range(n):
                d = (obs - state_means[k]) * inv_sig
                v = -0.5 * d * d
                if v > mxl:
                    mxl = v
            ssum = 0.0
            for k in range(n):
                d = (obs - state_means[k]) * inv_sig
                v = -0.5 * d * d
                belief[k] *= np.exp(v - mxl)
                ssum += belief[k]
            inv = 1.0 / ssum
            for k in range(n):
                belief[k] *= inv

            te = 0.0
            ze = 0.0
            for k in range(n):
                te += belief[k] * theta_vals[k]
                ze += belief[k] * z_vals[k]
            theta_est[t] = te
            z_est[t] = ze
        return theta_est, z_est

    @njit(cache=True)
    def _filter_with_beliefs_core(y, transition, state_means, sigma, theta_vals, z_vals):
        T = y.shape[0]
        n = 4
        belief = np.empty(n, dtype=np.float64)
        for k in range(n):
            belief[k] = 1.0 / n
        theta_est = np.empty(T, dtype=np.float64)
        z_est = np.empty(T, dtype=np.float64)
        beliefs = np.empty((T, n), dtype=np.float64)
        inv_sig = 1.0 / sigma
        for t in range(T):
            new_b = np.empty(n, dtype=np.float64)
            for j in range(n):
                s = 0.0
                for i in range(n):
                    s += transition[j, i] * belief[i]
                new_b[j] = s
            belief = new_b

            obs = y[t]
            mxl = -1.0e300
            for k in range(n):
                d = (obs - state_means[k]) * inv_sig
                v = -0.5 * d * d
                if v > mxl:
                    mxl = v
            ssum = 0.0
            for k in range(n):
                d = (obs - state_means[k]) * inv_sig
                v = -0.5 * d * d
                belief[k] *= np.exp(v - mxl)
                ssum += belief[k]
            inv = 1.0 / ssum
            for k in range(n):
                belief[k] *= inv

            for k in range(n):
                beliefs[t, k] = belief[k]

            te = 0.0
            ze = 0.0
            for k in range(n):
                te += belief[k] * theta_vals[k]
                ze += belief[k] * z_vals[k]
            theta_est[t] = te
            z_est[t] = ze
        return theta_est, z_est, beliefs

except ImportError:
    _HAVE_NUMBA = False
    _build_transition_core = None
    _filter_forward_core = None
    _filter_with_beliefs_core = None


STATES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def simulate(T, eps, rho, a, b, sigma, seed=0):
    rng = np.random.default_rng(seed)

    theta = np.zeros(T, dtype=int)
    z = np.zeros(T, dtype=int)
    theta[0] = rng.choice([-1, 1])
    z[0] = rng.choice([-1, 1])

    for t in range(1, T):
        theta[t] = -theta[t - 1] if rng.random() < eps else theta[t - 1]
        z[t] = -z[t - 1] if rng.random() < rho else z[t - 1]

    y = a * theta + b * z + rng.normal(0, sigma, T)
    return theta, z, y


def build_transition_matrix(eps, rho):
    n = len(STATES)
    transition = np.zeros((n, n), dtype=float)

    for i, (th1, z1) in enumerate(STATES):
        for j, (th2, z2) in enumerate(STATES):
            p_th = eps if th2 != th1 else 1.0 - eps
            p_z = rho if z2 != z1 else 1.0 - rho
            transition[j, i] = p_th * p_z

    return transition


def dual_filter(y, eps, rho, a, b, sigma):
    y = np.asarray(y, dtype=np.float64)
    state_means = np.array([a * th + b * z for th, z in STATES], dtype=np.float64)
    theta_vals = np.array([th for th, _ in STATES], dtype=np.float64)
    z_vals = np.array([z for _, z in STATES], dtype=np.float64)
    sigma = float(sigma)

    if _HAVE_NUMBA:
        transition = _build_transition_core(eps, rho)
        return _filter_forward_core(y, transition, state_means, sigma, theta_vals, z_vals)

    transition = build_transition_matrix(eps, rho)
    belief = np.ones(len(STATES), dtype=float) / len(STATES)
    theta_est = np.empty(len(y), dtype=float)
    z_est = np.empty(len(y), dtype=float)

    for t, obs in enumerate(y):
        belief = transition @ belief
        log_likelihood = -0.5 * ((obs - state_means) / sigma) ** 2
        likelihood = np.exp(log_likelihood - np.max(log_likelihood))
        belief *= likelihood
        belief /= belief.sum()

        theta_est[t] = belief @ theta_vals
        z_est[t] = belief @ z_vals

    return theta_est, z_est


def dual_filter_with_beliefs(y, eps, rho, a, b, sigma):
    """Dual-timescale filter with full belief trajectory (for diagnostics only).

    Not part of the single-timescale Q3 protocol. Used to test whether a GRU
    can decode from *optimal* dual beliefs vs single-filter beliefs.
    """
    y = np.asarray(y, dtype=np.float64)
    state_means = np.array([a * th + b * z for th, z in STATES], dtype=np.float64)
    theta_vals = np.array([th for th, _ in STATES], dtype=np.float64)
    z_vals = np.array([z for _, z in STATES], dtype=np.float64)
    sigma = float(sigma)

    if _HAVE_NUMBA:
        transition = _build_transition_core(eps, rho)
        return _filter_with_beliefs_core(
            y, transition, state_means, sigma, theta_vals, z_vals
        )

    transition = build_transition_matrix(eps, rho)
    belief = np.ones(len(STATES), dtype=float) / len(STATES)
    theta_est = np.empty(len(y), dtype=float)
    z_est = np.empty(len(y), dtype=float)
    beliefs = np.empty((len(y), len(STATES)), dtype=float)

    for t, obs in enumerate(y):
        belief = transition @ belief
        log_ll = -0.5 * ((obs - state_means) / sigma) ** 2
        ll = np.exp(log_ll - np.max(log_ll))
        belief *= ll
        belief /= belief.sum()
        beliefs[t] = belief
        theta_est[t] = belief @ theta_vals
        z_est[t] = belief @ z_vals

    return theta_est, z_est, beliefs


def single_filter(y, tau, a, b, sigma):
    return dual_filter(y, eps=tau, rho=tau, a=a, b=b, sigma=sigma)


def single_filter_with_beliefs(y, tau, a, b, sigma):
    """Single-timescale filter that also returns belief states.

    Returns:
        theta_est: (T,) marginal estimates of theta
        z_est: (T,) marginal estimates of z
        beliefs: (T, 4) full belief vector at each step — this is c_t,
                 the ONLY thing rescue decoders may receive. Never y.
    """
    y = np.asarray(y, dtype=np.float64)
    state_means = np.array([a * th + b * z for th, z in STATES], dtype=np.float64)
    theta_vals = np.array([th for th, _ in STATES], dtype=np.float64)
    z_vals = np.array([z for _, z in STATES], dtype=np.float64)
    sigma = float(sigma)

    if _HAVE_NUMBA:
        transition = _build_transition_core(tau, tau)
        return _filter_with_beliefs_core(
            y, transition, state_means, sigma, theta_vals, z_vals
        )

    transition = build_transition_matrix(tau, tau)
    belief = np.ones(len(STATES), dtype=float) / len(STATES)
    theta_est = np.empty(len(y), dtype=float)
    z_est = np.empty(len(y), dtype=float)
    beliefs = np.empty((len(y), len(STATES)), dtype=float)

    for t, obs in enumerate(y):
        belief = transition @ belief
        log_ll = -0.5 * ((obs - state_means) / sigma) ** 2
        ll = np.exp(log_ll - np.max(log_ll))
        belief *= ll
        belief /= belief.sum()
        beliefs[t] = belief
        theta_est[t] = belief @ theta_vals
        z_est[t] = belief @ z_vals

    return theta_est, z_est, beliefs


def mse_components(theta_true, z_true, theta_hat, z_hat, burnin=0):
    theta_mse = np.mean((theta_hat[burnin:] - theta_true[burnin:]) ** 2)
    z_mse = np.mean((z_hat[burnin:] - z_true[burnin:]) ** 2)
    return theta_mse, z_mse, theta_mse + z_mse


def batched_single_filter_mse(y, theta_true, z_true, taus, a, b, sigma, burnin):
    """All single-timescale filters for `taus` in one vectorized pass (BLAS-friendly).

    Emission does not depend on tau, so the likelihood vector is shared across
    candidates; only transitions differ. Same numerics as separate `single_filter`
    calls, up to float rounding.
    """
    y = np.asarray(y, dtype=np.float64)
    theta_true = np.asarray(theta_true, dtype=np.float64)
    z_true = np.asarray(z_true, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    B = taus.shape[0]
    T = y.shape[0]
    sigma = float(sigma)

    transitions = np.stack(
        [build_transition_matrix(float(tau), float(tau)) for tau in taus], axis=0
    )
    state_means = np.array([a * th + b * z for th, z in STATES], dtype=np.float64)
    theta_vals = np.array([th for th, _ in STATES], dtype=np.float64)
    z_vals = np.array([z for _, z in STATES], dtype=np.float64)

    belief = np.ones((B, 4), dtype=np.float64) / 4.0
    acc_theta = np.zeros(B, dtype=np.float64)
    acc_z = np.zeros(B, dtype=np.float64)
    n_valid = max(0, T - burnin)

    for t in range(T):
        belief = np.matmul(transitions, belief[:, :, np.newaxis])[:, :, 0]
        obs = y[t]
        log_ll = -0.5 * ((obs - state_means) / sigma) ** 2
        ll = np.exp(log_ll - np.max(log_ll))
        belief = belief * ll
        belief /= np.sum(belief, axis=1, keepdims=True)

        th_hat = np.sum(belief * theta_vals, axis=1)
        zh_hat = np.sum(belief * z_vals, axis=1)
        if t >= burnin:
            acc_theta += (th_hat - theta_true[t]) ** 2
            acc_z += (zh_hat - z_true[t]) ** 2

    if n_valid == 0:
        theta_mse = np.full(B, np.nan)
        z_mse = np.full(B, np.nan)
    else:
        theta_mse = acc_theta / n_valid
        z_mse = acc_z / n_valid
    joint_mse = theta_mse + z_mse
    return theta_mse, z_mse, joint_mse


def best_single_filter(y, theta_true, z_true, eps, rho, a, b, sigma, n_tau=50, burnin=0, verbose=True):
    if np.isclose(eps, rho):
        theta_hat, z_hat = single_filter(y, eps, a, b, sigma)
        theta_mse, z_mse, joint_mse = mse_components(theta_true, z_true, theta_hat, z_hat, burnin=burnin)
        return {
            "best_tau": eps,
            "theta_mse": theta_mse,
            "z_mse": z_mse,
            "joint_mse": joint_mse,
        }

    taus = np.logspace(np.log10(eps), np.log10(rho), n_tau)
    theta_mse_arr, z_mse_arr, joint_mse_arr = batched_single_filter_mse(
        y, theta_true, z_true, taus, a, b, sigma, burnin
    )
    j = int(np.nanargmin(joint_mse_arr))
    best = {
        "best_tau": float(taus[j]),
        "theta_mse": float(theta_mse_arr[j]),
        "z_mse": float(z_mse_arr[j]),
        "joint_mse": float(joint_mse_arr[j]),
    }
    if verbose:
        print(
            f"    tau sweep batched n_tau={n_tau}: best_tau={best['best_tau']:0.5f} "
            f"best_joint={best['joint_mse']:0.4f}",
            flush=True,
        )

    return best


def classification_error_components(theta_true, z_true, theta_hat, z_hat, alpha=1.0, beta=1.0, burnin=0):
    theta_true = theta_true[burnin:]
    z_true = z_true[burnin:]
    theta_hat = theta_hat[burnin:]
    z_hat = z_hat[burnin:]

    theta_pred = np.where(theta_hat >= 0.0, 1, -1)
    z_pred = np.where(z_hat >= 0.0, 1, -1)

    theta_err = np.mean(theta_pred != theta_true)
    z_err = np.mean(z_pred != z_true)
    weighted = alpha * theta_err + beta * z_err
    return theta_err, z_err, weighted


def summarize_metric(values):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return float(values[0]), 0.0
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return mean, 1.96 * sem


def run_single_instance(T, eps, rho, a, b, sigma, n_tau, seed, alpha=1.0, beta=1.0, verbose=True):
    theta, z, y = simulate(T=T, eps=eps, rho=rho, a=a, b=b, sigma=sigma, seed=seed)
    burnin = int(1.0 / eps)

    dual_theta_hat, dual_z_hat = dual_filter(y, eps, rho, a, b, sigma)
    dual_theta_mse, dual_z_mse, dual_joint_mse = mse_components(
        theta, z, dual_theta_hat, dual_z_hat, burnin=burnin
    )
    dual_theta_cls, dual_z_cls, dual_weighted_cls = classification_error_components(
        theta, z, dual_theta_hat, dual_z_hat, alpha=alpha, beta=beta, burnin=burnin
    )

    single = best_single_filter(
        y=y,
        theta_true=theta,
        z_true=z,
        eps=eps,
        rho=rho,
        a=a,
        b=b,
        sigma=sigma,
        n_tau=n_tau,
        burnin=burnin,
        verbose=verbose,
    )
    single_theta_hat, single_z_hat = single_filter(y, single["best_tau"], a, b, sigma)
    single_theta_cls, single_z_cls, single_weighted_cls = classification_error_components(
        theta, z, single_theta_hat, single_z_hat, alpha=alpha, beta=beta, burnin=burnin
    )

    return {
        "dual_theta_mse": dual_theta_mse,
        "dual_z_mse": dual_z_mse,
        "dual_joint_mse": dual_joint_mse,
        "dual_theta_cls": dual_theta_cls,
        "dual_z_cls": dual_z_cls,
        "dual_weighted_cls": dual_weighted_cls,
        "single_theta_mse": single["theta_mse"],
        "single_z_mse": single["z_mse"],
        "single_joint_mse": single["joint_mse"],
        "single_theta_cls": single_theta_cls,
        "single_z_cls": single_z_cls,
        "single_weighted_cls": single_weighted_cls,
        "best_tau": single["best_tau"],
        "gap": single["joint_mse"] - dual_joint_mse,
        "cls_gap": single_weighted_cls - dual_weighted_cls,
    }


def aggregate_instances(rho, eps, runs):
    result = {
        "rho": rho,
        "lambda": rho / eps,
        "n_seeds": len(runs),
    }

    metric_names = [
        "dual_theta_mse",
        "dual_z_mse",
        "dual_joint_mse",
        "dual_theta_cls",
        "dual_z_cls",
        "dual_weighted_cls",
        "single_theta_mse",
        "single_z_mse",
        "single_joint_mse",
        "single_theta_cls",
        "single_z_cls",
        "single_weighted_cls",
        "best_tau",
        "gap",
        "cls_gap",
    ]

    for name in metric_names:
        mean, ci95 = summarize_metric([run[name] for run in runs])
        result[name] = mean
        result[f"{name}_ci95"] = ci95

    return result


def _pool_worker_init():
    """One BLAS/OpenMP thread per worker so many processes do not oversubscribe."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def _run_single_instance_packed(packed):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable on Windows)."""
    T, eps, rho, a, b, sigma, n_tau, seed, alpha, beta = packed
    return run_single_instance(
        T, eps, rho, a, b, sigma, n_tau, seed, alpha, beta, verbose=False
    )


def resolve_workers(requested: int) -> int:
    """0 -> min(12, CPU count); else max(1, requested)."""
    ncpu = os.cpu_count() or 1
    if requested <= 0:
        return min(12, ncpu)
    return max(1, int(requested))


def run_sweep(
    T,
    eps,
    rhos,
    a,
    b,
    sigma,
    n_tau,
    base_seed,
    alpha=1.0,
    beta=1.0,
    n_seeds=1,
    workers=12,
):
    results = []
    w = resolve_workers(workers)

    for idx, rho in enumerate(rhos, start=1):
        print(f"[rho {idx:2d}/{len(rhos):2d}] lambda={rho / eps:0.2f} rho={rho:0.5f}", flush=True)
        seed0 = base_seed + idx * n_seeds
        packed = [
            (T, eps, rho, a, b, sigma, n_tau, seed0 + seed_offset, alpha, beta)
            for seed_offset in range(n_seeds)
        ]

        if w == 1 or n_seeds == 1:
            runs = []
            for seed_offset in range(n_seeds):
                seed = seed0 + seed_offset
                print(f"  seed {seed_offset + 1:2d}/{n_seeds:2d} (rng={seed})", flush=True)
                runs.append(
                    run_single_instance(
                        T=T,
                        eps=eps,
                        rho=rho,
                        a=a,
                        b=b,
                        sigma=sigma,
                        n_tau=n_tau,
                        seed=seed,
                        alpha=alpha,
                        beta=beta,
                        verbose=True,
                    )
                )
        else:
            pw = min(w, n_seeds)
            print(f"  seeds 1..{n_seeds} parallel ({pw} worker{'s' if pw != 1 else ''})", flush=True)
            with ProcessPoolExecutor(max_workers=pw, initializer=_pool_worker_init) as ex:
                runs = list(ex.map(_run_single_instance_packed, packed))

        result = aggregate_instances(rho, eps, runs)
        results.append(result)

        print(
            "lambda={lambda_:7.2f} rho={rho:0.5f} "
            "dual_joint={dj:0.4f}±{dj_ci:0.4f} "
            "single_joint={sj:0.4f}±{sj_ci:0.4f} "
            "gap={gap:0.4f}±{gap_ci:0.4f} "
            "cls_gap={cls_gap:0.4f}±{cls_gap_ci:0.4f} "
            "tau={tau:0.5f}±{tau_ci:0.5f}".format(
                lambda_=result["lambda"],
                rho=result["rho"],
                dj=result["dual_joint_mse"],
                dj_ci=result["dual_joint_mse_ci95"],
                sj=result["single_joint_mse"],
                sj_ci=result["single_joint_mse_ci95"],
                gap=result["gap"],
                gap_ci=result["gap_ci95"],
                cls_gap=result["cls_gap"],
                cls_gap_ci=result["cls_gap_ci95"],
                tau=result["best_tau"],
                tau_ci=result["best_tau_ci95"],
            )
        )

    return results


def save_csv(results, output_path):
    header = (
        "rho,lambda,n_seeds,"
        "dual_theta_mse,dual_z_mse,dual_joint_mse,"
        "dual_theta_cls,dual_z_cls,dual_weighted_cls,"
        "single_theta_mse,single_z_mse,single_joint_mse,"
        "single_theta_cls,single_z_cls,single_weighted_cls,"
        "best_tau,gap,cls_gap,"
        "dual_theta_mse_ci95,dual_z_mse_ci95,dual_joint_mse_ci95,"
        "dual_theta_cls_ci95,dual_z_cls_ci95,dual_weighted_cls_ci95,"
        "single_theta_mse_ci95,single_z_mse_ci95,single_joint_mse_ci95,"
        "single_theta_cls_ci95,single_z_cls_ci95,single_weighted_cls_ci95,"
        "best_tau_ci95,gap_ci95,cls_gap_ci95"
    )
    lines = [header]
    for row in results:
        lines.append(
            ",".join(
                [
                    f"{row['rho']:.10f}",
                    f"{row['lambda']:.10f}",
                    f"{row['n_seeds']}",
                    f"{row['dual_theta_mse']:.10f}",
                    f"{row['dual_z_mse']:.10f}",
                    f"{row['dual_joint_mse']:.10f}",
                    f"{row['dual_theta_cls']:.10f}",
                    f"{row['dual_z_cls']:.10f}",
                    f"{row['dual_weighted_cls']:.10f}",
                    f"{row['single_theta_mse']:.10f}",
                    f"{row['single_z_mse']:.10f}",
                    f"{row['single_joint_mse']:.10f}",
                    f"{row['single_theta_cls']:.10f}",
                    f"{row['single_z_cls']:.10f}",
                    f"{row['single_weighted_cls']:.10f}",
                    f"{row['best_tau']:.10f}",
                    f"{row['gap']:.10f}",
                    f"{row['cls_gap']:.10f}",
                    f"{row['dual_theta_mse_ci95']:.10f}",
                    f"{row['dual_z_mse_ci95']:.10f}",
                    f"{row['dual_joint_mse_ci95']:.10f}",
                    f"{row['dual_theta_cls_ci95']:.10f}",
                    f"{row['dual_z_cls_ci95']:.10f}",
                    f"{row['dual_weighted_cls_ci95']:.10f}",
                    f"{row['single_theta_mse_ci95']:.10f}",
                    f"{row['single_z_mse_ci95']:.10f}",
                    f"{row['single_joint_mse_ci95']:.10f}",
                    f"{row['single_theta_cls_ci95']:.10f}",
                    f"{row['single_z_cls_ci95']:.10f}",
                    f"{row['single_weighted_cls_ci95']:.10f}",
                    f"{row['best_tau_ci95']:.10f}",
                    f"{row['gap_ci95']:.10f}",
                    f"{row['cls_gap_ci95']:.10f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def make_plot(results, output_path, title_suffix):
    lambdas = np.array([row["lambda"] for row in results], dtype=float)
    dual_joint = np.array([row["dual_joint_mse"] for row in results], dtype=float)
    single_joint = np.array([row["single_joint_mse"] for row in results], dtype=float)
    gaps = np.array([row["gap"] for row in results], dtype=float)
    dual_joint_ci = np.array([row["dual_joint_mse_ci95"] for row in results], dtype=float)
    single_joint_ci = np.array([row["single_joint_mse_ci95"] for row in results], dtype=float)
    gap_ci = np.array([row["gap_ci95"] for row in results], dtype=float)
    dual_weighted_cls = np.array([row["dual_weighted_cls"] for row in results], dtype=float)
    single_weighted_cls = np.array([row["single_weighted_cls"] for row in results], dtype=float)
    cls_gaps = np.array([row["cls_gap"] for row in results], dtype=float)
    dual_cls_ci = np.array([row["dual_weighted_cls_ci95"] for row in results], dtype=float)
    single_cls_ci = np.array([row["single_weighted_cls_ci95"] for row in results], dtype=float)
    cls_gap_ci = np.array([row["cls_gap_ci95"] for row in results], dtype=float)
    dual_theta = np.array([row["dual_theta_mse"] for row in results], dtype=float)
    dual_z = np.array([row["dual_z_mse"] for row in results], dtype=float)
    single_theta = np.array([row["single_theta_mse"] for row in results], dtype=float)
    single_z = np.array([row["single_z_mse"] for row in results], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].plot(lambdas, dual_joint, label="Dual filter", linewidth=2)
    axes[0].plot(lambdas, single_joint, label="Best single-timescale", linewidth=2)
    axes[0].fill_between(lambdas, dual_joint - dual_joint_ci, dual_joint + dual_joint_ci, alpha=0.2)
    axes[0].fill_between(lambdas, single_joint - single_joint_ci, single_joint + single_joint_ci, alpha=0.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lambda = rho / eps")
    axes[0].set_ylabel("Joint MSE")
    axes[0].set_title(f"Joint performance{title_suffix}")
    axes[0].legend()

    axes[1].plot(lambdas, dual_theta, label="Dual theta", linewidth=2)
    axes[1].plot(lambdas, single_theta, label="Single theta", linewidth=2)
    axes[1].plot(lambdas, dual_z, label="Dual z", linewidth=2, linestyle="--")
    axes[1].plot(lambdas, single_z, label="Single z", linewidth=2, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("lambda = rho / eps")
    axes[1].set_ylabel("Component MSE")
    axes[1].set_title(f"Latent-wise errors{title_suffix}")
    axes[1].legend()

    axes[2].plot(lambdas, gaps, color="black", linewidth=2)
    axes[2].fill_between(lambdas, gaps - gap_ci, gaps + gap_ci, color="black", alpha=0.15)
    axes[2].axhline(0.0, color="gray", linestyle="--")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("lambda = rho / eps")
    axes[2].set_ylabel("MSE gap (single - dual)")
    axes[2].set_title(f"Gap curve{title_suffix}")

    axes[3].plot(lambdas, dual_weighted_cls, label="Dual weighted cls", linewidth=2)
    axes[3].plot(lambdas, single_weighted_cls, label="Single weighted cls", linewidth=2)
    axes[3].plot(lambdas, cls_gaps, label="Classification gap", linewidth=2, linestyle="--", color="black")
    axes[3].fill_between(lambdas, dual_weighted_cls - dual_cls_ci, dual_weighted_cls + dual_cls_ci, alpha=0.2)
    axes[3].fill_between(lambdas, single_weighted_cls - single_cls_ci, single_weighted_cls + single_cls_ci, alpha=0.2)
    axes[3].fill_between(lambdas, cls_gaps - cls_gap_ci, cls_gaps + cls_gap_ci, color="black", alpha=0.12)
    axes[3].axhline(0.0, color="gray", linestyle="--")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("lambda = rho / eps")
    axes[3].set_ylabel("Weighted classification loss")
    axes[3].set_title(f"Classification objective{title_suffix}")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Race dual- and single-timescale filters on a two-factor latent model.")
    parser.add_argument("--T", type=int, default=2000, help="Number of timesteps.")
    parser.add_argument("--eps", type=float, default=0.01, help="Slow switching rate.")
    parser.add_argument("--rho-max", type=float, default=0.5, help="Maximum fast switching rate.")
    parser.add_argument("--n-rho", type=int, default=30, help="Number of rho values in the sweep.")
    parser.add_argument("--a", type=float, default=1.0, help="Observation weight on theta.")
    parser.add_argument("--b", type=float, default=1.0, help="Observation weight on z.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Observation noise standard deviation.")
    parser.add_argument("--n-tau", type=int, default=50, help="Number of tau values for the single-timescale sweep.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for the rho sweep.")
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds to average per rho.")
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Parallel processes for the seed loop (1 = sequential with per-seed logs). "
        "0 = min(12, CPU count). Capped at n-seeds per rho.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight on theta classification loss.")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight on z classification loss.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs") / "filter_race",
        help="Prefix for CSV and PNG outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rhos = np.logspace(np.log10(args.eps), np.log10(args.rho_max), args.n_rho)
    results = run_sweep(
        T=args.T,
        eps=args.eps,
        rhos=rhos,
        a=args.a,
        b=args.b,
        sigma=args.sigma,
        n_tau=args.n_tau,
        base_seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        n_seeds=args.n_seeds,
        workers=args.workers,
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    plot_path = args.output_prefix.with_suffix(".png")
    save_csv(results, csv_path)
    make_plot(results, plot_path, title_suffix=f" (sigma={args.sigma}, a={args.a}, b={args.b})")

    best_gap = max(results, key=lambda row: row["gap"])
    mean_gap = float(np.mean([row["gap"] for row in results]))
    mean_cls_gap = float(np.mean([row["cls_gap"] for row in results]))
    print()
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print(
        "Null check: lambda={lambda_:0.2f} gap={gap:0.4f}±{gap_ci:0.4f}, cls_gap={cls_gap:0.4f}±{cls_gap_ci:0.4f}".format(
            lambda_=results[0]["lambda"],
            gap=results[0]["gap"],
            gap_ci=results[0]["gap_ci95"],
            cls_gap=results[0]["cls_gap"],
            cls_gap_ci=results[0]["cls_gap_ci95"],
        )
    )
    print(
        "Summary: mean gap={mean_gap:0.4f}, mean cls gap={mean_cls_gap:0.4f}, "
        "max gap={max_gap:0.4f}±{max_gap_ci:0.4f} at lambda={lambda_:0.2f}, best_tau={tau:0.5f}".format(
            mean_gap=mean_gap,
            mean_cls_gap=mean_cls_gap,
            max_gap=best_gap["gap"],
            max_gap_ci=best_gap["gap_ci95"],
            lambda_=best_gap["lambda"],
            tau=best_gap["best_tau"],
        )
    )


if __name__ == "__main__":
    main()
