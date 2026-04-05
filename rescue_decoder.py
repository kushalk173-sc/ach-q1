"""M3: Rescue decoder experiment for Q3.

Tests whether downstream recurrence can recover information lost
by the single-timescale encoder. Three decoders are evaluated:

  - **OnePole**: admissible single-timescale rescue model (matched `hidden_dim`).
  - **GRU**: strong unconstrained rescue baseline (same `hidden_dim`; parameter
    count differs — see exported counts / JSON metadata).
  - **Clockwork**: adversarial multiscale stress test (same `hidden_dim`).

**Diagnostic (not part of the admissible Q3 protocol):** a GRU trained on raw
`y_t` checks that poor rescue from `c_t` is due to the belief bottleneck, not
weak optimization. Compare GRU(beliefs) vs GRU(y) only as a sanity check.

HARD PROTOCOL RULE: rescue decoders in the main test receive only belief states
`c_t` from the single-timescale filter. They never see `y_t` directly. Violation
of this rule makes the Q3 test meaningless.

Speed: NumPy/Numba filtering is CPU-bound; PyTorch training speeds up with
`--device cuda` when a GPU is available. Use `--epoch-log-interval 0` to reduce
I/O during long runs.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from filter_race_experiment import (
    simulate,
    dual_filter,
    single_filter_with_beliefs,
    best_single_filter,
    mse_components,
    classification_error_components,
    summarize_metric,
    STATES,  # noqa: F401
)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def resolve_device(name: str) -> torch.device:
    """auto -> CUDA if available, else CPU."""
    n = (name or "auto").strip().lower()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if n == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown --device {name!r}; use auto, cpu, or cuda.")


class OnePoleDecoder(nn.Module):
    """Minimal admissible single-timescale decoder.

    Recurrent dynamics are a fixed scalar exponential decay.
    All eigenvalues of the recurrent map equal exp(-1/tau) — one pole.
    This is the tightest possible single-timescale rival.
    """

    def __init__(self, input_dim=4, hidden_dim=64, tau=1.0):
        super().__init__()
        self.alpha = float(np.exp(-1.0 / max(tau, 1e-6)))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_theta = nn.Linear(hidden_dim, 1)
        self.output_z = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, beliefs):
        # beliefs: (T, input_dim)
        T = beliefs.shape[0]
        h = torch.zeros(self.hidden_dim, device=beliefs.device)
        theta_out, z_out = [], []
        for t in range(T):
            x = torch.tanh(self.input_proj(beliefs[t]))
            h = self.alpha * h + (1.0 - self.alpha) * x
            theta_out.append(self.output_theta(h))
            z_out.append(self.output_z(h))
        return torch.stack(theta_out).squeeze(-1), torch.stack(z_out).squeeze(-1)


class GRUDecoder(nn.Module):
    """Unconstrained gated recurrence — the standard strong baseline.

    GRU gates implement heterogeneous effective timescales internally.
    If this fails, it is a strong result. If it succeeds, the rival
    class definition needs tightening because GRU is multiscale.
    """

    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.output_theta = nn.Linear(hidden_dim, 1)
        self.output_z = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        # seq: (T, input_dim) -> (1, T, input_dim) for batch_first GRU
        out, _ = self.gru(seq.unsqueeze(0))
        out = out.squeeze(0)  # (T, hidden)
        return self.output_theta(out).squeeze(-1), self.output_z(out).squeeze(-1)


class ClockworkDecoder(nn.Module):
    """Explicitly multi-clock decoder — the adversarial baseline.

    Neuron groups fire at different clock periods. This is a strong
    rescue attempt: it is explicitly designed to separate multiple
    timescales. If this fails consistently at high lambda under matched
    protocol settings, the loophole is substantially narrowed empirically
    in this model family.

    Clock periods should span the eps-to-rho range; configure via CLI
    (default 1, 8, 64). Use longer slow clocks (e.g. 128) to track the
    slow latent when eps is small.
    """

    def __init__(self, input_dim=4, hidden_dim=64, clock_periods=(1, 8, 64)):
        super().__init__()
        self.clock_periods = clock_periods
        n_groups = len(clock_periods)
        self.group_size = hidden_dim // n_groups
        actual_hidden = self.group_size * n_groups
        self.input_proj = nn.Linear(input_dim, actual_hidden)
        self.recurrent = nn.Linear(actual_hidden, actual_hidden, bias=False)
        self.output_theta = nn.Linear(actual_hidden, 1)
        self.output_z = nn.Linear(actual_hidden, 1)
        self.actual_hidden = actual_hidden

    def forward(self, beliefs):
        T = beliefs.shape[0]
        h = torch.zeros(self.actual_hidden, device=beliefs.device)
        theta_out, z_out = [], []

        inp_all = self.input_proj(beliefs)  # (T, hidden); nonlinearity only in-loop

        for t in range(T):
            rh = self.recurrent(h)
            new_h = h.clone()
            for g, period in enumerate(self.clock_periods):
                if t % period == 0:
                    s = g * self.group_size
                    e = s + self.group_size
                    new_h[s:e] = torch.tanh(inp_all[t, s:e] + rh[s:e])
            h = new_h
            theta_out.append(self.output_theta(h))
            z_out.append(self.output_z(h))

        return torch.stack(theta_out).squeeze(-1), torch.stack(z_out).squeeze(-1)


def train_decoder(
    decoder,
    inputs,
    theta_true,
    z_true,
    n_epochs=600,
    lr=1e-3,
    burnin=0,
    device=None,
    log_interval=100,
):
    """Train decoder on `inputs` only (beliefs c_t, or diagnostic y)."""
    if device is None:
        device = torch.device("cpu")
    decoder = decoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_t = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    th_t = torch.as_tensor(theta_true, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z_true, dtype=torch.float32, device=device)

    decoder.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        th_pred, z_pred = decoder(x_t)
        loss = loss_fn(th_pred[burnin:], th_t[burnin:]) + loss_fn(
            z_pred[burnin:], z_t[burnin:]
        )
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        if log_interval and (epoch + 1) % log_interval == 0:
            print(
                f"    epoch {epoch + 1}/{n_epochs} loss={loss.item():.5f}",
                flush=True,
            )

    decoder.eval()
    return decoder


def eval_decoder(decoder, inputs, theta_true, z_true, burnin=0, device=None):
    if device is None:
        device = next(decoder.parameters()).device
    with torch.no_grad():
        x_t = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        th_pred, z_pred = decoder(x_t)
    th_hat = th_pred.detach().cpu().numpy()
    z_hat = z_pred.detach().cpu().numpy()
    _, _, joint_mse = mse_components(theta_true, z_true, th_hat, z_hat, burnin=burnin)
    theta_cls, z_cls, weighted_cls = classification_error_components(
        theta_true, z_true, th_hat, z_hat, burnin=burnin
    )
    return {
        "joint_mse": joint_mse,
        "weighted_cls": weighted_cls,
        "theta_cls": theta_cls,
        "z_cls": z_cls,
    }


def filter_classification_metrics(theta_true, z_true, theta_hat, z_hat, burnin=0):
    te, ze, w = classification_error_components(
        theta_true, z_true, theta_hat, z_hat, burnin=burnin
    )
    return {"weighted_cls": w, "theta_cls": te, "z_cls": ze}


def run_rescue_sweep(
    T,
    eps,
    rhos,
    a,
    b,
    sigma,
    n_tau,
    n_seeds,
    base_seed,
    n_epochs=600,
    lr=1e-3,
    clock_periods=(1, 8, 64),
    hidden_dim=64,
    run_diagnostic_y_gru=True,
    device=None,
    log_interval=100,
    torch_compile=False,
):
    """
    For each rho value:
      1. Find best tau* for single filter (from M2)
      2. Extract belief states c_t (the code — never y for protocol decoders)
      3. Train rescue decoders on training sequence
      4. Evaluate all systems on held-out sequence
      5. Compute rescue fraction (MSE): how much of the gap does each decoder close?

    Train/eval split uses independent sequences. No temporal leakage.
    """
    if device is None:
        device = torch.device("cpu")

    T_train = T // 2
    T_eval = T - T_train
    burnin = int(1.0 / eps)
    if burnin >= T_train or burnin >= T_eval:
        raise ValueError(
            f"burnin={burnin} must be < T_train={T_train} and < T_eval={T_eval}. "
            "Increase T or decrease eps."
        )

    results = []

    for idx, rho in enumerate(rhos):
        lam = rho / eps
        print(f"\n[rho {idx+1}/{len(rhos)}] lambda={lam:.2f}", flush=True)

        seed_rows = []
        for s in range(n_seeds):
            seed_tr = base_seed + idx * n_seeds * 2 + s
            seed_ev = seed_tr + 100000

            # Generate train and eval sequences independently
            theta_tr, z_tr, y_tr = simulate(T_train, eps, rho, a, b, sigma, seed_tr)
            theta_ev, z_ev, y_ev = simulate(T_eval, eps, rho, a, b, sigma, seed_ev)

            # Find best tau* using training sequence
            best = best_single_filter(
                y_tr,
                theta_tr,
                z_tr,
                eps,
                rho,
                a,
                b,
                sigma,
                n_tau=n_tau,
                burnin=burnin,
            )
            tau_star = best["best_tau"]

            # Extract belief states — ONLY thing protocol decoders see
            _, _, beliefs_tr = single_filter_with_beliefs(y_tr, tau_star, a, b, sigma)
            _, _, beliefs_ev = single_filter_with_beliefs(y_ev, tau_star, a, b, sigma)

            # Dual filter on eval — the ceiling
            dual_th, dual_z = dual_filter(y_ev, eps, rho, a, b, sigma)
            dual_mse = mse_components(theta_ev, z_ev, dual_th, dual_z, burnin=burnin)[2]
            dual_cls = filter_classification_metrics(
                theta_ev, z_ev, dual_th, dual_z, burnin=burnin
            )

            # Single filter alone on eval — the floor
            single_th, single_z, _ = single_filter_with_beliefs(
                y_ev, tau_star, a, b, sigma
            )
            single_mse = mse_components(
                theta_ev, z_ev, single_th, single_z, burnin=burnin
            )[2]
            single_cls = filter_classification_metrics(
                theta_ev, z_ev, single_th, single_z, burnin=burnin
            )

            base_gap = single_mse - dual_mse

            # Torch seed offsets — stable across Python versions
            def set_torch_seed(name: str) -> None:
                off = {"one_pole": 11, "gru": 13, "clockwork": 17, "diag_y_gru": 19}[
                    name
                ]
                torch.manual_seed((seed_tr + off * 1_000_003 + s * 97) % (2**31))

            # Train and evaluate each rescue decoder (protocol: c_t only)
            rescue_results = {}
            for name, factory in [
                ("one_pole", lambda: OnePoleDecoder(hidden_dim=hidden_dim, tau=tau_star)),
                ("gru", lambda: GRUDecoder(hidden_dim=hidden_dim)),
                (
                    "clockwork",
                    lambda: ClockworkDecoder(
                        hidden_dim=hidden_dim, clock_periods=clock_periods
                    ),
                ),
            ]:
                print(f"  Training {name}...", flush=True)
                set_torch_seed(name)
                decoder = factory()
                decoder = decoder.to(device)
                if torch_compile and hasattr(torch, "compile"):
                    decoder = torch.compile(decoder)  # type: ignore[assignment]
                train_decoder(
                    decoder,
                    beliefs_tr,
                    theta_tr,
                    z_tr,
                    n_epochs=n_epochs,
                    lr=lr,
                    burnin=burnin,
                    device=device,
                    log_interval=log_interval,
                )
                metrics = eval_decoder(
                    decoder,
                    beliefs_ev,
                    theta_ev,
                    z_ev,
                    burnin=burnin,
                    device=device,
                )
                mse = metrics["joint_mse"]
                frac = (single_mse - mse) / base_gap if base_gap > 1e-8 else 0.0
                rescue_results[name] = {**metrics, "fraction": frac}
                print(
                    f"    mse={mse:.4f} w_cls={metrics['weighted_cls']:.4f} "
                    f"fraction={frac:.3f}",
                    flush=True,
                )

            diag_y = {}
            if run_diagnostic_y_gru:
                print("  Training diag_y_gru (y -> GRU, not protocol)...", flush=True)
                set_torch_seed("diag_y_gru")
                y_seq_tr = y_tr.reshape(-1, 1).astype(np.float32)
                y_seq_ev = y_ev.reshape(-1, 1).astype(np.float32)
                diag_dec = GRUDecoder(input_dim=1, hidden_dim=hidden_dim)
                diag_dec = diag_dec.to(device)
                if torch_compile and hasattr(torch, "compile"):
                    diag_dec = torch.compile(diag_dec)  # type: ignore[assignment]
                train_decoder(
                    diag_dec,
                    y_seq_tr,
                    theta_tr,
                    z_tr,
                    n_epochs=n_epochs,
                    lr=lr,
                    burnin=burnin,
                    device=device,
                    log_interval=log_interval,
                )
                diag_y = eval_decoder(
                    diag_dec,
                    y_seq_ev,
                    theta_ev,
                    z_ev,
                    burnin=burnin,
                    device=device,
                )
                print(
                    f"    [diagnostic] y->GRU mse={diag_y['joint_mse']:.4f} "
                    f"w_cls={diag_y['weighted_cls']:.4f}",
                    flush=True,
                )

            row_seed = {
                "dual_mse": dual_mse,
                "dual_weighted_cls": dual_cls["weighted_cls"],
                "dual_theta_cls": dual_cls["theta_cls"],
                "dual_z_cls": dual_cls["z_cls"],
                "single_mse": single_mse,
                "single_weighted_cls": single_cls["weighted_cls"],
                "single_theta_cls": single_cls["theta_cls"],
                "single_z_cls": single_cls["z_cls"],
                "base_gap": base_gap,
                "tau_star": tau_star,
            }
            for name in ["one_pole", "gru", "clockwork"]:
                m = rescue_results[name]
                row_seed[f"{name}_mse"] = m["joint_mse"]
                row_seed[f"{name}_weighted_cls"] = m["weighted_cls"]
                row_seed[f"{name}_theta_cls"] = m["theta_cls"]
                row_seed[f"{name}_z_cls"] = m["z_cls"]
                row_seed[f"{name}_fraction"] = m["fraction"]

            if run_diagnostic_y_gru:
                row_seed["diag_y_gru_mse"] = diag_y["joint_mse"]
                row_seed["diag_y_gru_weighted_cls"] = diag_y["weighted_cls"]
                row_seed["diag_y_gru_theta_cls"] = diag_y["theta_cls"]
                row_seed["diag_y_gru_z_cls"] = diag_y["z_cls"]

            seed_rows.append(row_seed)

        # Aggregate across seeds
        row = {"lambda": lam, "rho": rho}
        for key in seed_rows[0]:
            vals = [r[key] for r in seed_rows]
            mean, ci = summarize_metric(vals)
            row[key] = mean
            row[f"{key}_ci95"] = ci
        results.append(row)

        print(
            f"  dual={row['dual_mse']:.4f} single={row['single_mse']:.4f} "
            f"one_pole_frac={row['one_pole_fraction']:.3f} "
            f"gru_frac={row['gru_fraction']:.3f} "
            f"clockwork_frac={row['clockwork_fraction']:.3f}"
        )

    return results


def _ci_band(y, ci):
    y = np.asarray(y, dtype=float)
    ci = np.asarray(ci, dtype=float)
    return y - ci, y + ci


def make_rescue_plot(results, output_path):
    lambdas = np.array([r["lambda"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: MSE + 95% CI bands
    ax = axes[0, 0]
    series_mse = [
        ("dual_mse", "Dual filter (ceiling)", "-", None),
        ("single_mse", "Single filter (floor)", "-", None),
        ("one_pole_mse", "+ OnePole (protocol)", "--", "C0"),
        ("gru_mse", "+ GRU (protocol)", "--", "C1"),
        ("clockwork_mse", "+ Clockwork (protocol)", "--", "C2"),
    ]
    for key, label, ls, color in series_mse:
        y = np.array([r[key] for r in results])
        ci = np.array([r[f"{key}_ci95"] for r in results])
        kw = {"linewidth": 2, "linestyle": ls, "label": label}
        if color is not None:
            kw["color"] = color
        ax.plot(lambdas, y, **kw)
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color=color or (0.5, 0.5, 0.5))

    if results and "diag_y_gru_mse" in results[0]:
        y = np.array([r["diag_y_gru_mse"] for r in results])
        ci = np.array([r["diag_y_gru_mse_ci95"] for r in results])
        ax.plot(
            lambdas,
            y,
            linewidth=2,
            linestyle=":",
            color="C3",
            label="GRU | y (diagnostic only)",
        )
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color="C3")

    ax.set_xscale("log")
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel("Joint MSE (eval set)")
    ax.set_title("Rescue decoder performance (MSE + 95% CI)")
    ax.legend(fontsize=8, loc="best")

    # Top-right: rescue fraction — MSE-based Q3 statistic
    ax = axes[0, 1]
    for name, color in [("one_pole", "C0"), ("gru", "C1"), ("clockwork", "C2")]:
        fracs = np.array([r[f"{name}_fraction"] for r in results])
        ci = np.array([r[f"{name}_fraction_ci95"] for r in results])
        ax.plot(lambdas, fracs, label=name, linewidth=2, color=color)
        lo, hi = _ci_band(fracs, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.2, color=color)
    ax.axhline(0.0, color="gray", linestyle="--", label="no rescue")
    ax.axhline(1.0, color="gray", linestyle=":", label="full rescue")
    ax.set_xscale("log")
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel("Rescue fraction (MSE, 0–1)")
    ax.set_title("Q3: MSE rescue fraction")
    ax.legend(fontsize=8)

    # Bottom-left: weighted classification error + CI
    ax = axes[1, 0]
    series_cls = [
        ("dual_weighted_cls", "Dual filter (ceiling)", "-", None),
        ("single_weighted_cls", "Single filter (floor)", "-", None),
        ("one_pole_weighted_cls", "+ OnePole (protocol)", "--", "C0"),
        ("gru_weighted_cls", "+ GRU (protocol)", "--", "C1"),
        ("clockwork_weighted_cls", "+ Clockwork (protocol)", "--", "C2"),
    ]
    for key, label, ls, color in series_cls:
        y = np.array([r[key] for r in results])
        ci = np.array([r[f"{key}_ci95"] for r in results])
        kw = {"linewidth": 2, "linestyle": ls, "label": label}
        if color is not None:
            kw["color"] = color
        ax.plot(lambdas, y, **kw)
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color=color or (0.5, 0.5, 0.5))

    if results and "diag_y_gru_weighted_cls" in results[0]:
        y = np.array([r["diag_y_gru_weighted_cls"] for r in results])
        ci = np.array([r["diag_y_gru_weighted_cls_ci95"] for r in results])
        ax.plot(
            lambdas,
            y,
            linewidth=2,
            linestyle=":",
            color="C3",
            label="GRU | y (diagnostic only)",
        )
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color="C3")

    ax.set_xscale("log")
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel(r"Weighted classification error ($\alpha\epsilon_\theta+\beta\epsilon_z$)")
    ax.set_title("Weighted classification error (eval + 95% CI)")
    ax.legend(fontsize=8, loc="best")

    # Bottom-right: protocol GRU vs diagnostic y->GRU (MSE)
    ax = axes[1, 1]
    gru_m = np.array([r["gru_mse"] for r in results])
    if results and "diag_y_gru_mse" in results[0]:
        dy = np.array([r["diag_y_gru_mse"] for r in results])
        ax.plot(lambdas, gru_m, label="GRU | beliefs (protocol)", linewidth=2, color="C1")
        ax.plot(
            lambdas,
            dy,
            label="GRU | y (diagnostic)",
            linewidth=2,
            linestyle="--",
            color="C3",
        )
        ax.fill_between(
            lambdas,
            *_ci_band(gru_m, np.array([r["gru_mse_ci95"] for r in results])),
            alpha=0.15,
            color="C1",
        )
        ax.fill_between(
            lambdas,
            *_ci_band(dy, np.array([r["diag_y_gru_mse_ci95"] for r in results])),
            alpha=0.15,
            color="C3",
        )
    else:
        ax.plot(lambdas, gru_m, label="GRU | beliefs (protocol)", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel("Joint MSE (eval)")
    ax.set_title("Sanity: belief bottleneck vs raw y (diagnostic)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved to {output_path}")


def to_jsonable(obj):
    """Convert numpy scalars and Paths for json.dumps."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_rescue_csv(results, output_path):
    if not results:
        output_path.write_text("", encoding="ascii")
        print(f"Saved empty CSV to {output_path}")
        return

    keys = sorted(results[0].keys())
    lines = [",".join(keys)]
    for row in results:
        parts = []
        for k in keys:
            v = row.get(k, "")
            if isinstance(v, (float, np.floating)):
                parts.append(f"{float(v):.8f}")
            elif isinstance(v, (int, np.integer)):
                parts.append(str(int(v)))
            else:
                parts.append(str(v))
        lines.append(",".join(parts))
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Saved CSV to {output_path}")


def save_rescue_json(results, meta, output_path):
    payload = {"meta": to_jsonable(meta), "results": to_jsonable(results)}
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved JSON to {output_path}")


def build_param_counts(hidden_dim, clock_periods):
    return {
        "one_pole": count_parameters(
            OnePoleDecoder(hidden_dim=hidden_dim, tau=1.0)
        ),
        "gru_beliefs": count_parameters(GRUDecoder(hidden_dim=hidden_dim)),
        "clockwork": count_parameters(
            ClockworkDecoder(hidden_dim=hidden_dim, clock_periods=clock_periods)
        ),
        "diag_gru_y": count_parameters(GRUDecoder(input_dim=1, hidden_dim=hidden_dim)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=6000)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--rho-max", type=float, default=0.5)
    p.add_argument("--n-rho", type=int, default=15)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--n-tau", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument(
        "--n-epochs",
        type=int,
        default=600,
        help="Training epochs per decoder (smoke tests often need fewer; full runs often need 600+).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for all decoders. Try 3e-3 if loss is still falling at epoch end.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-prefix", type=Path, default=Path("outputs") / "rescue"
    )
    p.add_argument(
        "--clock-periods",
        type=int,
        nargs="+",
        default=[1, 8, 64],
        help="Clock periods for ClockworkDecoder (one per group).",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden width for all decoders (budget-matched width).",
    )
    p.add_argument(
        "--no-diagnostic-y",
        action="store_true",
        help="Skip raw y -> GRU positive control (not part of Q3 protocol).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="PyTorch device for decoder training. auto = CUDA if available. Filtering stays NumPy/CPU.",
    )
    p.add_argument(
        "--epoch-log-interval",
        type=int,
        default=100,
        help="Print training loss every N epochs; 0 = silent (faster logging).",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Wrap decoders with torch.compile (PyTorch 2+). First epoch may be slower; often helps GRU on GPU.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rhos = np.logspace(np.log10(args.eps), np.log10(args.rho_max), args.n_rho)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"PyTorch device: {device}", flush=True)

    clock_periods = tuple(args.clock_periods)
    hidden_dim = args.hidden_dim
    param_counts = build_param_counts(hidden_dim, clock_periods)

    results = run_rescue_sweep(
        T=args.T,
        eps=args.eps,
        rhos=rhos,
        a=args.a,
        b=args.b,
        sigma=args.sigma,
        n_tau=args.n_tau,
        n_seeds=args.n_seeds,
        base_seed=args.seed,
        n_epochs=args.n_epochs,
        lr=args.lr,
        clock_periods=clock_periods,
        hidden_dim=hidden_dim,
        run_diagnostic_y_gru=not args.no_diagnostic_y,
        device=device,
        log_interval=args.epoch_log_interval,
        torch_compile=args.torch_compile,
    )

    meta = {
        "torch_device": str(device),
        "param_counts": param_counts,
        "hidden_dim": hidden_dim,
        "clock_periods": list(clock_periods),
        "gru_num_layers": 2,
        "protocol_note": (
            "OnePole = admissible single-timescale rescue; GRU/Clockwork = "
            "stress tests with same hidden_dim but different parameter totals "
            "(see param_counts)."
        ),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }

    save_rescue_json(results, meta, args.output_prefix.with_suffix(".json"))
    save_rescue_csv(results, args.output_prefix.with_suffix(".csv"))
    make_rescue_plot(results, args.output_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
