"""M3: Rescue decoder experiment for Q3.

Tests whether downstream recurrence can recover information lost
by the single-timescale encoder. Main decoders:

  - **OnePole**: admissible single-timescale rescue (same `hidden_dim` width).
  - **GRU** / **Clockwork**: **stress tests**, not equal-budget admissible
    controls — same `hidden_dim` does **not** mean matched parameter count;
    gating and multi-clock modules add capacity and multiscale inductive bias.
  - **gru_matched** (optional): GRU with **smaller** hidden size chosen so total
    parameters are close to OnePole — addresses “GRU only wins by width.”

**Diagnostics (not Q3 protocol):** `y -> GRU` (raw observations); optional
`dual beliefs -> GRU` (ceiling beliefs) to check decoder capacity vs bottleneck.

HARD PROTOCOL RULE: main rescue decoders see only **single-filter** beliefs
`c_t`. They never see `y_t`. Violation voids the Q3 test.

Speed: use `--device cuda` when available. On CPU, OnePole uses a vectorized
input projection; set `--cpu-threads` (or `OMP_NUM_THREADS`) so PyTorch can use
multiple cores for GRU/MKL. `--torch-compile` can speed CPU too (PyTorch 2+).
Use `--epoch-log-interval 0` to reduce logging overhead.

**Checkpoints:** by default writes `<output-prefix>.checkpoint.json` after each seed
(completed rho rows + in-progress seeds). If the run stops, re-launch with the same
arguments plus `--resume`. Finished runs delete the checkpoint. Use `--fresh-start`
to ignore an old checkpoint, or `--no-checkpoint` to disable.

**Progress:** `python rescue_decoder.py --report-checkpoint YOUR.checkpoint.json`
prints how many rho values are done and optional `--report-plot partial.png` for curves.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from filter_race_experiment import (
    simulate,
    dual_filter,
    dual_filter_with_beliefs,
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


def configure_pytorch_cpu_threads(num_threads: int) -> int:
    """Cap BLAS/MKL intra-op threads for CPU training (GRU, etc.)."""
    ncpu = os.cpu_count() or 4
    if num_threads <= 0:
        env = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
        num_threads = env if env > 0 else ncpu
    num_threads = max(1, min(int(num_threads), ncpu))
    torch.set_num_threads(num_threads)
    interop = max(1, min(4, num_threads // 4 or 1))
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    return num_threads


def gru_hidden_dim_matched_to_onepole(
    input_dim: int, one_pole_hidden_dim: int, num_layers: int = 2, max_h: int = 512
):
    """Pick GRU hidden size so parameter count is closest to OnePoleDecoder."""
    target = count_parameters(
        OnePoleDecoder(input_dim=input_dim, hidden_dim=one_pole_hidden_dim, tau=1.0)
    )
    best_h, best_n, best_diff = 8, 0, float("inf")
    for h in range(4, max_h + 1):
        n = count_parameters(
            GRUDecoder(input_dim=input_dim, hidden_dim=h, num_layers=num_layers)
        )
        d = abs(n - target)
        if d < best_diff:
            best_diff, best_h, best_n = d, h, n
    return best_h, target, best_n


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
        # beliefs: (T, input_dim) — vectorized input proj; EMA recurrence in-loop
        x = torch.tanh(self.input_proj(beliefs))
        T, _ = x.shape
        device, dtype = x.device, x.dtype
        h = torch.zeros(self.hidden_dim, device=device, dtype=dtype)
        alpha = torch.as_tensor(self.alpha, device=device, dtype=dtype)
        one_m_a = 1.0 - alpha
        theta_out = torch.empty(T, device=device, dtype=dtype)
        z_out = torch.empty(T, device=device, dtype=dtype)
        for t in range(T):
            h = alpha * h + one_m_a * x[t]
            theta_out[t] = self.output_theta(h).squeeze(-1)
            z_out[t] = self.output_z(h).squeeze(-1)
        return theta_out, z_out


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
        if hidden_dim % n_groups != 0:
            warnings.warn(
                f"ClockworkDecoder: hidden_dim={hidden_dim} is not divisible by "
                f"len(clock_periods)={n_groups}; using effective width "
                f"{(hidden_dim // n_groups) * n_groups} (group_size={hidden_dim // n_groups}).",
                stacklevel=2,
            )
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
    val_inputs=None,
    val_theta=None,
    val_z=None,
    val_burnin=None,
    early_stopping_patience=0,
):
    """Train decoder on `inputs` only (beliefs c_t, or diagnostic y).

    If ``val_inputs`` is set and ``early_stopping_patience > 0``, train only on
    ``inputs`` and use the validation sequence for checkpoint selection (lowest
    joint MSE on the val segment). Otherwise trains on the full ``inputs``.
    """
    if device is None:
        device = torch.device("cpu")
    decoder = decoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_t = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    th_t = torch.as_tensor(theta_true, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z_true, dtype=torch.float32, device=device)

    use_es = (
        val_inputs is not None
        and val_theta is not None
        and val_z is not None
        and early_stopping_patience > 0
    )
    if use_es:
        xv = torch.as_tensor(val_inputs, dtype=torch.float32, device=device)
        thv = torch.as_tensor(val_theta, dtype=torch.float32, device=device)
        zv = torch.as_tensor(val_z, dtype=torch.float32, device=device)
        vb = val_burnin if val_burnin is not None else burnin
        vb = min(vb, max(0, xv.shape[0] - 1))

    best_state = None
    best_val = float("inf")
    stall = 0

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

        if use_es:
            with torch.no_grad():
                th_p, z_p = decoder(xv)
                vloss = loss_fn(th_p[vb:], thv[vb:]) + loss_fn(z_p[vb:], zv[vb:])
                v = float(vloss.item())
            if v < best_val - 1e-9:
                best_val = v
                best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= early_stopping_patience:
                    break

        if log_interval and (epoch + 1) % log_interval == 0:
            msg = f"    epoch {epoch + 1}/{n_epochs} loss={loss.item():.5f}"
            if use_es:
                msg += f" val={v:.5f}"
            print(msg, flush=True)

    if use_es and best_state is not None:
        decoder.load_state_dict(best_state)

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
    theta_mse, z_mse, joint_mse = mse_components(
        theta_true, z_true, th_hat, z_hat, burnin=burnin
    )
    theta_cls, z_cls, weighted_cls = classification_error_components(
        theta_true, z_true, th_hat, z_hat, burnin=burnin
    )
    return {
        "joint_mse": joint_mse,
        "theta_mse": theta_mse,
        "z_mse": z_mse,
        "weighted_cls": weighted_cls,
        "theta_cls": theta_cls,
        "z_cls": z_cls,
    }


def filter_classification_metrics(theta_true, z_true, theta_hat, z_hat, burnin=0):
    te, ze, w = classification_error_components(
        theta_true, z_true, theta_hat, z_hat, burnin=burnin
    )
    return {"weighted_cls": w, "theta_cls": te, "z_cls": ze}


def rescue_fractions(
    metrics,
    single_tm,
    single_zm,
    single_jm,
    dual_tm,
    dual_zm,
    dual_jm,
):
    """Joint and per-latent rescue fractions (MSE; 0 = no rescue, 1 = closes gap)."""
    gap = single_jm - dual_jm
    gap_t = single_tm - dual_tm
    gap_z = single_zm - dual_zm
    mj = metrics["joint_mse"]
    mt = metrics["theta_mse"]
    mz = metrics["z_mse"]
    frac = (single_jm - mj) / gap if gap > 1e-8 else 0.0
    frac_t = (single_tm - mt) / gap_t if gap_t > 1e-8 else 0.0
    frac_z = (single_zm - mz) / gap_z if gap_z > 1e-8 else 0.0
    return frac, frac_t, frac_z


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
    run_diagnostic_dual_gru=True,
    run_gru_matched=True,
    val_frac=0.0,
    early_stopping_patience=0,
    device=None,
    log_interval=100,
    torch_compile=False,
    checkpoint_path=None,
    resume=False,
    config_signature=None,
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

    hm, one_pole_param_target, gru_matched_param_count = gru_hidden_dim_matched_to_onepole(
        4, hidden_dim
    )

    results = []
    resume_at_rho = None
    resume_seed_rows = None

    if checkpoint_path is not None and config_signature is not None and resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"--resume requested but checkpoint not found: {checkpoint_path}"
            )
        completed, pr, ps = load_rescue_checkpoint(checkpoint_path, config_signature)
        results = completed
        resume_at_rho = pr
        resume_seed_rows = ps
        if results:
            print(
                f"Resume: loaded {len(results)} completed rho row(s) from checkpoint; "
                f"next index {len(results)}"
                + (f", partial rho seed progress restored" if resume_at_rho is not None else ""),
                flush=True,
            )

    for idx, rho in enumerate(rhos):
        lam = rho / eps
        if idx < len(results):
            continue

        print(f"\n[rho {idx+1}/{len(rhos)}] lambda={lam:.2f}", flush=True)

        seed_rows = []
        start_s = 0
        if resume_at_rho is not None and idx == resume_at_rho and resume_seed_rows is not None:
            seed_rows = list(resume_seed_rows)
            start_s = len(seed_rows)
            resume_at_rho = None
            resume_seed_rows = None
            print(
                f"  Resuming {len(seed_rows)} seed(s) already done for this rho; "
                f"continuing from seed {start_s + 1}/{n_seeds}",
                flush=True,
            )

        for s in range(start_s, n_seeds):
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
            dual_tm, dual_zm, dual_mse = mse_components(
                theta_ev, z_ev, dual_th, dual_z, burnin=burnin
            )
            dual_cls = filter_classification_metrics(
                theta_ev, z_ev, dual_th, dual_z, burnin=burnin
            )

            # Single filter alone on eval — the floor
            single_th, single_z, _ = single_filter_with_beliefs(
                y_ev, tau_star, a, b, sigma
            )
            single_tm, single_zm, single_mse = mse_components(
                theta_ev, z_ev, single_th, single_z, burnin=burnin
            )
            single_cls = filter_classification_metrics(
                theta_ev, z_ev, single_th, single_z, burnin=burnin
            )

            base_gap = single_mse - dual_mse
            gap_theta = single_tm - dual_tm
            gap_z = single_zm - dual_zm

            # Optional train / val split on the training sequence for early stopping
            if val_frac > 0 and early_stopping_patience > 0:
                T_fit = int(T_train * (1.0 - val_frac))
                if T_fit <= burnin or (T_train - T_fit) < max(2, burnin // 2):
                    raise ValueError(
                        f"val_frac={val_frac} leaves too few steps after burnin="
                        f"{burnin}; reduce val_frac or increase T."
                    )
                beliefs_fit = beliefs_tr[:T_fit]
                theta_fit = theta_tr[:T_fit]
                z_fit = z_tr[:T_fit]
                beliefs_val = beliefs_tr[T_fit:]
                theta_val = theta_tr[T_fit:]
                z_val = z_tr[T_fit:]
                val_burnin = min(burnin, max(0, len(beliefs_val) - 1))
                train_kw = {
                    "val_inputs": beliefs_val,
                    "val_theta": theta_val,
                    "val_z": z_val,
                    "val_burnin": val_burnin,
                    "early_stopping_patience": early_stopping_patience,
                }
            else:
                beliefs_fit = beliefs_tr
                theta_fit = theta_tr
                z_fit = z_tr
                train_kw = {}

            def set_torch_seed(name: str) -> None:
                off = {
                    "one_pole": 11,
                    "gru": 13,
                    "clockwork": 17,
                    "gru_matched": 23,
                    "diag_y_gru": 19,
                    "diag_dual_gru": 31,
                }[name]
                torch.manual_seed((seed_tr + off * 1_000_003 + s * 97) % (2**31))

            dec_list = [
                ("one_pole", lambda: OnePoleDecoder(hidden_dim=hidden_dim, tau=tau_star)),
                ("gru", lambda: GRUDecoder(hidden_dim=hidden_dim)),
                (
                    "clockwork",
                    lambda: ClockworkDecoder(
                        hidden_dim=hidden_dim, clock_periods=clock_periods
                    ),
                ),
            ]
            if run_gru_matched:
                dec_list.append(("gru_matched", lambda: GRUDecoder(hidden_dim=hm)))

            rescue_results = {}
            for name, factory in dec_list:
                print(f"  Training {name}...", flush=True)
                set_torch_seed(name)
                decoder = factory()
                decoder = decoder.to(device)
                if torch_compile and hasattr(torch, "compile"):
                    decoder = torch.compile(decoder)  # type: ignore[assignment]
                train_decoder(
                    decoder,
                    beliefs_fit,
                    theta_fit,
                    z_fit,
                    n_epochs=n_epochs,
                    lr=lr,
                    burnin=burnin,
                    device=device,
                    log_interval=log_interval,
                    **train_kw,
                )
                metrics = eval_decoder(
                    decoder,
                    beliefs_ev,
                    theta_ev,
                    z_ev,
                    burnin=burnin,
                    device=device,
                )
                frac, frac_t, frac_z = rescue_fractions(
                    metrics,
                    single_tm,
                    single_zm,
                    single_mse,
                    dual_tm,
                    dual_zm,
                    dual_mse,
                )
                rescue_results[name] = {
                    **metrics,
                    "fraction": frac,
                    "fraction_theta": frac_t,
                    "fraction_z": frac_z,
                }
                print(
                    f"    mse={metrics['joint_mse']:.4f} frac={frac:.3f} "
                    f"frac_theta={frac_t:.3f} frac_z={frac_z:.3f}",
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
                    theta_fit,
                    z_fit,
                    n_epochs=n_epochs,
                    lr=lr,
                    burnin=burnin,
                    device=device,
                    log_interval=log_interval,
                    **train_kw,
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

            diag_dual = {}
            if run_diagnostic_dual_gru:
                print(
                    "  Training diag_dual_gru (dual beliefs -> GRU, diagnostic)...",
                    flush=True,
                )
                set_torch_seed("diag_dual_gru")
                _, _, b_dual_tr = dual_filter_with_beliefs(y_tr, eps, rho, a, b, sigma)
                _, _, b_dual_ev = dual_filter_with_beliefs(y_ev, eps, rho, a, b, sigma)
                dd = GRUDecoder(input_dim=4, hidden_dim=hidden_dim)
                dd = dd.to(device)
                if torch_compile and hasattr(torch, "compile"):
                    dd = torch.compile(dd)  # type: ignore[assignment]
                train_decoder(
                    dd,
                    b_dual_tr,
                    theta_fit,
                    z_fit,
                    n_epochs=n_epochs,
                    lr=lr,
                    burnin=burnin,
                    device=device,
                    log_interval=log_interval,
                    **train_kw,
                )
                diag_dual = eval_decoder(
                    dd, b_dual_ev, theta_ev, z_ev, burnin=burnin, device=device
                )
                print(
                    f"    [diagnostic] dual->GRU mse={diag_dual['joint_mse']:.4f}",
                    flush=True,
                )

            row_seed = {
                "dual_mse": dual_mse,
                "dual_theta_mse": dual_tm,
                "dual_z_mse": dual_zm,
                "dual_weighted_cls": dual_cls["weighted_cls"],
                "dual_theta_cls": dual_cls["theta_cls"],
                "dual_z_cls": dual_cls["z_cls"],
                "single_mse": single_mse,
                "single_theta_mse": single_tm,
                "single_z_mse": single_zm,
                "single_weighted_cls": single_cls["weighted_cls"],
                "single_theta_cls": single_cls["theta_cls"],
                "single_z_cls": single_cls["z_cls"],
                "base_gap": base_gap,
                "gap_theta": gap_theta,
                "gap_z": gap_z,
                "tau_star": tau_star,
            }
            for name, m in rescue_results.items():
                row_seed[f"{name}_mse"] = m["joint_mse"]
                row_seed[f"{name}_theta_mse"] = m["theta_mse"]
                row_seed[f"{name}_z_mse"] = m["z_mse"]
                row_seed[f"{name}_weighted_cls"] = m["weighted_cls"]
                row_seed[f"{name}_theta_cls"] = m["theta_cls"]
                row_seed[f"{name}_z_cls"] = m["z_cls"]
                row_seed[f"{name}_fraction"] = m["fraction"]
                row_seed[f"{name}_fraction_theta"] = m["fraction_theta"]
                row_seed[f"{name}_fraction_z"] = m["fraction_z"]

            if run_diagnostic_y_gru:
                row_seed["diag_y_gru_mse"] = diag_y["joint_mse"]
                row_seed["diag_y_gru_theta_mse"] = diag_y["theta_mse"]
                row_seed["diag_y_gru_z_mse"] = diag_y["z_mse"]
                row_seed["diag_y_gru_weighted_cls"] = diag_y["weighted_cls"]
                row_seed["diag_y_gru_theta_cls"] = diag_y["theta_cls"]
                row_seed["diag_y_gru_z_cls"] = diag_y["z_cls"]

            if run_diagnostic_dual_gru:
                row_seed["diag_dual_gru_mse"] = diag_dual["joint_mse"]
                row_seed["diag_dual_gru_theta_mse"] = diag_dual["theta_mse"]
                row_seed["diag_dual_gru_z_mse"] = diag_dual["z_mse"]
                row_seed["diag_dual_gru_weighted_cls"] = diag_dual["weighted_cls"]
                row_seed["diag_dual_gru_theta_cls"] = diag_dual["theta_cls"]
                row_seed["diag_dual_gru_z_cls"] = diag_dual["z_cls"]

            seed_rows.append(row_seed)

            if checkpoint_path is not None and config_signature is not None:
                save_rescue_checkpoint(
                    checkpoint_path,
                    config_signature,
                    results,
                    partial_rho_idx=idx,
                    partial_seed_rows=seed_rows,
                )

        # Aggregate across seeds
        row = {"lambda": lam, "rho": rho}
        for key in seed_rows[0]:
            vals = [r[key] for r in seed_rows]
            mean, ci = summarize_metric(vals)
            row[key] = mean
            row[f"{key}_ci95"] = ci
        results.append(row)

        if checkpoint_path is not None and config_signature is not None:
            save_rescue_checkpoint(
                checkpoint_path,
                config_signature,
                results,
                None,
                None,
            )

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


def make_rescue_plot(results, output_path, suptitle_note: str = ""):
    lambdas = np.array([r["lambda"] for r in results])
    r0 = results[0] if results else {}

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    if suptitle_note:
        fig.suptitle(suptitle_note, fontsize=12, y=1.01)

    # Row 0 left: MSE + 95% CI bands
    ax = axes[0, 0]
    series_mse = [
        ("dual_mse", "Dual filter (ceiling)", "-", None),
        ("single_mse", "Single filter (floor)", "-", None),
        ("one_pole_mse", "+ OnePole (protocol)", "--", "C0"),
        ("gru_mse", "+ GRU (protocol)", "--", "C1"),
        ("clockwork_mse", "+ Clockwork (protocol)", "--", "C2"),
    ]
    if "gru_matched_mse" in r0:
        series_mse.append(
            ("gru_matched_mse", "+ GRU (param-matched)", "--", "C4")
        )
    for key, label, ls, color in series_mse:
        if key not in r0:
            continue
        y = np.array([r[key] for r in results])
        ci = np.array([r[f"{key}_ci95"] for r in results])
        kw = {"linewidth": 2, "linestyle": ls, "label": label}
        if color is not None:
            kw["color"] = color
        ax.plot(lambdas, y, **kw)
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color=color or (0.5, 0.5, 0.5))

    if results and "diag_y_gru_mse" in r0:
        y = np.array([r["diag_y_gru_mse"] for r in results])
        ci = np.array([r["diag_y_gru_mse_ci95"] for r in results])
        ax.plot(
            lambdas,
            y,
            linewidth=2,
            linestyle=":",
            color="C3",
            label="GRU | y (diagnostic)",
        )
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color="C3")
    if results and "diag_dual_gru_mse" in r0:
        y = np.array([r["diag_dual_gru_mse"] for r in results])
        ci = np.array([r["diag_dual_gru_mse_ci95"] for r in results])
        ax.plot(
            lambdas,
            y,
            linewidth=2,
            linestyle="-.",
            color="C5",
            label="GRU | dual beliefs (diagnostic)",
        )
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.12, color="C5")

    ax.set_xscale("log")
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel("Joint MSE (eval set)")
    ax.set_title("MSE (eval + 95% CI)")
    ax.legend(fontsize=7, loc="best")

    # Row 0 right: joint rescue fraction
    ax = axes[0, 1]
    for name, color in [
        ("one_pole", "C0"),
        ("gru", "C1"),
        ("clockwork", "C2"),
        ("gru_matched", "C4"),
    ]:
        k = f"{name}_fraction"
        if k not in r0:
            continue
        fracs = np.array([r[k] for r in results])
        ci = np.array([r[f"{k}_ci95"] for r in results])
        ax.plot(lambdas, fracs, label=name, linewidth=2, color=color)
        lo, hi = _ci_band(fracs, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.2, color=color)
    ax.axhline(0.0, color="gray", linestyle="--", label="no rescue")
    ax.axhline(1.0, color="gray", linestyle=":", label="full rescue")
    ax.set_xscale("log")
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel("Joint rescue fraction")
    ax.set_title("Q3: joint MSE rescue fraction")
    ax.legend(fontsize=7)

    # Row 1 left: weighted classification error + CI
    ax = axes[1, 0]
    series_cls = [
        ("dual_weighted_cls", "Dual filter (ceiling)", "-", None),
        ("single_weighted_cls", "Single filter (floor)", "-", None),
        ("one_pole_weighted_cls", "+ OnePole (protocol)", "--", "C0"),
        ("gru_weighted_cls", "+ GRU (protocol)", "--", "C1"),
        ("clockwork_weighted_cls", "+ Clockwork (protocol)", "--", "C2"),
    ]
    if "gru_matched_mse" in r0:
        series_cls.append(
            ("gru_matched_weighted_cls", "+ GRU matched", "--", "C4")
        )
    for key, label, ls, color in series_cls:
        if key not in r0:
            continue
        y = np.array([r[key] for r in results])
        ci = np.array([r[f"{key}_ci95"] for r in results])
        kw = {"linewidth": 2, "linestyle": ls, "label": label}
        if color is not None:
            kw["color"] = color
        ax.plot(lambdas, y, **kw)
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color=color or (0.5, 0.5, 0.5))

    if results and "diag_y_gru_weighted_cls" in r0:
        y = np.array([r["diag_y_gru_weighted_cls"] for r in results])
        ci = np.array([r["diag_y_gru_weighted_cls_ci95"] for r in results])
        ax.plot(
            lambdas,
            y,
            linewidth=2,
            linestyle=":",
            color="C3",
            label="GRU | y (diagnostic)",
        )
        lo, hi = _ci_band(y, ci)
        ax.fill_between(lambdas, lo, hi, alpha=0.15, color="C3")

    ax.set_xscale("log")
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel(r"Weighted classification error")
    ax.set_title("Weighted classification error (eval + 95% CI)")
    ax.legend(fontsize=7, loc="best")

    # Row 1 right: sanity GRU comparisons (MSE)
    ax = axes[1, 1]
    gru_m = np.array([r["gru_mse"] for r in results])
    ax.plot(lambdas, gru_m, label="GRU | single beliefs", linewidth=2, color="C1")
    ax.fill_between(
        lambdas,
        *_ci_band(gru_m, np.array([r["gru_mse_ci95"] for r in results])),
        alpha=0.15,
        color="C1",
    )
    if results and "diag_y_gru_mse" in r0:
        dy = np.array([r["diag_y_gru_mse"] for r in results])
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
            *_ci_band(dy, np.array([r["diag_y_gru_mse_ci95"] for r in results])),
            alpha=0.15,
            color="C3",
        )
    if results and "diag_dual_gru_mse" in r0:
        dd = np.array([r["diag_dual_gru_mse"] for r in results])
        ax.plot(
            lambdas,
            dd,
            label="GRU | dual beliefs (diagnostic)",
            linewidth=2,
            linestyle="-.",
            color="C5",
        )
        ax.fill_between(
            lambdas,
            *_ci_band(dd, np.array([r["diag_dual_gru_mse_ci95"] for r in results])),
            alpha=0.12,
            color="C5",
        )
    ax.set_xscale("log")
    ax.set_xlabel("lambda = rho / eps")
    ax.set_ylabel("Joint MSE (eval)")
    ax.set_title("Sanity: decoders on different codes")
    ax.legend(fontsize=7)

    # Row 2: per-latent (theta / z) MSE rescue fractions
    for col, suffix, title in [
        (0, "fraction_theta", r"Rescue fraction: $\theta$ (MSE)"),
        (1, "fraction_z", r"Rescue fraction: $z$ (MSE)"),
    ]:
        ax = axes[2, col]
        for name, color in [
            ("one_pole", "C0"),
            ("gru", "C1"),
            ("clockwork", "C2"),
            ("gru_matched", "C4"),
        ]:
            k = f"{name}_{suffix}"
            if k not in r0:
                continue
            fr = np.array([r[k] for r in results])
            ci = np.array([r[f"{k}_ci95"] for r in results])
            ax.plot(lambdas, fr, label=name, linewidth=2, color=color)
            lo, hi = _ci_band(fr, ci)
            ax.fill_between(lambdas, lo, hi, alpha=0.2, color=color)
        ax.axhline(0.0, color="gray", linestyle="--")
        ax.axhline(1.0, color="gray", linestyle=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("lambda = rho / eps")
        ax.set_ylabel("Rescue fraction")
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.98 if suptitle_note else 1))
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


CHECKPOINT_FORMAT = "rescue_decoder_checkpoint_v1"


def default_checkpoint_path(output_prefix: Path) -> Path:
    return output_prefix.parent / f"{output_prefix.name}.checkpoint.json"


def build_rescue_config_signature(
    *,
    T: int,
    eps: float,
    rhos: np.ndarray,
    a: float,
    b: float,
    sigma: float,
    n_tau: int,
    n_seeds: int,
    base_seed: int,
    n_epochs: int,
    lr: float,
    clock_periods: tuple,
    hidden_dim: int,
    run_diagnostic_y_gru: bool,
    run_diagnostic_dual_gru: bool,
    run_gru_matched: bool,
    val_frac: float,
    early_stopping_patience: int,
    torch_compile: bool,
    device_str: str,
) -> dict:
    """Stable dict for resume validation (must match exactly)."""
    return {
        "signature_version": 1,
        "T": int(T),
        "eps": float(eps),
        "rhos": [float(x) for x in np.asarray(rhos, dtype=float).ravel()],
        "a": float(a),
        "b": float(b),
        "sigma": float(sigma),
        "n_tau": int(n_tau),
        "n_seeds": int(n_seeds),
        "base_seed": int(base_seed),
        "n_epochs": int(n_epochs),
        "lr": float(lr),
        "clock_periods": [int(x) for x in clock_periods],
        "hidden_dim": int(hidden_dim),
        "run_diagnostic_y_gru": bool(run_diagnostic_y_gru),
        "run_diagnostic_dual_gru": bool(run_diagnostic_dual_gru),
        "run_gru_matched": bool(run_gru_matched),
        "val_frac": float(val_frac),
        "early_stopping_patience": int(early_stopping_patience),
        "torch_compile": bool(torch_compile),
        "device_str": str(device_str),
    }


def _configs_match(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if a[k] != b[k]:
            return False
    return True


def save_rescue_checkpoint(
    path: Path,
    config: dict,
    completed_results: list,
    partial_rho_idx: int | None,
    partial_seed_rows: list | None,
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": to_jsonable(config),
        "completed_results": to_jsonable(completed_results),
        "partial_rho_idx": partial_rho_idx,
        "partial_seed_rows": to_jsonable(partial_seed_rows) if partial_seed_rows else None,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_rescue_checkpoint(path: Path, expected_config: dict) -> tuple[list, int | None, list | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unknown checkpoint format in {path}")
    cfg = raw["config"]
    if not _configs_match(cfg, expected_config):
        raise ValueError(
            "Checkpoint config does not match this run (T, rhos, seeds, epochs, etc.). "
            "Use the same CLI as when the checkpoint was written, or delete the checkpoint file."
        )
    completed = raw.get("completed_results") or []
    pr = raw.get("partial_rho_idx")
    ps = raw.get("partial_seed_rows")
    if pr is not None and ps is not None:
        return completed, int(pr), list(ps)
    return completed, None, None


def load_checkpoint_raw(path: Path) -> dict:
    """Load checkpoint JSON without config validation (for --report-checkpoint)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a rescue_decoder checkpoint (expected format={CHECKPOINT_FORMAT!r})."
        )
    return raw


def report_checkpoint_progress(path: Path, plot_path: Path | None = None) -> None:
    """Print human-readable progress from a checkpoint file; optionally save partial plot."""
    raw = load_checkpoint_raw(path)
    cfg = raw["config"]
    rhos = cfg.get("rhos") or []
    done = raw.get("completed_results") or []
    pr = raw.get("partial_rho_idx")
    ps = raw.get("partial_seed_rows")

    print(f"Checkpoint file: {path.resolve()}")
    print(f"  T={cfg.get('T')}, n_epochs={cfg.get('n_epochs')}, n_seeds={cfg.get('n_seeds')}, device={cfg.get('device_str')}")
    print(f"  Full rho grid: {len(rhos)} values (lambda = rho/eps)")
    print(f"  Completed rho rows (aggregated): {len(done)}")
    if done:
        lam0, lam1 = float(done[0]["lambda"]), float(done[-1]["lambda"])
        print(f"  Lambda range covered so far: {lam0:.6g} .. {lam1:.6g}")
        last = done[-1]
        gap = float(last["single_mse"]) - float(last["dual_mse"])
        print(
            f"  Last row: dual_joint={float(last['dual_mse']):.4f}  "
            f"single_joint={float(last['single_mse']):.4f}  "
            f"gap={gap:.4f}"
        )
    n_seeds = int(cfg.get("n_seeds", 1))
    if pr is not None and ps is not None:
        print(
            f"  In progress: rho index {pr} (grid position {pr + 1}/{len(rhos)}), "
            f"seeds finished for this rho: {len(ps)}/{n_seeds}"
        )
    elif len(done) < len(rhos):
        print(
            f"  Remaining rho values: {len(rhos) - len(done)} "
            f"(next grid index {len(done)})"
        )
    else:
        print("  All rho rows in grid appear complete (or checkpoint is stale).")

    if plot_path is not None:
        if not done:
            print("  --report-plot: skipped (no completed rows to plot).")
        else:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            note = f"Partial progress ({len(done)}/{len(rhos)} rho values) — checkpoint"
            make_rescue_plot(done, plot_path, suptitle_note=note)
            print(f"  Wrote partial plot: {plot_path.resolve()}")


def build_param_counts(hidden_dim, clock_periods, gru_matched_hidden: int):
    return {
        "one_pole": count_parameters(
            OnePoleDecoder(hidden_dim=hidden_dim, tau=1.0)
        ),
        "gru_beliefs": count_parameters(GRUDecoder(hidden_dim=hidden_dim)),
        "gru_matched": count_parameters(
            GRUDecoder(hidden_dim=gru_matched_hidden)
        ),
        "clockwork": count_parameters(
            ClockworkDecoder(hidden_dim=hidden_dim, clock_periods=clock_periods)
        ),
        "diag_gru_y": count_parameters(GRUDecoder(input_dim=1, hidden_dim=hidden_dim)),
        "diag_dual_gru": count_parameters(GRUDecoder(input_dim=4, hidden_dim=hidden_dim)),
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
        "--no-diagnostic-dual-gru",
        action="store_true",
        help="Skip dual beliefs -> GRU sanity check (not part of Q3 protocol).",
    )
    p.add_argument(
        "--no-gru-matched",
        action="store_true",
        help="Skip parameter-matched GRU ablation (vs OnePole count).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="Fraction of train sequence held out for early stopping (0 = disabled).",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop if val joint MSE does not improve for this many epochs (0 = off).",
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
        help="Wrap decoders with torch.compile (PyTorch 2+). Helps GRU on GPU; can help CPU too.",
    )
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="PyTorch CPU intra-op threads (0 = OMP_NUM_THREADS if set, else all cores). Speeds CPU GRU/MKL.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint JSON path (default: <output-prefix>.checkpoint.json next to outputs).",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint/resume (no periodic saves).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from --checkpoint if config matches (same rho grid, T, seeds, etc.).",
    )
    p.add_argument(
        "--fresh-start",
        action="store_true",
        help="Delete checkpoint file before running (discard partial progress).",
    )
    p.add_argument(
        "--report-checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load checkpoint JSON, print progress summary, and exit (see --report-plot).",
    )
    p.add_argument(
        "--report-plot",
        type=Path,
        default=None,
        help="With --report-checkpoint, save partial multi-panel PNG to this path.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.report_checkpoint is not None:
        report_checkpoint_progress(args.report_checkpoint, args.report_plot)
        return

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rhos = np.logspace(np.log10(args.eps), np.log10(args.rho_max), args.n_rho)

    cpu_threads = configure_pytorch_cpu_threads(args.cpu_threads)
    print(f"PyTorch CPU threads: {cpu_threads}", flush=True)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"PyTorch device: {device}", flush=True)

    ck_path: Path | None = None
    if not args.no_checkpoint:
        ck_path = args.checkpoint if args.checkpoint is not None else default_checkpoint_path(args.output_prefix)
        if args.fresh_start and ck_path.is_file():
            ck_path.unlink()
            print(f"Removed checkpoint (--fresh-start): {ck_path}", flush=True)
        elif ck_path.is_file() and not args.resume:
            raise SystemExit(
                f"Checkpoint exists: {ck_path}\n"
                "Use --resume to continue from it, or --fresh-start to delete it and run from scratch."
            )

    clock_periods = tuple(args.clock_periods)
    hidden_dim = args.hidden_dim
    hm, one_pole_param_target, gru_matched_n = gru_hidden_dim_matched_to_onepole(
        4, hidden_dim
    )
    param_counts = build_param_counts(hidden_dim, clock_periods, hm)

    config_signature = build_rescue_config_signature(
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
        run_diagnostic_dual_gru=not args.no_diagnostic_dual_gru,
        run_gru_matched=not args.no_gru_matched,
        val_frac=args.val_frac,
        early_stopping_patience=args.early_stopping_patience,
        torch_compile=args.torch_compile,
        device_str=str(device),
    )

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
        run_diagnostic_dual_gru=not args.no_diagnostic_dual_gru,
        run_gru_matched=not args.no_gru_matched,
        val_frac=args.val_frac,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        log_interval=args.epoch_log_interval,
        torch_compile=args.torch_compile,
        checkpoint_path=ck_path,
        resume=args.resume,
        config_signature=config_signature if ck_path is not None else None,
    )

    meta = {
        "torch_device": str(device),
        "torch_cpu_threads": cpu_threads,
        "T_train": args.T // 2,
        "T_eval": args.T - args.T // 2,
        "burnin": int(1.0 / args.eps),
        "val_frac": args.val_frac,
        "early_stopping_patience": args.early_stopping_patience,
        "gru_matched_hidden_dim": hm,
        "one_pole_param_target_for_match": one_pole_param_target,
        "gru_matched_param_count": gru_matched_n,
        "param_counts": param_counts,
        "hidden_dim": hidden_dim,
        "clock_periods": list(clock_periods),
        "gru_num_layers": 2,
        "protocol_note": (
            "OnePole = admissible single-timescale rescue. GRU and Clockwork at "
            "the same hidden_dim are stress tests (different parameter counts; "
            "multiscale inductive bias). gru_matched approximates OnePole "
            "parameter count. Diagnostics (y->GRU, dual->GRU) are not Q3 protocol."
        ),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }

    save_rescue_json(results, meta, args.output_prefix.with_suffix(".json"))
    save_rescue_csv(results, args.output_prefix.with_suffix(".csv"))
    make_rescue_plot(results, args.output_prefix.with_suffix(".png"))

    if ck_path is not None and ck_path.is_file():
        ck_path.unlink()
        print(f"Removed checkpoint after successful run: {ck_path}", flush=True)


if __name__ == "__main__":
    main()
