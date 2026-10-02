from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import torch

_TORCH_AVAILABLE = True

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from padopauli import (
    CliffordGate,
    PauliRotation,
    PauliSum,
    compile_program,
    resolve_preset,
)
from safe_maqaoa.problem_factory import IsingProblem, load_problem


def _require_pennylane():
    try:
        import pennylane as qml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PennyLane is required for exact finetuning.") from exc
    return qml


def _build_ising_observable(problem: IsingProblem) -> PauliSum:
    obs = PauliSum(int(problem.n_qubits))
    for u, v, j in problem.edges:
        if float(j) != 0.0:
            obs.add_from_str("ZZ", float(j), qubits=[int(u), int(v)])
    for q, h in problem.fields:
        if float(h) != 0.0:
            obs.add_from_str("Z", float(h), qubits=[int(q)])
    return obs


def _ising_energy_from_spins(problem: IsingProblem, spins: np.ndarray) -> float:
    s = np.asarray(spins, dtype=np.int8).reshape(-1)
    e = 0.0
    for u, v, j in problem.edges:
        e += float(j) * float(s[int(u)] * s[int(v)])
    for q, h in problem.fields:
        e += float(h) * float(s[int(q)])
    return float(e)


def _simulated_annealing_ground_sample(
    *,
    problem: IsingProblem,
    seed: int,
    restarts: int,
    sweeps: int,
    t_max: float,
    t_min: float,
) -> Tuple[float, List[int], List[int]]:
    n = int(problem.n_qubits)
    if n <= 0:
        raise ValueError("problem.n_qubits must be positive")
    if int(restarts) <= 0 or int(sweeps) <= 0:
        raise ValueError("restarts and sweeps must be positive")

    h_vec = np.zeros((n,), dtype=np.float64)
    j_mat = np.zeros((n, n), dtype=np.float64)
    for q, h in problem.fields:
        h_vec[int(q)] += float(h)
    for u, v, j in problem.edges:
        uu = int(u)
        vv = int(v)
        w = float(j)
        j_mat[uu, vv] += w
        j_mat[vv, uu] += w

    rng = np.random.default_rng(int(seed))
    best_e = float("inf")
    best_s = np.ones((n,), dtype=np.int8)

    temp_sched = np.geomspace(float(t_max), float(t_min), num=int(sweeps)).astype(np.float64)
    for _ in range(int(restarts)):
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n,), replace=True).astype(np.int8)
        e_curr = _ising_energy_from_spins(problem, s)

        if e_curr < best_e:
            best_e = float(e_curr)
            best_s = s.copy()

        for t in temp_sched:
            order = rng.permutation(n)
            for i in order:
                local_field = h_vec[int(i)] + float(np.dot(j_mat[int(i)], s.astype(np.float64)))
                # Ising flip energy change: E_new - E_old = -2 * s_i * local_field
                delta_e = -2.0 * float(s[int(i)]) * float(local_field)
                if (delta_e <= 0.0) or (rng.random() < np.exp(-delta_e / max(float(t), 1e-12))):
                    s[int(i)] = np.int8(-s[int(i)])
                    e_curr += float(delta_e)
                    if e_curr < best_e:
                        best_e = float(e_curr)
                        best_s = s.copy()

    bits = [0 if int(v) == 1 else 1 for v in best_s.tolist()]
    return float(best_e), [int(v) for v in best_s.tolist()], bits


def _build_ma_qaoa_circuit(problem: IsingProblem, p_layers: int) -> Tuple[List[Any], int]:
    circuit: List[Any] = []
    n_qubits = int(problem.n_qubits)

    for q in range(n_qubits):
        circuit.append(CliffordGate("H", [int(q)]))

    pidx = 0
    for _ in range(int(p_layers)):
        for (u, v, _j) in problem.edges:
            circuit.append(PauliRotation("ZZ", [int(u), int(v)], param_idx=int(pidx)))
            pidx += 1
        for (q, _h) in problem.fields:
            circuit.append(PauliRotation("Z", [int(q)], param_idx=int(pidx)))
            pidx += 1
        for q in range(n_qubits):
            circuit.append(PauliRotation("X", [int(q)], param_idx=int(pidx)))
            pidx += 1

    return circuit, int(pidx)


def _choose_torch_device(device_arg: str) -> str:
    if str(device_arg) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device_arg)


def _to_qml_obs(problem: IsingProblem, qml):
    terms = []
    for u, v, j in problem.edges:
        terms.append(float(j) * (qml.PauliZ(int(u)) @ qml.PauliZ(int(v))))
    for q, h in problem.fields:
        terms.append(float(h) * qml.PauliZ(int(q)))
    if not terms:
        raise ValueError("Problem has no observable terms.")
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def _apply_circuit_qml(circuit: Sequence[Any], params: Any, qml) -> None:
    shape = tuple(int(x) for x in np.shape(params))
    if len(shape) == 1:
        batched = False
        width = int(shape[0])
    elif len(shape) == 2:
        batched = True
        width = int(shape[1])
    else:
        raise ValueError(f"params rank must be 1 or 2, got shape={shape!r}")

    for gate in circuit:
        gname = gate.__class__.__name__
        if gname == "CliffordGate":
            symbol = str(gate.symbol).upper()
            if symbol == "H":
                qml.Hadamard(wires=gate.qubits[0])
            elif symbol == "X":
                qml.PauliX(wires=gate.qubits[0])
            elif symbol == "Y":
                qml.PauliY(wires=gate.qubits[0])
            elif symbol == "Z":
                qml.PauliZ(wires=gate.qubits[0])
            elif symbol == "S":
                qml.S(wires=gate.qubits[0])
            elif symbol == "CNOT":
                qml.CNOT(wires=gate.qubits)
            elif symbol == "CZ":
                qml.CZ(wires=gate.qubits)
            elif symbol == "SWAP":
                qml.SWAP(wires=gate.qubits)
            else:
                raise ValueError(f"Unsupported Clifford symbol: {symbol}")
            continue

        if gname == "PauliRotation":
            pidx = int(gate.param_idx)
            if pidx < 0 or pidx >= width:
                raise ValueError(f"Invalid param_idx={pidx} for params width={width}")
            angle = params[pidx] if (not batched) else params[:, pidx]
            qml.PauliRot(angle, str(gate.pauli), wires=list(gate.qubits))
            continue

        raise TypeError(f"Unsupported gate class for exact qnode: {gname}")


def _thetas_to_numpy(thetas: Any) -> np.ndarray:
    if _TORCH_AVAILABLE and isinstance(thetas, torch.Tensor):
        return np.asarray(thetas.detach().cpu().numpy(), dtype=np.float64)
    return np.asarray(thetas, dtype=np.float64)


def _thetas_to_torch(
    thetas: Any,
    *,
    device: Optional[str] = None,
    requires_grad: bool = False,
) -> Tensor:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for this code path.")

    if isinstance(thetas, torch.Tensor):
        t = thetas if requires_grad else thetas.detach()
        if device is not None and str(t.device) != str(device):
            t = t.to(device)
        if t.dtype != torch.float64:
            t = t.to(dtype=torch.float64)
        if requires_grad and (not t.requires_grad or not t.is_leaf):
            t = t.clone().detach().requires_grad_(True)
        return t

    t = torch.as_tensor(thetas, dtype=torch.float64, device=device)
    if requires_grad:
        t = t.clone().detach().requires_grad_(True)
    return t


def _collapse_batched_exact_output(exact_e: torch.Tensor, batch_size: int) -> torch.Tensor:
    if exact_e.ndim == 0:
        return exact_e.reshape(1)
    if exact_e.ndim == 1:
        return exact_e.reshape(-1)
    if exact_e.ndim == 2 and int(exact_e.shape[0]) == int(batch_size) and int(exact_e.shape[1]) == int(batch_size):
        return torch.diagonal(exact_e, dim1=0, dim2=1)
    return exact_e.reshape(-1)


def _build_exact_qnode(circuit: Sequence[Any], problem: IsingProblem, device_name: str):
    qml = _require_pennylane()
    import pennylane.numpy as pnp  # type: ignore

    obs = _to_qml_obs(problem, qml)

    candidates = (
        [str(device_name)]
        if device_name
        else ["lightning.gpu", "lightning.amdgpu", "lightning.qubit", "default.qubit"]
    )
    last_error: Exception | None = None
    for cand in candidates:
        try:
            dev = qml.device(cand, wires=int(problem.n_qubits))
        except Exception as exc:  # pragma: no cover
            last_error = exc
            continue

        build_attempts = (
            ("torch", "backprop", torch),
            ("torch", "adjoint", torch),
            ("autograd", "adjoint" if str(cand).startswith("lightning.") else "backprop", pnp),
        )
        for interface_name, diff, lib in build_attempts:
            try:
                @qml.qnode(dev, interface=interface_name, diff_method=diff)
                def qnode(params):
                    _apply_circuit_qml(circuit, params, qml)
                    return qml.expval(obs)

                return qnode, str(cand), str(diff), lib
            except Exception as exc:  # pragma: no cover
                last_error = exc
                continue
    raise RuntimeError(f"Failed to build exact qnode from {candidates!r}") from last_error


def _exact_energy_batch(qnode: Any, exact_lib: Any, thetas: torch.Tensor, exact_dev: str) -> torch.Tensor:
    if thetas.dim() != 2:
        raise ValueError("thetas must be (batch, n_params)")

    if exact_lib is torch:
        qnode_device = "cuda" if "gpu" in str(exact_dev).lower() else "cpu"
        target_device = thetas.device if isinstance(thetas, torch.Tensor) else qnode_device
        batch_size = int(thetas.shape[0])
        thetas_t = _thetas_to_torch(thetas, device=qnode_device, requires_grad=False)
        try:
            exact_e = qnode(thetas_t)
            if not torch.is_tensor(exact_e):
                exact_e = torch.as_tensor(exact_e, dtype=thetas_t.dtype, device=thetas_t.device)
            exact_e = _collapse_batched_exact_output(exact_e, batch_size=batch_size)
            if int(exact_e.shape[0]) != batch_size:
                raise ValueError(
                    f"batched exact qnode returned shape {tuple(exact_e.shape)} for batch_size={batch_size}"
                )
            return exact_e.to(device=target_device)
        except Exception:
            vals: List[Tensor] = []
            for row in thetas_t:
                v = qnode(row)
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v, dtype=thetas_t.dtype, device=thetas_t.device)
                vals.append(v.reshape(-1)[0] if v.dim() > 0 else v)
            return torch.stack(vals, dim=0).to(device=target_device)

    theta_np = _thetas_to_numpy(thetas)
    vals: List[float] = []
    for row in theta_np:
        v = qnode(exact_lib.array(row, requires_grad=False))
        vals.append(float(v))
    return torch.as_tensor(vals, dtype=thetas.dtype, device=thetas.device)


def _exact_energy_and_grad_batch(qnode: Any, exact_lib: Any, thetas: torch.Tensor, exact_dev: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if thetas.dim() != 2:
        raise ValueError("thetas must be (batch, n_params)")

    if exact_lib is torch:
        qnode_device = "cuda" if "gpu" in str(exact_dev).lower() else "cpu"
        target_device = thetas.device if isinstance(thetas, torch.Tensor) else qnode_device
        batch_size = int(thetas.shape[0])
        thetas_t = _thetas_to_torch(thetas, device=qnode_device, requires_grad=True)
        try:
            exact_e = qnode(thetas_t)
            if not torch.is_tensor(exact_e):
                exact_e = torch.as_tensor(exact_e, dtype=thetas_t.dtype, device=thetas_t.device)
            exact_e_batch = _collapse_batched_exact_output(exact_e, batch_size=batch_size)
            if int(exact_e_batch.shape[0]) != batch_size:
                raise ValueError(
                    f"batched exact qnode returned shape {tuple(exact_e.shape)} for batch_size={batch_size}"
                )
            exact_g = torch.autograd.grad(exact_e_batch.sum(), thetas_t, retain_graph=False, create_graph=False)[0]
            return exact_e_batch.detach().to(device=target_device), exact_g.detach().to(device=target_device)
        except Exception:
            vals: List[Tensor] = []
            grads: List[Tensor] = []
            for row in thetas_t:
                row_t = row.detach().clone().requires_grad_(True)
                v = qnode(row_t)
                if not torch.is_tensor(v):
                    v = torch.as_tensor(v, dtype=row_t.dtype, device=row_t.device)
                v_flat = v.reshape(-1)
                g = torch.autograd.grad(v_flat.sum(), row_t, retain_graph=False, create_graph=False)[0]
                vals.append(v_flat.detach().reshape(-1)[0])
                grads.append(g.detach())
            return (
                torch.stack(vals, dim=0).reshape(-1).to(device=target_device),
                torch.stack(grads, dim=0).to(device=target_device),
            )

    import autograd as _autograd

    theta_np = _thetas_to_numpy(thetas)
    vals: List[float] = []
    grads: List[np.ndarray] = []
    value_and_grad_factory: Any = _autograd.value_and_grad
    value_and_grad_fn = value_and_grad_factory(qnode)
    for row in theta_np:
        x = exact_lib.array(row, requires_grad=True)
        v, g = value_and_grad_fn(x)
        vals.append(float(v))
        grads.append(np.asarray(g, dtype=np.float64))

    vals_t = torch.as_tensor(vals, dtype=thetas.dtype, device=thetas.device)
    grads_t = torch.as_tensor(np.asarray(grads), dtype=thetas.dtype, device=thetas.device)
    return vals_t, grads_t


def _exact_energy_and_grad_chunked(
    qnode: Any,
    exact_lib: Any,
    thetas: torch.Tensor,
    exact_dev: str,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-split wrapper around ``_exact_energy_and_grad_batch``.

    Evaluates the batch in chunks of ``chunk_size`` inits. Each init's energy
    depends only on its own parameters, so the chunked result agrees with the
    single-batch call up to floating-point reduction order. ``chunk_size <= 0``
    or ``>= batch`` makes a single call.
    """
    batch = int(thetas.shape[0])
    cs = int(chunk_size)
    if cs <= 0 or cs >= batch:
        return _exact_energy_and_grad_batch(qnode, exact_lib, thetas, exact_dev)
    es: List[Tensor] = []
    gs: List[Tensor] = []
    for start in range(0, batch, cs):
        e, g = _exact_energy_and_grad_batch(
            qnode, exact_lib, thetas[start : start + cs], exact_dev
        )
        es.append(e)
        gs.append(g)
    return torch.cat(es, dim=0), torch.cat(gs, dim=0)


@dataclass
class TrainTrace:
    rows: List[Dict[str, Any]]
    snapshots: List[Dict[str, Any]]


def _append_csv_row(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, None) for k in fieldnames})


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class LiveTrainingPlotter:
    def __init__(
        self,
        *,
        batch_size: int,
        init_labels: Optional[Sequence[str]],
        title: str,
        reference_energy: Optional[float],
        refresh_every: int,
        save_path: Path,
        show_window: bool,
    ) -> None:
        import matplotlib.pyplot as plt  # local import for optional dependency

        self.plt = plt
        self.batch_size = int(batch_size)
        self.title = str(title)
        self.reference_energy = float(reference_energy) if reference_energy is not None else None
        self.refresh_every = max(1, int(refresh_every))
        self.save_path = Path(save_path)
        self.show_window = bool(show_window)
        self.init_labels = list(init_labels) if init_labels is not None else [f"init_{i + 1}" for i in range(self.batch_size)]

        self.group_colors: Dict[str, str] = {
            "random": "tab:blue",
            "const": "tab:green",
            "relax": "tab:red",
            "other": "tab:gray",
        }
        self.group_names: Dict[str, str] = {
            "random": "random",
            "const": "const",
            "relax": "relax",
            "other": "other",
        }
        self.group_to_indices: Dict[str, List[int]] = {"random": [], "const": [], "relax": [], "other": []}
        for i, label in enumerate(self.init_labels):
            tag = str(label).lower()
            if tag.startswith("random"):
                self.group_to_indices["random"].append(int(i))
            elif tag.startswith("const"):
                self.group_to_indices["const"].append(int(i))
            elif tag.startswith("qaoa_relax") or tag.startswith("relax"):
                self.group_to_indices["relax"].append(int(i))
            else:
                self.group_to_indices["other"].append(int(i))

        self.lwpp_energy_hist: List[np.ndarray] = []
        self.lwpp_exact_hist: List[np.ndarray] = []
        self.lwpp_step_hist: List[int] = []
        self.fine_exact_hist: List[np.ndarray] = []
        self.direct_exact_hist: List[np.ndarray] = []

        self.phase = "init"
        self.last_step = -1

        self.fig, self.axes = self.plt.subplots(2, 2, figsize=(12.5, 8.0))
        if self.show_window:
            self.plt.ion()
            self.plt.show(block=False)

    @staticmethod
    def _to_np_1d(x: torch.Tensor) -> np.ndarray:
        return np.asarray(x.detach().cpu().tolist(), dtype=np.float64).reshape(-1)

    @staticmethod
    def _stack_or_empty(rows: List[np.ndarray]) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return np.vstack(rows)

    def _plot_grouped_lines(self, ax, mat: np.ndarray, *, alpha: float = 0.9, xs: Optional[np.ndarray] = None) -> None:
        if mat.size == 0:
            return
        xs = np.arange(mat.shape[0], dtype=np.float64) if xs is None else np.asarray(xs, dtype=np.float64)
        width = int(mat.shape[1])
        for group in ("random", "const", "relax", "other"):
            color = self.group_colors[group]
            for idx in self.group_to_indices[group]:
                if int(idx) < width:
                    ax.plot(xs, mat[:, int(idx)], lw=1.5, alpha=alpha, color=color)

    def _add_group_legend(self, ax) -> None:
        from matplotlib.lines import Line2D

        handles = []
        for group in ("random", "const", "relax", "other"):
            if len(self.group_to_indices[group]) > 0:
                handles.append(Line2D([0], [0], color=self.group_colors[group], lw=2, label=self.group_names[group]))
        if len(handles) > 0:
            ax.legend(handles=handles, loc="best", fontsize=8)

    def _refresh(self, *, force: bool = False) -> None:
        if (not force) and self.last_step >= 0 and (self.last_step % self.refresh_every != 0):
            return

        ax_exact_only = self.axes[0, 0]
        ax_lwpp_exact = self.axes[0, 1]
        ax_lwpp = self.axes[1, 0]
        ax_lwpp_abs = self.axes[1, 1]

        for ax in (ax_exact_only, ax_lwpp_exact, ax_lwpp, ax_lwpp_abs):
            ax.clear()

        direct_mat = self._stack_or_empty(self.direct_exact_hist)
        self._plot_grouped_lines(ax_exact_only, direct_mat)
        ax_exact_only.set_title("Direct exact optimization (grouped by init)")
        ax_exact_only.set_xlabel("Iteration")
        ax_exact_only.set_ylabel("Energy")
        ax_exact_only.grid(True, alpha=0.25)
        if self.reference_energy is not None:
            ax_exact_only.axhline(self.reference_energy, color="gray", ls=":", lw=1.2, alpha=0.9)
        self._add_group_legend(ax_exact_only)

        lwpp_exact_mat = self._stack_or_empty(self.lwpp_exact_hist)
        fine_mat = self._stack_or_empty(self.fine_exact_hist)
        # LWPP exact-trace points sit at their true pre-training steps
        # (0, eval_every, 2*eval_every, ...), not at their sample index; the
        # exact fine-tuning then continues after the last LWPP step.
        lwpp_xs = np.asarray(self.lwpp_step_hist, dtype=np.float64)
        if lwpp_exact_mat.size > 0 and fine_mat.size > 0:
            last_lwpp = float(lwpp_xs[-1]) if lwpp_xs.size else 0.0
            fine_xs = last_lwpp + 1.0 + np.arange(fine_mat.shape[0], dtype=np.float64)
            y_mat = np.concatenate([lwpp_exact_mat, fine_mat], axis=0)
            x_all = np.concatenate([lwpp_xs, fine_xs])
            self._plot_grouped_lines(ax_lwpp_exact, y_mat, xs=x_all)
            ax_lwpp_exact.axvline(last_lwpp, color="gray", ls="--", lw=1.0, alpha=0.6)
        elif lwpp_exact_mat.size > 0:
            self._plot_grouped_lines(ax_lwpp_exact, lwpp_exact_mat, xs=lwpp_xs)
        else:
            self._plot_grouped_lines(ax_lwpp_exact, fine_mat)
        ax_lwpp_exact.set_title("LWPP-initialized optimization (grouped by init)")
        ax_lwpp_exact.set_xlabel("Iteration")
        ax_lwpp_exact.set_ylabel("Energy")
        ax_lwpp_exact.grid(True, alpha=0.25)
        if self.reference_energy is not None:
            ax_lwpp_exact.axhline(self.reference_energy, color="gray", ls=":", lw=1.2, alpha=0.9)
        self._add_group_legend(ax_lwpp_exact)

        lwpp_mat = self._stack_or_empty(self.lwpp_energy_hist)
        self._plot_grouped_lines(ax_lwpp, fine_mat)
        ax_lwpp.set_title("Exact finetune (grouped by init)")
        ax_lwpp.set_xlabel("Iteration")
        ax_lwpp.set_ylabel("Energy")
        ax_lwpp.grid(True, alpha=0.25)
        if self.reference_energy is not None:
            ax_lwpp.axhline(self.reference_energy, color="gray", ls=":", lw=1.2, alpha=0.9)
        self._add_group_legend(ax_lwpp)

        self._plot_grouped_lines(ax_lwpp_abs, lwpp_mat, xs=lwpp_xs)
        ax_lwpp_abs.set_title("LWPP pre-optimization energy (grouped by init)")
        ax_lwpp_abs.set_xlabel("Iteration")
        ax_lwpp_abs.set_ylabel("E_LWPP")
        ax_lwpp_abs.grid(True, alpha=0.25, which="both")
        if self.reference_energy is not None:
            ax_lwpp_abs.axhline(self.reference_energy, color="gray", ls=":", lw=1.2, alpha=0.9)
        self._add_group_legend(ax_lwpp_abs)

        self.fig.suptitle(f"{self.title}  |  phase={self.phase}  step={self.last_step}", fontsize=14)
        self.fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.save_path, dpi=150)
        if self.show_window:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def update_lwpp(self, step: int, lwpp_energy: torch.Tensor, exact_energy: torch.Tensor) -> None:
        self.phase = "lwpp"
        self.last_step = int(step)
        self.lwpp_step_hist.append(int(step))
        self.lwpp_energy_hist.append(self._to_np_1d(lwpp_energy))
        self.lwpp_exact_hist.append(self._to_np_1d(exact_energy))
        self._refresh(force=False)

    def update_fine(self, step: int, exact_energy: torch.Tensor) -> None:
        self.phase = "exact_finetune"
        self.last_step = int(step)
        self.fine_exact_hist.append(self._to_np_1d(exact_energy))
        self._refresh(force=False)

    def update_direct(self, step: int, exact_energy: torch.Tensor) -> None:
        self.phase = "exact_only"
        self.last_step = int(step)
        self.direct_exact_hist.append(self._to_np_1d(exact_energy))
        self._refresh(force=False)

    def close(self) -> None:
        self._refresh(force=True)
        if self.show_window:
            self.plt.ioff()


def _train_lwpp_phase(
    *,
    thetas: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lwpp_program: Any,
    exact_qnode: Any,
    exact_lib: Any,
    exact_dev: str,
    steps: int,
    snapshot_every: int,
    log_every: int,
    exact_eval_every: int = 10,
    start_step: int = 0,
    initial_rows: Optional[List[Dict[str, Any]]] = None,
    step_callback: Optional[Any] = None,
    live_plot: Optional[LiveTrainingPlotter] = None,
) -> TrainTrace:
    rows: List[Dict[str, Any]] = list(initial_rows) if initial_rows is not None else []
    snaps: List[Dict[str, Any]] = []

    for step in range(int(start_step), int(steps)):
        optimizer.zero_grad(set_to_none=True)

        lwpp_out = lwpp_program.expvals(thetas, diff_mode="vjp")
        # one Ising observable -> per-init energy
        lwpp_e = lwpp_out[..., 0] if lwpp_out.dim() >= 2 else lwpp_out.reshape(-1)
        if lwpp_e.dim() == 0:
            lwpp_e = lwpp_e.reshape(1)

        loss = torch.mean(lwpp_e)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # The exact energy is trace-only (the LWPP gradient above is the
            # training signal); it is evaluated every exact_eval_every steps and
            # at the final step. A row is still written every step because
            # resume derives the start step from the trace row count.
            do_exact = (
                int(step) % max(1, int(exact_eval_every)) == 0 or int(step) == int(steps) - 1
            )
            exact_e = (
                _exact_energy_batch(exact_qnode, exact_lib, thetas, exact_dev) if do_exact else None
            )
            if live_plot is not None and exact_e is not None:
                live_plot.update_lwpp(int(step), lwpp_e, exact_e)
            row = {
                "step": int(step),
                "lwpp_energy_mean": float(torch.mean(lwpp_e).detach().cpu().item()),
                "lwpp_energy_min": float(torch.min(lwpp_e).detach().cpu().item()),
                "exact_energy_mean": float(torch.mean(exact_e).detach().cpu().item()) if exact_e is not None else None,
                "exact_energy_min": float(torch.min(exact_e).detach().cpu().item()) if exact_e is not None else None,
            }
            rows.append(row)

            if step_callback is not None:
                step_callback(
                    stage="lwpp",
                    step=int(step),
                    row=row,
                    thetas=thetas,
                    optimizer=optimizer,
                    extras={
                        "lwpp_energy": lwpp_e.detach().cpu().tolist(),
                        "exact_energy": exact_e.detach().cpu().tolist() if exact_e is not None else [],
                    },
                )

            if int(step) % max(1, int(log_every)) == 0:
                exact_txt = (
                    f"exact_mean={row['exact_energy_mean']:.6f} exact_min={row['exact_energy_min']:.6f}"
                    if exact_e is not None
                    else "exact=skipped"
                )
                print(
                    f"[LWPP] step={step:04d} "
                    f"lwpp_mean={row['lwpp_energy_mean']:.6f} "
                    f"lwpp_min={row['lwpp_energy_min']:.6f} "
                    f"{exact_txt}"
                )

            if int(step) % max(1, int(snapshot_every)) == 0 or int(step) == int(steps) - 1:
                snaps.append(
                    {
                        "step": int(step),
                        "thetas": thetas.detach().cpu().tolist(),
                        "lwpp_energy": lwpp_e.detach().cpu().tolist(),
                        "exact_energy": exact_e.detach().cpu().tolist() if exact_e is not None else None,
                    }
                )

    return TrainTrace(rows=rows, snapshots=snaps)


def _train_exact_phase(
    *,
    thetas: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    exact_qnode: Any,
    exact_lib: Any,
    exact_dev: str,
    steps: int,
    snapshot_every: int,
    log_every: int,
    prefix: str,
    start_step: int = 0,
    initial_rows: Optional[List[Dict[str, Any]]] = None,
    step_callback: Optional[Any] = None,
    live_plot: Optional[LiveTrainingPlotter] = None,
    live_stage: str = "exact_only",
    exact_chunk_size: int = 1,
) -> TrainTrace:
    rows: List[Dict[str, Any]] = list(initial_rows) if initial_rows is not None else []
    snaps: List[Dict[str, Any]] = []

    for step in range(int(start_step), int(steps)):
        optimizer.zero_grad(set_to_none=True)

        exact_e, exact_g = _exact_energy_and_grad_chunked(
            exact_qnode, exact_lib, thetas, exact_dev, exact_chunk_size
        )
        thetas.grad = exact_g
        optimizer.step()

        with torch.no_grad():
            if live_plot is not None:
                if str(live_stage) == "exact_finetune":
                    live_plot.update_fine(int(step), exact_e)
                else:
                    live_plot.update_direct(int(step), exact_e)
            row = {
                "step": int(step),
                "exact_energy_mean": float(torch.mean(exact_e).detach().cpu().item()),
                "exact_energy_min": float(torch.min(exact_e).detach().cpu().item()),
            }
            rows.append(row)

            if step_callback is not None:
                step_callback(
                    stage=str(live_stage),
                    step=int(step),
                    row=row,
                    thetas=thetas,
                    optimizer=optimizer,
                    extras={
                        "exact_energy": exact_e.detach().cpu().tolist(),
                    },
                )

            if int(step) % max(1, int(log_every)) == 0:
                print(
                    f"[{prefix}] step={step:04d} "
                    f"exact_mean={row['exact_energy_mean']:.6f} "
                    f"exact_min={row['exact_energy_min']:.6f}"
                )

            if int(step) % max(1, int(snapshot_every)) == 0 or int(step) == int(steps) - 1:
                snaps.append(
                    {
                        "step": int(step),
                        "thetas": thetas.detach().cpu().tolist(),
                        "exact_energy": exact_e.detach().cpu().tolist(),
                    }
                )

    return TrainTrace(rows=rows, snapshots=snaps)


def _flatten_ma_layer(
    *,
    gamma_edges: np.ndarray,
    gamma_fields: np.ndarray,
    beta_mixers: np.ndarray,
) -> np.ndarray:
    return np.concatenate([gamma_edges.reshape(-1), gamma_fields.reshape(-1), beta_mixers.reshape(-1)], axis=0)


def _build_qaoa_relax_ma_theta(
    *,
    tied_params: Any,
    p_layers: int,
    problem: IsingProblem,
    pnp: Any,
) -> Any:
    n_qubits = int(problem.n_qubits)
    n_edges = int(len(problem.edges))
    n_fields = int(len(problem.fields))

    if pnp is torch:
        gammas = tied_params[: int(p_layers)]
        betas = tied_params[int(p_layers) : int(2 * p_layers)]

        layer_blocks = []
        for l in range(int(p_layers)):
            g = gammas[l]
            b = betas[l]

            edge_angles = (
                torch.stack([g * float(j) for (_u, _v, j) in problem.edges], dim=0)
                if n_edges > 0
                else tied_params.new_empty((0,))
            )
            field_angles = (
                torch.stack([g * float(h) for (_q, h) in problem.fields], dim=0)
                if n_fields > 0
                else tied_params.new_empty((0,))
            )
            mixer_angles = (
                torch.stack([b for _ in range(n_qubits)], dim=0)
                if n_qubits > 0
                else tied_params.new_empty((0,))
            )
            block = torch.cat(
                [
                    edge_angles.reshape(n_edges),
                    field_angles.reshape(n_fields),
                    mixer_angles.reshape(n_qubits),
                ],
                dim=0,
            )
            layer_blocks.append(block)

        return torch.cat(layer_blocks, dim=0) if len(layer_blocks) > 0 else tied_params.new_empty((0,))

    gammas = tied_params[: int(p_layers)]
    betas = tied_params[int(p_layers) : int(2 * p_layers)]

    layer_blocks = []
    for l in range(int(p_layers)):
        g = gammas[l]
        b = betas[l]
        edge_angles = [g * float(j) for (_u, _v, j) in problem.edges]
        field_angles = [g * float(h) for (_q, h) in problem.fields]
        mixer_angles = [b for _ in range(n_qubits)]
        block = pnp.concatenate([
            pnp.array(edge_angles, dtype=pnp.float64).reshape(n_edges),
            pnp.array(field_angles, dtype=pnp.float64).reshape(n_fields),
            pnp.array(mixer_angles, dtype=pnp.float64).reshape(n_qubits),
        ])
        layer_blocks.append(block)

    return pnp.concatenate(layer_blocks)


def _train_qaoa_relax_from_tqa(
    *,
    exact_qnode: Any,
    exact_lib: Any,
    problem: IsingProblem,
    p_layers: int,
    seed: int,
    steps: int,
    lr: float,
    tqa_dt_min: float,
    tqa_dt_max: float,
    log_every: int,
    qnode_device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    dt = float(rng.uniform(float(tqa_dt_min), float(tqa_dt_max)))

    p = int(p_layers)
    gammas0 = np.asarray([((l + 1) / p) * dt for l in range(p)], dtype=np.float64)
    betas0 = np.asarray([-(1.0 - ((l + 1) / p)) * dt for l in range(p)], dtype=np.float64)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros((2 * p,), dtype=np.float64)
    v = np.zeros((2 * p,), dtype=np.float64)
    trace_rows: List[Dict[str, Any]] = []

    if exact_lib is torch:
        x_np = np.asarray(np.concatenate([gammas0, betas0]), dtype=np.float64)
        for step in range(int(steps)):
            x_t = torch.as_tensor(x_np, dtype=torch.float64, device=qnode_device).clone().detach().requires_grad_(True)
            theta_full = _build_qaoa_relax_ma_theta(tied_params=x_t, p_layers=p, problem=problem, pnp=torch)
            loss_t = exact_qnode(theta_full)
            if not torch.is_tensor(loss_t):
                loss_t = torch.as_tensor(loss_t, dtype=x_t.dtype, device=x_t.device)
            loss_scalar = loss_t.reshape(-1).sum()
            g_t = torch.autograd.grad(loss_scalar, x_t, retain_graph=False, create_graph=False)[0]
            loss = float(loss_scalar.detach().cpu().item())
            g = g_t.detach().cpu().numpy().astype(np.float64, copy=False)

            t = step + 1
            m = np.add(np.multiply(m, beta1), np.multiply(g, (1.0 - beta1))).astype(np.float64)
            v = np.add(np.multiply(v, beta2), np.multiply(np.multiply(g, g), (1.0 - beta2))).astype(np.float64)
            m_hat = m / (1.0 - (beta1**t))
            v_hat = v / (1.0 - (beta2**t))
            x_np = x_np - float(lr) * m_hat / (np.sqrt(v_hat) + eps)

            row = {
                "step": int(step),
                "loss": float(loss),
                "grad_norm": float(np.linalg.norm(g)),
            }
            trace_rows.append(row)
            if int(step) % max(1, int(log_every)) == 0:
                print(f"[QAOA-Relax pretrain] step={step:04d} loss={row['loss']:.6f} grad_norm={row['grad_norm']:.6f}")

        x_final = torch.as_tensor(x_np, dtype=torch.float64, device=qnode_device)
        theta_relax = _build_qaoa_relax_ma_theta(tied_params=x_final, p_layers=p, problem=problem, pnp=torch)
        theta_relax_np = _thetas_to_numpy(theta_relax)
    else:
        import autograd as _autograd

        def f_tied(xx):
            theta_full = _build_qaoa_relax_ma_theta(tied_params=xx, p_layers=p, problem=problem, pnp=exact_lib)
            return exact_qnode(theta_full)

        value_and_grad_factory: Any = _autograd.value_and_grad
        value_and_grad_fn = value_and_grad_factory(f_tied)

        x = exact_lib.array(np.concatenate([gammas0, betas0]), requires_grad=True)
        x_np = np.asarray(x, dtype=np.float64)
        for step in range(int(steps)):
            x_ad = exact_lib.array(x_np, requires_grad=True)
            loss_val, g_val = value_and_grad_fn(x_ad)
            loss = float(loss_val)
            g = np.asarray(g_val, dtype=np.float64)

            t = step + 1
            m = np.add(np.multiply(m, beta1), np.multiply(g, (1.0 - beta1))).astype(np.float64)
            v = np.add(np.multiply(v, beta2), np.multiply(np.multiply(g, g), (1.0 - beta2))).astype(np.float64)
            m_hat = m / (1.0 - (beta1**t))
            v_hat = v / (1.0 - (beta2**t))
            x_np = x_np - float(lr) * m_hat / (np.sqrt(v_hat) + eps)

            row = {
                "step": int(step),
                "loss": float(loss),
                "grad_norm": float(np.linalg.norm(g)),
            }
            trace_rows.append(row)
            if int(step) % max(1, int(log_every)) == 0:
                print(f"[QAOA-Relax pretrain] step={step:04d} loss={row['loss']:.6f} grad_norm={row['grad_norm']:.6f}")

        x_final = exact_lib.array(x_np, requires_grad=False)
        theta_relax_np = _thetas_to_numpy(
            _build_qaoa_relax_ma_theta(tied_params=x_final, p_layers=p, problem=problem, pnp=exact_lib)
        )
    meta = {
        "tqa_dt": float(dt),
        "steps": int(steps),
        "lr": float(lr),
        "trace": trace_rows,
        "tied_gammas": [float(vv) for vv in x_np[:p]],
        "tied_betas": [float(vv) for vv in x_np[p:]],
    }
    return theta_relax_np, meta


def _build_init_pool(
    *,
    problem: IsingProblem,
    p_layers: int,
    seed: int,
    random_count: int,
    const_values: Sequence[float],
    qaoa_relax_count: int,
    random_gamma_low: float,
    random_gamma_high: float,
    random_beta_low: float,
    random_beta_high: float,
    exact_qnode: Any,
    exact_lib: Any,
    qnode_device: str,
    qaoa_relax_steps: int,
    qaoa_relax_lr: float,
    tqa_dt_min: float,
    tqa_dt_max: float,
    log_every: int,
    device: str,
) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    n_edges = int(len(problem.edges))
    n_fields = int(len(problem.fields))
    n_qubits = int(problem.n_qubits)
    p = int(p_layers)

    init_rows: List[np.ndarray] = []
    labels: List[str] = []
    relax_meta_list: List[Dict[str, Any]] = []

    for i in range(int(random_count)):
        flat_blocks = []
        for _ in range(p):
            g_edges = rng.uniform(float(random_gamma_low), float(random_gamma_high), size=(n_edges,)).astype(np.float64)
            g_fields = rng.uniform(float(random_gamma_low), float(random_gamma_high), size=(n_fields,)).astype(np.float64)
            b_mixers = rng.uniform(float(random_beta_low), float(random_beta_high), size=(n_qubits,)).astype(np.float64)
            flat_blocks.append(_flatten_ma_layer(gamma_edges=g_edges, gamma_fields=g_fields, beta_mixers=b_mixers))
        init_rows.append(np.concatenate(flat_blocks, axis=0))
        labels.append(f"random_{i + 1}")

    for i, c in enumerate(list(const_values)):
        flat_blocks = []
        for _ in range(p):
            g_edges = np.full((n_edges,), float(c), dtype=np.float64)
            g_fields = np.full((n_fields,), float(c), dtype=np.float64)
            b_mixers = np.full((n_qubits,), -float(c), dtype=np.float64)
            flat_blocks.append(_flatten_ma_layer(gamma_edges=g_edges, gamma_fields=g_fields, beta_mixers=b_mixers))
        init_rows.append(np.concatenate(flat_blocks, axis=0))
        labels.append(f"const_{i + 1}_{float(c):.4f}")

    for i in range(int(qaoa_relax_count)):
        theta_relax, relax_meta = _train_qaoa_relax_from_tqa(
            exact_qnode=exact_qnode,
            exact_lib=exact_lib,
            problem=problem,
            p_layers=p,
            seed=int(seed) + 1000 + i,
            steps=int(qaoa_relax_steps),
            lr=float(qaoa_relax_lr),
            tqa_dt_min=float(tqa_dt_min),
            tqa_dt_max=float(tqa_dt_max),
            log_every=int(log_every),
            qnode_device=qnode_device,
        )
        init_rows.append(theta_relax)
        labels.append(f"qaoa_relax_{i + 1}")
        relax_meta_list.append(relax_meta)

    theta_np = np.asarray(init_rows, dtype=np.float64)
    theta_t = torch.as_tensor(theta_np, dtype=torch.float64, device=device)
    meta = {
        "n_random": int(random_count),
        "n_constant": int(len(const_values)),
        "n_qaoa_relax": int(qaoa_relax_count),
        "qaoa_relax": relax_meta_list,
    }
    return theta_t, labels, meta


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_lwpp_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Preset overrides for the LWPP compile, with the effective dtype resolved."""
    overrides: Dict[str, Any] = {"max_weight": int(args.lwpp_max_weight)}
    if str(args.lwpp_dtype).strip():
        overrides["dtype"] = str(args.lwpp_dtype).strip()
    overrides["dtype"] = resolve_preset(str(args.lwpp_preset), overrides=overrides).dtype
    return overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MA-QAOA experiment: compare direct exact-only training vs LWPP pretrain + exact finetuning "
            "with batched parameter instances."
        )
    )
    p.add_argument("--problem-json", type=str, required=True)
    p.add_argument("--p-layers", type=int, default=3)

    p.add_argument("--batch-size", type=int, default=10, help="Used only when --init-strategy gaussian is selected.")
    p.add_argument("--init-std", type=float, default=0.05, help="Used only when --init-strategy gaussian is selected.")
    p.add_argument("--init-strategy", type=str, default="baseline11", choices=["baseline11", "gaussian"])
    p.add_argument("--n-random-inits", type=int, default=5)
    p.add_argument("--const-values", type=str, default="0.01,0.05,0.1,0.2,0.4")
    p.add_argument("--n-qaoa-relax-inits", type=int, default=1)
    p.add_argument("--random-gamma-range", type=float, nargs=2, default=[-0.5, 0.5])
    p.add_argument("--random-beta-range", type=float, nargs=2, default=[-0.5, 0.5])
    p.add_argument("--qaoa-relax-steps", type=int, default=200)
    p.add_argument("--qaoa-relax-lr", type=float, default=0.02)
    p.add_argument("--tqa-dt-range", type=float, nargs=2, default=[0.5, 1.5])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lwpp-steps", type=int, default=200)
    p.add_argument("--exact-steps", type=int, default=300)
    p.add_argument("--exact-only-steps", type=int, default=500)

    p.add_argument("--lwpp-lr", type=float, default=0.02)
    p.add_argument("--exact-lr", type=float, default=0.02)
    p.add_argument("--exact-only-lr", type=float, default=0.02)
    p.add_argument(
        "--exact-chunk-size",
        type=int,
        default=1,
        help=(
            "Number of initializations evaluated per exact-device call "
            "(1 = one at a time; >= batch size = single call)."
        ),
    )

    p.add_argument("--snapshot-every-lwpp", type=int, default=20)
    p.add_argument("--snapshot-every-exact", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--lwpp-exact-eval-every",
        type=int,
        default=10,
        help="Evaluate the (trace-only) exact energy every N LWPP steps; the final step is always evaluated. 1 evaluates every step.",
    )

    p.add_argument("--lwpp-preset", type=str, default="gpu", choices=["cpu", "gpu", "hybrid"])
    p.add_argument("--lwpp-max-weight", type=int, default=4, help="Maximum Pauli weight kept by the LWPP surrogate (w_max).")
    p.add_argument(
        "--lwpp-dtype",
        type=str,
        default="",
        choices=["", "float16", "float32", "float64"],
        help="Override LWPP surrogate compute dtype (empty keeps the preset default; paper sweeps used float16).",
    )
    p.add_argument(
        "--theta-dtype",
        type=str,
        default="float64",
        choices=["float64"],
        help="Trainable parameter dtype. Fixed to float64.",
    )

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--exact-device", type=str, default="", help="Optional PennyLane device override.")

    p.add_argument("--sa-baseline", dest="sa_baseline", action="store_true", help="Use simulated annealing to estimate ground sample/energy baseline.")
    p.add_argument("--no-sa-baseline", dest="sa_baseline", action="store_false", help="Disable the simulated-annealing reference baseline.")
    p.set_defaults(sa_baseline=True)
    p.add_argument("--sa-restarts", type=int, default=64)
    p.add_argument("--sa-sweeps", type=int, default=300)
    p.add_argument("--sa-tmax", type=float, default=3.0)
    p.add_argument("--sa-tmin", type=float, default=0.01)

    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--run-name", type=str, default="ma_qaoa_lwpp_exact_compare")
    p.add_argument("--resume-run-dir", type=str, default="", help="Resume from an existing run directory using resume_state.pt.")
    p.add_argument("--stream-save-every", type=int, default=1, help="Write the resume checkpoint every N steps.")

    p.add_argument("--live-plot", dest="live_plot", action="store_true", help="Enable live-updating training plot.")
    p.add_argument("--no-live-plot", dest="live_plot", action="store_false", help="Disable live-updating plot.")
    p.set_defaults(live_plot=True)
    p.add_argument("--plot-refresh-every", type=int, default=5, help="Refresh live plot every N steps.")
    p.add_argument("--no-plot-window", action="store_true", help="Do not open interactive window; still saves refreshed plot image.")
    return p.parse_args()


def _cuda_reset_peak() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _cuda_peak_bytes() -> Optional[int]:
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.max_memory_allocated())
        except Exception:
            return None
    return None


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    run_root = Path(str(args.resume_run_dir)).resolve() if str(args.resume_run_dir).strip() else (Path(args.out_dir) / f"{args.run_name}_{_now_tag()}")
    run_root.mkdir(parents=True, exist_ok=True)
    meta_path = run_root / "run_meta.json"

    meta: Dict[str, Any] = {
        "status": "running",
        "created_at": _now_tag(),
        "run_dir": str(run_root),
        "command": " ".join(sys.argv),
        "problem_json": str(Path(args.problem_json).resolve()),
        "p_layers": int(args.p_layers),
        "seed": int(args.seed),
        "save_format": {
            "trace": "csv",
            "per_seed_trace": "csv",
        },
        "sa_baseline": {
            "enabled": bool(args.sa_baseline),
            "restarts": int(args.sa_restarts),
            "sweeps": int(args.sa_sweeps),
            "t_max": float(args.sa_tmax),
            "t_min": float(args.sa_tmin),
        },
        "init": {
            "strategy": str(args.init_strategy),
            "batch_size": int(args.batch_size),
            "init_std": float(args.init_std),
            "n_random_inits": int(args.n_random_inits),
            "const_values": [float(v) for v in str(args.const_values).split(",") if v.strip() != ""],
            "n_qaoa_relax_inits": int(args.n_qaoa_relax_inits),
            "qaoa_relax_steps": int(args.qaoa_relax_steps),
            "qaoa_relax_lr": float(args.qaoa_relax_lr),
            "tqa_dt_range": [float(args.tqa_dt_range[0]), float(args.tqa_dt_range[1])],
        },
        "lwpp": {
            "preset": str(args.lwpp_preset),
            "dtype": str(_resolve_lwpp_overrides(args)["dtype"]),
            "max_weight": int(args.lwpp_max_weight),
            "steps": int(args.lwpp_steps),
            "lr": float(args.lwpp_lr),
        },
        "finetune_exact": {
            "steps": int(args.exact_steps),
            "lr": float(args.exact_lr),
        },
        "exact_only": {
            "steps": int(args.exact_only_steps),
            "lr": float(args.exact_only_lr),
        },
        "runtime": {
            "device_arg": str(args.device),
            "theta_dtype": str(args.theta_dtype),
            "exact_device_arg": str(args.exact_device),
            "live_plot": bool(args.live_plot),
            "plot_refresh_every": int(args.plot_refresh_every),
            "no_plot_window": bool(args.no_plot_window),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    problem = load_problem(args.problem_json)
    obs = _build_ising_observable(problem)
    circuit, n_params = _build_ma_qaoa_circuit(problem, p_layers=int(args.p_layers))

    sa_info: Optional[Dict[str, Any]] = None
    reference_energy: Optional[float] = None
    if bool(args.sa_baseline):
        sa_energy, sa_spin, sa_bits = _simulated_annealing_ground_sample(
            problem=problem,
            seed=int(args.seed) + 7919,
            restarts=int(args.sa_restarts),
            sweeps=int(args.sa_sweeps),
            t_max=float(args.sa_tmax),
            t_min=float(args.sa_tmin),
        )
        reference_energy = float(sa_energy)
        sa_info = {
            "energy": float(sa_energy),
            "spin_sample": sa_spin,
            "bit_sample": sa_bits,
            "restarts": int(args.sa_restarts),
            "sweeps": int(args.sa_sweeps),
            "t_max": float(args.sa_tmax),
            "t_min": float(args.sa_tmin),
        }
        print(f"[SA baseline] energy={reference_energy:.6f} restarts={int(args.sa_restarts)} sweeps={int(args.sa_sweeps)}")

    device = _choose_torch_device(str(args.device))

    exact_qnode, exact_dev, exact_diff, exact_lib = _build_exact_qnode(circuit, problem, str(args.exact_device).strip())
    qnode_device = "cuda" if "gpu" in str(exact_dev).lower() else "cpu"
    meta["runtime"]["device"] = str(device)
    meta["runtime"]["exact_device"] = str(exact_dev)
    meta["runtime"]["exact_diff_method"] = str(exact_diff)
    meta["runtime"]["exact_interface"] = "torch" if exact_lib is torch else "autograd"
    meta["runtime"]["exact_tensor_device"] = str(qnode_device)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    resume_ckpt_path = run_root / "resume_state.pt"
    resume_data: Dict[str, Any] = {}
    if resume_ckpt_path.exists():
        resume_data = dict(torch.load(resume_ckpt_path, map_location="cpu"))
        print(f"[resume] loaded checkpoint: {resume_ckpt_path}")

    init_labels: List[str] = []
    init_meta: Dict[str, Any] = {}
    if "theta_init" in resume_data:
        theta_init = torch.as_tensor(resume_data["theta_init"], dtype=torch.float64, device=device)
        init_labels = list(resume_data.get("init_labels", []))
        init_meta = dict(resume_data.get("init_meta", {}))
        print(f"[resume] reused theta_init with batch={int(theta_init.shape[0])}")
    else:
        if str(args.init_strategy) == "gaussian":
            theta_init = torch.randn(
                int(args.batch_size),
                int(n_params),
                dtype=torch.float64,
                device=device,
            ) * float(args.init_std)
            init_labels = [f"gaussian_{i + 1}" for i in range(int(args.batch_size))]
            init_meta = {
                "strategy": "gaussian",
                "init_std": float(args.init_std),
            }
        else:
            const_values = [float(x.strip()) for x in str(args.const_values).split(",") if x.strip()]
            if int(args.n_random_inits) < 0 or int(args.n_qaoa_relax_inits) < 0:
                raise ValueError("n-random-inits and n-qaoa-relax-inits must be >= 0")
            if len(const_values) == 0:
                raise ValueError("const-values must contain at least one value for baseline11 strategy")

            theta_init, init_labels, init_meta = _build_init_pool(
                problem=problem,
                p_layers=int(args.p_layers),
                seed=int(args.seed),
                random_count=int(args.n_random_inits),
                const_values=const_values,
                qaoa_relax_count=int(args.n_qaoa_relax_inits),
                random_gamma_low=float(args.random_gamma_range[0]),
                random_gamma_high=float(args.random_gamma_range[1]),
                random_beta_low=float(args.random_beta_range[0]),
                random_beta_high=float(args.random_beta_range[1]),
                exact_qnode=exact_qnode,
                exact_lib=exact_lib,
                qnode_device=qnode_device,
                qaoa_relax_steps=int(args.qaoa_relax_steps),
                qaoa_relax_lr=float(args.qaoa_relax_lr),
                tqa_dt_min=float(args.tqa_dt_range[0]),
                tqa_dt_max=float(args.tqa_dt_range[1]),
                log_every=int(args.log_every),
                device=device,
            )
            init_meta["strategy"] = "baseline11"

    actual_batch_size = int(theta_init.shape[0])

    live_plot: Optional[LiveTrainingPlotter] = None
    if bool(args.live_plot):
        try:
            live_plot = LiveTrainingPlotter(
                batch_size=actual_batch_size,
                init_labels=init_labels,
                title=(
                    f"MA-QAOA trajectories (n={int(problem.n_qubits)}, "
                    f"p={int(args.p_layers)}, batch={actual_batch_size})"
                ),
                reference_energy=reference_energy,
                refresh_every=int(args.plot_refresh_every),
                save_path=run_root / "training_live.png",
                show_window=not bool(args.no_plot_window),
            )
        except Exception as exc:
            print(f"[warn] live plot disabled due to plotting backend error: {exc}")
            live_plot = None

    print(f"[init] strategy={init_meta.get('strategy', 'unknown')} batch={actual_batch_size}")
    print(f"[init] labels={init_labels}")
    meta["init"]["actual_batch_size"] = int(actual_batch_size)
    meta["init"]["labels"] = list(init_labels)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lwpp_csv = run_root / "lwpp_pre_trace.csv"
    fine_csv = run_root / "exact_finetune_trace.csv"
    direct_csv = run_root / "exact_only_trace.csv"
    lwpp_seed_csv = run_root / "lwpp_pre_trace_per_seed.csv"
    fine_seed_csv = run_root / "exact_finetune_trace_per_seed.csv"
    direct_seed_csv = run_root / "exact_only_trace_per_seed.csv"
    progress_json = run_root / "progress.json"

    lwpp_existing = _read_csv_rows(lwpp_csv)
    fine_existing = _read_csv_rows(fine_csv)
    direct_existing = _read_csv_rows(direct_csv)
    lwpp_start = int(len(lwpp_existing))
    fine_start = int(len(fine_existing))
    direct_start = int(len(direct_existing))

    def _save_resume_state(*, stage: str, step: int, thetas: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        state = {
            "stage": str(stage),
            "step": int(step),
            "theta_init": theta_init.detach().cpu(),
            "init_labels": list(init_labels),
            "init_meta": dict(init_meta),
            "lwpp_completed": int(len(_read_csv_rows(lwpp_csv))),
            "fine_completed": int(len(_read_csv_rows(fine_csv))),
            "direct_completed": int(len(_read_csv_rows(direct_csv))),
            "thetas": thetas.detach().cpu(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(state, resume_ckpt_path)
        progress_json.write_text(
            json.dumps(
                {
                    "stage": str(stage),
                    "step": int(step),
                    "lwpp_completed": int(state["lwpp_completed"]),
                    "fine_completed": int(state["fine_completed"]),
                    "direct_completed": int(state["direct_completed"]),
                    "updated_at": _now_tag(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _on_step(
        *,
        stage: str,
        step: int,
        row: Dict[str, Any],
        thetas: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        s = str(stage)
        ext = dict(extras) if extras is not None else {}

        def _write_seed_rows(*, out_csv: Path, payload: Dict[str, Any]) -> None:
            exact_vals = payload.get("exact_energy", [])
            lwpp_vals = payload.get("lwpp_energy", [])
            n = max(len(exact_vals), len(lwpp_vals), len(init_labels))
            for i in range(int(n)):
                r = {
                    "stage": s,
                    "step": int(step),
                    "init_index": int(i),
                    "init_label": str(init_labels[i]) if i < len(init_labels) else f"init_{i + 1}",
                    "exact_energy": float(exact_vals[i]) if i < len(exact_vals) else None,
                    "lwpp_energy": float(lwpp_vals[i]) if i < len(lwpp_vals) else None,
                }
                _append_csv_row(
                    out_csv,
                    r,
                    fieldnames=["stage", "step", "init_index", "init_label", "exact_energy", "lwpp_energy"],
                )

        if s == "lwpp":
            _append_csv_row(
                lwpp_csv,
                row,
                fieldnames=["step", "lwpp_energy_mean", "lwpp_energy_min", "exact_energy_mean", "exact_energy_min"],
            )
            _write_seed_rows(out_csv=lwpp_seed_csv, payload=ext)
            if int(step) == int(args.lwpp_steps) - 1:
                # LWPP-final thetas are the distillation input.
                (run_root / "lwpp_final_thetas.json").write_text(
                    json.dumps(
                        {
                            "step": int(step),
                            "init_labels": list(init_labels),
                            "thetas": thetas.detach().cpu().tolist(),
                        }
                    ),
                    encoding="utf-8",
                )
        elif s == "exact_finetune":
            _append_csv_row(fine_csv, row, fieldnames=["step", "exact_energy_mean", "exact_energy_min"])
            _write_seed_rows(out_csv=fine_seed_csv, payload=ext)
        elif s == "exact_only":
            _append_csv_row(direct_csv, row, fieldnames=["step", "exact_energy_mean", "exact_energy_min"])
            _write_seed_rows(out_csv=direct_seed_csv, payload=ext)

        if int(step) % max(1, int(args.stream_save_every)) == 0:
            _save_resume_state(stage=s, step=int(step), thetas=thetas, optimizer=optimizer)

    # Branch A: LWPP warmup then exact finetuning.
    # Enable padopauli's compile profiler so the propagated / zero-filtered term
    # counts and compile timing are recorded (see cost_metrics below).
    os.environ.setdefault("PPS_COMPILE_PROFILE", "1")
    _cuda_reset_peak()
    _t_compile0 = time.perf_counter()
    lwpp_program = compile_program(
        circuit=circuit,
        observables=[obs],
        preset=str(args.lwpp_preset),
        preset_overrides=_resolve_lwpp_overrides(args),
    )
    lwpp_compile_wall_s = float(time.perf_counter() - _t_compile0)
    lwpp_compile_vram_bytes = _cuda_peak_bytes()
    _lwpp_summary = dict((getattr(lwpp_program, "compile_stats", {}) or {}).get("summary", {}) or {})
    _terms_before_zf = _lwpp_summary.get("propagated_terms")
    _terms_before_zf = int(_terms_before_zf) if _terms_before_zf is not None else None
    try:
        _terms_after_zf = int(lwpp_program.psum_union.x_mask.shape[0])
    except Exception:
        _fin = _lwpp_summary.get("final_terms")
        _terms_after_zf = int(_fin) if _fin is not None else None

    theta_lwpp = torch.nn.Parameter(theta_init.detach().clone())
    opt_lwpp = torch.optim.Adam([theta_lwpp], lr=float(args.lwpp_lr))
    if resume_data.get("stage") == "lwpp" and "thetas" in resume_data:
        theta_lwpp.data = torch.as_tensor(resume_data["thetas"], dtype=theta_lwpp.dtype, device=theta_lwpp.device)
        if "optimizer_state" in resume_data:
            opt_lwpp.load_state_dict(dict(resume_data["optimizer_state"]))
    _cuda_reset_peak()
    _t_lwpp0 = time.perf_counter()
    lwpp_trace = _train_lwpp_phase(
        thetas=theta_lwpp,
        optimizer=opt_lwpp,
        lwpp_program=lwpp_program,
        exact_qnode=exact_qnode,
        exact_lib=exact_lib,
        exact_dev=exact_dev,
        steps=int(args.lwpp_steps),
        snapshot_every=int(args.snapshot_every_lwpp),
        log_every=int(args.log_every),
        exact_eval_every=int(args.lwpp_exact_eval_every),
        start_step=lwpp_start,
        initial_rows=lwpp_existing,
        step_callback=_on_step,
        live_plot=live_plot,
    )
    lwpp_train_wall_s = float(time.perf_counter() - _t_lwpp0)
    lwpp_train_vram_bytes = _cuda_peak_bytes()

    theta_fine = torch.nn.Parameter(theta_lwpp.detach().clone())
    opt_fine = torch.optim.Adam([theta_fine], lr=float(args.exact_lr))
    if resume_data.get("stage") == "exact_finetune" and "thetas" in resume_data:
        theta_fine.data = torch.as_tensor(resume_data["thetas"], dtype=theta_fine.dtype, device=theta_fine.device)
        if "optimizer_state" in resume_data:
            opt_fine.load_state_dict(dict(resume_data["optimizer_state"]))
    _cuda_reset_peak()
    _t_fine0 = time.perf_counter()
    fine_trace = _train_exact_phase(
        thetas=theta_fine,
        optimizer=opt_fine,
        exact_qnode=exact_qnode,
        exact_lib=exact_lib,
        exact_dev=exact_dev,
        steps=int(args.exact_steps),
        snapshot_every=int(args.snapshot_every_exact),
        log_every=int(args.log_every),
        prefix="LWPP+Exact",
        start_step=fine_start,
        initial_rows=fine_existing,
        step_callback=_on_step,
        live_plot=live_plot,
        live_stage="exact_finetune",
        exact_chunk_size=int(args.exact_chunk_size),
    )
    fine_wall_s = float(time.perf_counter() - _t_fine0)
    fine_vram_bytes = _cuda_peak_bytes()

    # Branch B: direct exact-only training from same initial batch.
    theta_direct = torch.nn.Parameter(theta_init.detach().clone())
    opt_direct = torch.optim.Adam([theta_direct], lr=float(args.exact_only_lr))
    if resume_data.get("stage") == "exact_only" and "thetas" in resume_data:
        theta_direct.data = torch.as_tensor(resume_data["thetas"], dtype=theta_direct.dtype, device=theta_direct.device)
        if "optimizer_state" in resume_data:
            opt_direct.load_state_dict(dict(resume_data["optimizer_state"]))
    _cuda_reset_peak()
    _t_direct0 = time.perf_counter()
    direct_trace = _train_exact_phase(
        thetas=theta_direct,
        optimizer=opt_direct,
        exact_qnode=exact_qnode,
        exact_lib=exact_lib,
        exact_dev=exact_dev,
        steps=int(args.exact_only_steps),
        snapshot_every=int(args.snapshot_every_exact),
        log_every=int(args.log_every),
        prefix="ExactOnly",
        start_step=direct_start,
        initial_rows=direct_existing,
        step_callback=_on_step,
        live_plot=live_plot,
        live_stage="exact_only",
        exact_chunk_size=int(args.exact_chunk_size),
    )
    direct_wall_s = float(time.perf_counter() - _t_direct0)
    direct_vram_bytes = _cuda_peak_bytes()

    with torch.no_grad():
        lwpp_final_exact = _exact_energy_batch(exact_qnode, exact_lib, theta_fine, exact_dev).detach().cpu().tolist()
        direct_final_exact = _exact_energy_batch(exact_qnode, exact_lib, theta_direct, exact_dev).detach().cpu().tolist()

    result = {
        "config": {
            "problem_json": str(Path(args.problem_json).resolve()),
            "p_layers": int(args.p_layers),
            "n_qubits": int(problem.n_qubits),
            "n_edges": int(len(problem.edges)),
            "n_fields": int(len(problem.fields)),
            "batch_size": int(actual_batch_size),
            "n_params": int(n_params),
            "seed": int(args.seed),
            "device": str(device),
            "exact_device": str(exact_dev),
            "exact_diff_method": str(exact_diff),
            "sa_baseline": sa_info,
            "init": {
                "strategy": str(args.init_strategy),
                "labels": list(init_labels),
                "details": init_meta,
            },
            "lwpp": {
                "preset": str(args.lwpp_preset),
                "max_weight": int(args.lwpp_max_weight),
                "steps": int(args.lwpp_steps),
                "lr": float(args.lwpp_lr),
            },
            "finetune_exact": {
                "steps": int(args.exact_steps),
                "lr": float(args.exact_lr),
            },
            "exact_only": {
                "steps": int(args.exact_only_steps),
                "lr": float(args.exact_only_lr),
            },
        },
        "comparison": {
            "lwpp_plus_exact_final_exact_energy": lwpp_final_exact,
            "exact_only_final_exact_energy": direct_final_exact,
            "lwpp_plus_exact_mean": float(np.mean(np.asarray(lwpp_final_exact, dtype=np.float64))),
            "exact_only_mean": float(np.mean(np.asarray(direct_final_exact, dtype=np.float64))),
            "lwpp_plus_exact_best": float(np.min(np.asarray(lwpp_final_exact, dtype=np.float64))),
            "exact_only_best": float(np.min(np.asarray(direct_final_exact, dtype=np.float64))),
        },
        "lwpp_initialized_exact_finetuning": {
            "pre_trace": lwpp_trace.rows,
            "fine_trace": fine_trace.rows,
            "pre_snapshots": lwpp_trace.snapshots,
            "fine_snapshots": fine_trace.snapshots,
            "final_thetas": theta_fine.detach().cpu().tolist(),
        },
        "direct_exact_optimization": {
            "trace": direct_trace.rows,
            "snapshots": direct_trace.snapshots,
            "final_thetas": theta_direct.detach().cpu().tolist(),
        },
    }

    # Computation-cost metrics (LWPP zero-filtering term counts, per-phase wall
    # time and peak VRAM). Written to a dedicated cost_metrics.json and embedded
    # in results.json.
    cost_metrics = {
        "schema": "safe_maqaoa.cost_metrics/v1",
        "recorded_at": _now_tag(),
        "resumed_run": bool(resume_data.get("stage")),
        "run": Path(args.problem_json).stem,
        "system": {
            "n_qubits": int(problem.n_qubits),
            "n_edges": int(len(problem.edges)),
            "n_params": int(n_params),
            "p_layers": int(args.p_layers),
            "lwpp_max_weight": int(args.lwpp_max_weight),
            "lwpp_preset": str(args.lwpp_preset),
            "lwpp_dtype": str(args.lwpp_dtype),
            "exact_device": str(exact_dev),
            "exact_chunk_size": int(args.exact_chunk_size),
        },
        "lwpp": {
            "terms_before_zero_filtering": _terms_before_zf,
            "terms_after_zero_filtering": _terms_after_zf,
            "keep_fraction": (
                float(_terms_after_zf) / float(_terms_before_zf)
                if (_terms_before_zf and _terms_after_zf is not None)
                else None
            ),
            "steps": int(args.lwpp_steps),
            "compile_wall_s": lwpp_compile_wall_s,
            "compile_propagate_s": _lwpp_summary.get("propagate_total_s"),
            "compile_zero_filter_s": _lwpp_summary.get("zero_filter_total_s"),
            "n_propagate_steps": _lwpp_summary.get("n_propagate_steps"),
            "train_wall_s": lwpp_train_wall_s,
            "compile_peak_vram_bytes": lwpp_compile_vram_bytes,
            "train_peak_vram_bytes": lwpp_train_vram_bytes,
        },
        "exact_finetune": {
            "steps": int(args.exact_steps),
            "chunk_size": int(args.exact_chunk_size),
            "wall_s": fine_wall_s,
            "peak_vram_bytes": fine_vram_bytes,
        },
        "exact_only": {
            "steps": int(args.exact_only_steps),
            "chunk_size": int(args.exact_chunk_size),
            "wall_s": direct_wall_s,
            "peak_vram_bytes": direct_vram_bytes,
        },
        "totals": {
            "train_wall_s": float(lwpp_train_wall_s + fine_wall_s + direct_wall_s),
            "wall_including_compile_s": float(
                lwpp_compile_wall_s + lwpp_train_wall_s + fine_wall_s + direct_wall_s
            ),
        },
        "notes": {
            "terms": "Pauli propagation term counts before/after zero-filtering.",
            "vram": "torch.cuda peak (padopauli/torch allocator only). PennyLane lightning statevector uses a separate allocator NOT captured here, so exact-phase VRAM under-reports GPU statevector memory.",
            "timing": "Wall time for this process invocation; partial if resumed_run is true, and inflated under concurrent (5-way) sweep execution. Isolate a run for clean per-system timing.",
        },
    }
    (run_root / "cost_metrics.json").write_text(json.dumps(cost_metrics, indent=2), encoding="utf-8")
    result["cost_metrics"] = cost_metrics

    (run_root / "problem_copy.json").write_text(Path(args.problem_json).read_text(encoding="utf-8"), encoding="utf-8")
    (run_root / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    meta["status"] = "completed"
    meta["completed_at"] = _now_tag()
    meta["results_file"] = str((run_root / "results.json").resolve())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    torch.save(
        {
            "theta_init": theta_init.detach().cpu(),
            "theta_init_labels": init_labels,
            "theta_lwpp_end": theta_lwpp.detach().cpu(),
            "theta_fine_end": theta_fine.detach().cpu(),
            "theta_direct_end": theta_direct.detach().cpu(),
        },
        run_root / "theta_checkpoints.pt",
    )

    if live_plot is not None:
        live_plot.close()

    print(f"saved run directory: {run_root}")
    print(
        "comparison: "
        f"lwpp+exact mean={result['comparison']['lwpp_plus_exact_mean']:.6f}, "
        f"exact-only mean={result['comparison']['exact_only_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
