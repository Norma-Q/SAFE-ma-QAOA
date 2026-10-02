from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safe_maqaoa.problem_factory import build_grid2d_problem, build_sk_problem
def _is_connected(n_qubits: int, edges) -> bool:
    n = int(n_qubits)
    if n <= 1:
        return True
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                stack.append(w)
    return all(seen)


def _make_erdos_renyi_graph(
    *, n_qubits: int, edge_prob: float, seed: int, ensure_connected: bool, max_tries: int
):
    n = int(n_qubits)
    p = float(edge_prob)
    if not (0.0 < p <= 1.0):
        raise ValueError("edge_prob must be in (0,1].")
    rng = np.random.default_rng(int(seed))
    for _ in range(int(max_tries)):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if float(rng.random()) < p:
                    edges.append((i, j))
        if len(edges) == 0:
            continue
        if (not ensure_connected) or _is_connected(n, edges):
            return sorted(edges)
    raise RuntimeError("Failed to generate Erdos-Renyi graph with requested constraints.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the QAOA problem-family schematic.")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/output/fig2.png",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--sk-qubits", type=int, default=12)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--maxcut-qubits", type=int, default=12)
    parser.add_argument("--maxcut-edge-prob", type=float, default=0.3)
    parser.add_argument("--symlog-linthresh", type=float, default=0.02)
    parser.add_argument("--color-quantile", type=float, default=0.72)
    return parser.parse_args()


def draw_weighted_edges(
    ax: plt.Axes,
    positions: dict[int, tuple[float, float]],
    edges: list[tuple[int, int]],
    weights: np.ndarray,
    cmap,
    norm,
    *,
    width: float,
    alpha_base: float,
    alpha_scale: float,
    dashed: bool = False,
) -> None:
    if len(weights) == 0:
        return
    denom = max(float(np.max(np.abs(weights))), 1e-12)
    linestyle = "--" if dashed else "-"
    for (u, v), w in zip(edges, weights):
        x0, y0 = positions[int(u)]
        x1, y1 = positions[int(v)]
        alpha = float(np.clip(alpha_base + alpha_scale * abs(float(w)) / denom, 0.42, 0.92))
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=cmap(norm(float(w))),
            linewidth=width,
            linestyle=linestyle,
            alpha=alpha,
            zorder=1,
        )


def draw_nodes(
    ax: plt.Axes,
    positions: dict[int, tuple[float, float]],
    *,
    halo_values: np.ndarray | None = None,
    cmap=None,
    norm=None,
    node_size: float = 380,
    font_size: float = 12,
) -> None:
    for node, (x, y) in positions.items():
        if halo_values is not None:
            value = float(halo_values[int(node)])
            halo_size = 720 + 260 * min(abs(value) / 0.2, 1.5)
            ax.scatter([x], [y], s=halo_size, color=cmap(norm(value)), alpha=0.22, linewidths=0, zorder=2)
        ax.scatter([x], [y], s=node_size, color="#F8F7F2", edgecolors="#111111", linewidths=2.2, zorder=3)
        ax.text(x, y, f"{int(node) + 1}", ha="center", va="center", fontsize=font_size, zorder=4)


def draw_sk_panel(ax: plt.Axes, problem, cmap, norm) -> None:
    n = int(problem.n_qubits)
    radius = 1.22
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + math.radians(15.0)
    positions = {i: (radius * np.cos(theta), radius * np.sin(theta)) for i, theta in enumerate(angles)}
    edges = [(u, v) for u, v, _ in problem.edges]
    weights = np.asarray([w for _, _, w in problem.edges], dtype=np.float64)
    draw_weighted_edges(ax, positions, edges, weights, cmap, norm, width=1.35, alpha_base=0.18, alpha_scale=0.38)
    draw_nodes(ax, positions, node_size=360, font_size=12)
    ax.text(
        0.0,
        -1.62,
        r"$J_{ij} \sim \mathcal{N}(0,(1/\sqrt{N})^2)$",
        ha="center",
        va="center",
        fontsize=13,
    )
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.82, 1.62)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_grid_panel(ax: plt.Axes, problem, cmap, norm) -> None:
    rows = int(problem.meta["rows"])
    cols = int(problem.meta["cols"])
    spacing = 1.18
    positions = {
        node: (spacing * float(node % cols), spacing * float(rows - 1 - node // cols))
        for node in range(problem.n_qubits)
    }

    nn_set: set[tuple[int, int]] = set()
    nnn_set: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                nn_set.add(tuple(sorted((u, r * cols + c + 1))))
            if r + 1 < rows:
                nn_set.add(tuple(sorted((u, (r + 1) * cols + c))))
            if r + 1 < rows and c + 1 < cols:
                nnn_set.add(tuple(sorted((u, (r + 1) * cols + c + 1))))
            if r + 1 < rows and c - 1 >= 0:
                nnn_set.add(tuple(sorted((u, (r + 1) * cols + c - 1))))

    nn_edges: list[tuple[int, int]] = []
    nn_weights: list[float] = []
    nnn_edges: list[tuple[int, int]] = []
    nnn_weights: list[float] = []
    for u, v, w in problem.edges:
        pair = tuple(sorted((int(u), int(v))))
        if pair in nn_set:
            nn_edges.append(pair)
            nn_weights.append(float(w))
        elif pair in nnn_set:
            nnn_edges.append(pair)
            nnn_weights.append(float(w))

    fields = np.zeros((problem.n_qubits,), dtype=np.float64)
    for q, h in problem.fields:
        fields[int(q)] = float(h)

    draw_weighted_edges(
        ax, positions, nn_edges, np.asarray(nn_weights), cmap, norm, width=2.3, alpha_base=0.42, alpha_scale=0.30
    )
    draw_weighted_edges(
        ax,
        positions,
        nnn_edges,
        np.asarray(nnn_weights),
        cmap,
        norm,
        width=1.9,
        alpha_base=0.28,
        alpha_scale=0.34,
        dashed=True,
    )
    draw_nodes(ax, positions, halo_values=fields, cmap=cmap, norm=norm, node_size=330, font_size=12)
    ax.text(
        spacing * (cols - 1) / 2.0,
        -0.78,
        r"NN $\sim \mathcal{N}(0,1)$, NNN $\sim \mathcal{N}(0,0.3^2)$, $h_i \sim \mathcal{N}(0,0.1^2)$",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_xlim(-0.62, spacing * (cols - 1) + 0.62)
    ax.set_ylim(-1.02, spacing * (rows - 1) + 0.62)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_maxcut_panel(ax: plt.Axes, edges: list[tuple[int, int]], n_qubits: int, edge_prob: float) -> None:
    radius = 1.22
    angles = np.linspace(0.0, 2.0 * np.pi, n_qubits, endpoint=False) + math.radians(15.0)
    positions = {i: (radius * np.cos(theta), radius * np.sin(theta)) for i, theta in enumerate(angles)}
    for u, v in edges:
        x0, y0 = positions[int(u)]
        x1, y1 = positions[int(v)]
        ax.plot([x0, x1], [y0, y1], color="#C9272D", linewidth=1.75, alpha=0.72, zorder=1)
    draw_nodes(ax, positions, node_size=360, font_size=12)
    ax.text(0.0, -1.62, rf"edge probability $={100.0 * edge_prob:.0f}\%$, unit weights", ha="center", va="center", fontsize=13)
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.82, 1.62)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    sk_problem = build_sk_problem(n_qubits=int(args.sk_qubits), seed=seed)
    grid_problem = build_grid2d_problem(rows=int(args.grid_rows), cols=int(args.grid_cols), seed=seed)
    maxcut_edges = _make_erdos_renyi_graph(
        n_qubits=int(args.maxcut_qubits),
        edge_prob=float(args.maxcut_edge_prob),
        seed=seed,
        ensure_connected=True,
        max_tries=300,
    )

    signed_values = (
        [w for _, _, w in sk_problem.edges]
        + [w for _, _, w in grid_problem.edges]
        + [h for _, h in grid_problem.fields]
    )
    abs_values = np.abs(np.asarray(signed_values, dtype=np.float64))
    quantile = float(np.clip(float(args.color_quantile), 0.50, 1.0))
    vmax = float(np.quantile(abs_values, quantile))
    vmax = max(vmax, 1.0, float(args.symlog_linthresh) * 1.05, 1e-12)
    cmap = LinearSegmentedColormap.from_list(
        "signed_blue_tint_red",
        ["#1038C6", "#BFD0FF", "#FFD2D2", "#D21F1F"],
    )
    norm = SymLogNorm(
        linthresh=max(float(args.symlog_linthresh), 1e-6),
        linscale=2.5,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )

    fig = plt.figure(figsize=(15.3, 4.2), constrained_layout=False)
    axes = [
        fig.add_axes([0.03, 0.19, 0.29, 0.68]),
        fig.add_axes([0.33, 0.19, 0.34, 0.68]),
        fig.add_axes([0.69, 0.19, 0.29, 0.68]),
    ]
    draw_sk_panel(axes[0], sk_problem, cmap, norm)
    draw_grid_panel(axes[1], grid_problem, cmap, norm)
    draw_maxcut_panel(axes[2], maxcut_edges, int(args.maxcut_qubits), float(args.maxcut_edge_prob))

    titles = [
        f"(a) SK {int(args.sk_qubits)}Q",
        rf"(b) 2D grid ({int(args.grid_rows)}$\times${int(args.grid_cols)})",
        f"(c) MaxCut {int(args.maxcut_qubits)}Q",
    ]
    for ax, title in zip(axes, titles):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2.0, 0.905, title, ha="center", va="bottom", fontsize=18)

    cax = fig.add_axes([0.18, 0.085, 0.64, 0.033])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Signed coupling/field scale (symlog)", fontsize=14, labelpad=5)
    cbar.ax.tick_params(labelsize=13, length=6, width=1.2)
    cbar.outline.set_linewidth(1.6)
    cbar.set_ticks([-1.0, -0.1, 0.0, 0.1, 1.0])
    cbar.set_ticklabels([r"$-10^0$", r"$-10^{-1}$", r"$0$", r"$10^{-1}$", r"$10^0$"])

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    print(f"saved figure: {output}")
    print(f"maxcut_edges={len(maxcut_edges)} edge_prob={float(args.maxcut_edge_prob):.3f}")


if __name__ == "__main__":
    main()
