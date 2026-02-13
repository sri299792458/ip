from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("instant_policy/ip/docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = "DejaVu Serif"
INK = "#111827"
MUTED = "#4B5563"

# Colorblind-safe palette (Okabe-Ito inspired)
C_NODE_SL = "#009E73"  # scene_left
C_NODE_SR = "#E69F00"  # scene_right
C_NODE_GL = "#0072B2"  # gripper_left
C_NODE_GR = "#CC79A7"  # gripper_right

C_EDGE_SSL = "#009E73"   # scene_left rel scene_left
C_EDGE_SSR = "#E69F00"   # scene_right rel scene_right
C_EDGE_SG = "#7F7F7F"    # scene->gripper (left+right)
C_EDGE_GGL = "#0072B2"   # gripper_left rel gripper_left
C_EDGE_GGR = "#CC79A7"   # gripper_right rel gripper_right
C_EDGE_CROSS = "#D55E00" # gripper_left <-> gripper_right cross
C_EDGE_TEMP = "#A3A9B1"  # temporal/context linking


def txt(ax, x, y, s, size=14, weight="normal", color=INK, ha="center", va="center", rotation=0):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color,
        fontfamily=FONT,
        ha=ha,
        va=va,
        rotation=rotation,
    )


def pairs_complete(n, undirected=True):
    if undirected:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def pairs_bipartite(n_src, n_dst):
    return [(i, j) for i in range(n_src) for j in range(n_dst)]


def state_template():
    # Two grippers (6 nodes each) + two scene subsets (8 nodes each).
    g = np.array(
        [
            [-0.28, 0.30],
            [0.00, 0.36],
            [0.28, 0.30],
            [-0.16, 0.08],
            [0.16, 0.08],
            [0.00, -0.06],
        ],
        dtype=np.float32,
    )
    s = np.array(
        [
            [-0.36, -0.30],
            [-0.12, -0.40],
            [0.10, -0.34],
            [0.34, -0.24],
            [-0.28, -0.60],
            [-0.02, -0.68],
            [0.25, -0.58],
            [-0.06, -0.16],
        ],
        dtype=np.float32,
    )
    return {
        "gripper_left": g + np.array([-1.15, 0.95], dtype=np.float32),
        "scene_left": s + np.array([-1.15, 0.00], dtype=np.float32),
        "gripper_right": g + np.array([1.15, 0.95], dtype=np.float32),
        "scene_right": s + np.array([1.15, 0.00], dtype=np.float32),
    }


def draw_edges(ax, src, dst, pairs, color, lw=0.8, alpha=0.28, ls="-", z=1):
    for i, j in pairs:
        p = src[i]
        q = dst[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=alpha, ls=ls, zorder=z)


def draw_nodes(ax, pts, color, marker="o", size=24, z=4):
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=size,
        marker=marker,
        c=color,
        edgecolors="white",
        linewidths=0.65,
        zorder=z,
    )


def draw_state_graph(ax, cx, cy, scale=0.06, dense=False):
    t = state_template()
    sl = t["scene_left"] * scale + np.array([cx, cy], dtype=np.float32)
    sr = t["scene_right"] * scale + np.array([cx, cy], dtype=np.float32)
    gl = t["gripper_left"] * scale + np.array([cx, cy], dtype=np.float32)
    gr = t["gripper_right"] * scale + np.array([cx, cy], dtype=np.float32)

    # Graph contract relations in ip/bimanual/graph_rep.py.
    ss_l = pairs_complete(len(sl), undirected=True)
    ss_r = pairs_complete(len(sr), undirected=True)
    sg_l = pairs_bipartite(len(sl), len(gl))
    sg_r = pairs_bipartite(len(sr), len(gr))
    gg_l = pairs_complete(len(gl), undirected=True)  # self-loops omitted visually
    gg_r = pairs_complete(len(gr), undirected=True)
    cross = pairs_bipartite(len(gl), len(gr))

    if dense:
        lw_ss, lw_sg, lw_gg, lw_x = 1.60, 1.35, 1.60, 1.70
        a_ss, a_sg, a_gg, a_x = 0.52, 0.44, 0.58, 0.62
        node_scene_sz, node_grip_sz = 140, 172
    else:
        lw_ss, lw_sg, lw_gg, lw_x = 1.22, 1.02, 1.22, 1.28
        a_ss, a_sg, a_gg, a_x = 0.48, 0.42, 0.52, 0.58
        node_scene_sz, node_grip_sz = 86, 102

    draw_edges(ax, sl, sl, ss_l, C_EDGE_SSL, lw=lw_ss, alpha=a_ss, ls="-")
    draw_edges(ax, sr, sr, ss_r, C_EDGE_SSR, lw=lw_ss, alpha=a_ss, ls="-")
    draw_edges(ax, sl, gl, sg_l, C_EDGE_SG, lw=lw_sg, alpha=a_sg, ls="--")
    draw_edges(ax, sr, gr, sg_r, C_EDGE_SG, lw=lw_sg, alpha=a_sg, ls="--")
    draw_edges(ax, gl, gl, gg_l, C_EDGE_GGL, lw=lw_gg, alpha=a_gg, ls="-")
    draw_edges(ax, gr, gr, gg_r, C_EDGE_GGR, lw=lw_gg, alpha=a_gg, ls="-")
    draw_edges(ax, gl, gr, cross, C_EDGE_CROSS, lw=lw_x, alpha=a_x, ls="-.")

    draw_nodes(ax, sl, C_NODE_SL, marker="o", size=node_scene_sz)
    draw_nodes(ax, sr, C_NODE_SR, marker="o", size=node_scene_sz)
    draw_nodes(ax, gl, C_NODE_GL, marker="s", size=node_grip_sz)
    draw_nodes(ax, gr, C_NODE_GR, marker="D", size=node_grip_sz)

    return {"scene_left": sl, "scene_right": sr, "gripper_left": gl, "gripper_right": gr}


def draw_arrow(ax, p0, p1, color="#374151", lw=1.3, style="-|>", rad=0.0, alpha=0.95, z=3, mscale=13):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            mutation_scale=mscale,
            lw=lw,
            color=color,
            alpha=alpha,
            zorder=z,
        )
    )


def draw_action_axes(ax, x, y, s=0.012):
    draw_arrow(ax, (x, y), (x + s, y), color="#16A34A", lw=1.35, rad=0)
    draw_arrow(ax, (x, y), (x, y + s), color="#2563EB", lw=1.35, rad=0)
    draw_arrow(ax, (x, y), (x - 0.7 * s, y + 0.55 * s), color="#DC2626", lw=1.35, rad=0)


def draw_type_legend(ax, x0, y0):
    box = FancyBboxPatch((x0, y0), 0.20, 0.135, boxstyle="round,pad=0.008,rounding_size=0.008", fc="#F9FAFB", ec="#D1D5DB", lw=0.8)
    ax.add_patch(box)
    txt(ax, x0 + 0.010, y0 + 0.116, "Node Types", size=11, weight="bold", ha="left")

    y = y0 + 0.090
    entries = [
        ("o", C_NODE_SL, "scene_left (N_l)"),
        ("o", C_NODE_SR, "scene_right (N_r)"),
        ("s", C_NODE_GL, "gripper_left (G=6)"),
        ("D", C_NODE_GR, "gripper_right (G=6)"),
    ]
    for m, c, label in entries:
        ax.scatter([x0 + 0.012], [y], s=40, marker=m, c=c, edgecolors="white", linewidths=0.4, zorder=6)
        txt(ax, x0 + 0.022, y, label, size=10, ha="left")
        y -= 0.023


def draw_edge_legend(ax, x0, y0):
    box = FancyBboxPatch((x0, y0), 0.300, 0.185, boxstyle="round,pad=0.008,rounding_size=0.008", fc="#F9FAFB", ec="#D1D5DB", lw=0.8)
    ax.add_patch(box)
    txt(ax, x0 + 0.010, y0 + 0.169, "Edge Types (from BimanualGraphRep)", size=10.8, weight="bold", ha="left")

    entries = [
        (C_EDGE_SSL, "-", '("scene_left","rel","scene_left")'),
        (C_EDGE_SSR, "-", '("scene_right","rel","scene_right")'),
        (C_EDGE_SG, "--", '("scene_left/right","rel","gripper_left/right")'),
        (C_EDGE_GGL, "-", '("gripper_left","rel","gripper_left")'),
        (C_EDGE_GGR, "-", '("gripper_right","rel","gripper_right")'),
        (C_EDGE_CROSS, "-.", '("gripper_left","cross","gripper_right") + reverse'),
    ]
    y = y0 + 0.143
    for c, ls, label in entries:
        ax.plot([x0 + 0.010, x0 + 0.038], [y, y], color=c, lw=2.0, ls=ls, solid_capstyle="round")
        txt(ax, x0 + 0.042, y, label, size=9.0, ha="left")
        y -= 0.024


def main():
    fig = plt.figure(figsize=(16, 9), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top title row intentionally omitted.

    # Vertical separators
    ax.add_line(Line2D([0.25, 0.25], [0.26, 0.82], color="#2B5EA8", lw=1.6, ls=(0, (5, 4))))
    ax.add_line(Line2D([0.65, 0.65], [0.31, 0.82], color="#2B5EA8", lw=1.2, ls=(0, (5, 4)), alpha=0.8))

    # Left panel
    draw_state_graph(ax, 0.125, 0.50, scale=0.098, dense=True)
    draw_type_legend(ax, x0=0.035, y0=0.165)

    # Center context panel
    context_box = FancyBboxPatch((0.275, 0.33), 0.355, 0.48, boxstyle="round,pad=0.008,rounding_size=0.014", fc="none", ec="#2B5EA8", lw=1.5, ls=(0, (4, 3)))
    ax.add_patch(context_box)
    txt(ax, 0.452, 0.787, "Context", size=17, color="#D97706")
    txt(ax, 0.452, 0.748, "Sequence of States", size=17)
    txt(ax, 0.340, 0.716, "1", size=16)
    txt(ax, 0.447, 0.716, "2", size=16)
    txt(ax, 0.548, 0.716, "L", size=16)
    txt(ax, 0.279, 0.598, "Demo 1", size=16, rotation=90)
    txt(ax, 0.279, 0.432, "Demo N", size=16, rotation=90)

    c_top = [0.338, 0.444, 0.548]
    c_bot = [0.338, 0.444, 0.548]
    [draw_state_graph(ax, x, 0.61, scale=0.048, dense=False) for x in c_top]
    [draw_state_graph(ax, x, 0.43, scale=0.048, dense=False) for x in c_bot]

    # Temporal arrows inside demos
    draw_arrow(ax, (0.362, 0.61), (0.419, 0.61), color=C_EDGE_TEMP, style="-|>", lw=1.20, rad=-0.20, alpha=0.9)
    draw_arrow(ax, (0.468, 0.61), (0.525, 0.61), color=C_EDGE_TEMP, style="-|>", lw=1.20, rad=-0.20, alpha=0.9)
    draw_arrow(ax, (0.362, 0.43), (0.419, 0.43), color=C_EDGE_TEMP, style="-|>", lw=1.20, rad=-0.20, alpha=0.9)
    draw_arrow(ax, (0.468, 0.43), (0.525, 0.43), color=C_EDGE_TEMP, style="-|>", lw=1.20, rad=-0.20, alpha=0.9)

    # Right rollout panel
    txt(ax, 0.73, 0.787, "Current State", size=18, color="#15803D")
    txt(ax, 0.835, 0.787, "Action 1", size=18, color="#DC2626")
    txt(ax, 0.93, 0.787, "Action T", size=18, color="#DC2626")

    txt(ax, 0.73, 0.718, "G_t", size=11, color=MUTED)
    txt(ax, 0.835, 0.718, "A_1", size=11, color=MUTED)
    txt(ax, 0.93, 0.718, "A_T", size=11, color=MUTED)

    g_cur = draw_state_graph(ax, 0.73, 0.47, scale=0.048, dense=False)
    g_a1 = draw_state_graph(ax, 0.835, 0.47, scale=0.048, dense=False)
    g_at = draw_state_graph(ax, 0.93, 0.47, scale=0.048, dense=False)

    draw_action_axes(ax, 0.835, 0.695, s=0.019)
    draw_action_axes(ax, 0.93, 0.695, s=0.019)

    # Explicit context-to-current transfer from terminal context states only.
    draw_arrow(ax, (0.548, 0.61), (0.704, 0.505), color="#1D4ED8", lw=2.2, rad=0.0, alpha=0.98, z=8, mscale=14)
    draw_arrow(ax, (0.548, 0.43), (0.704, 0.455), color="#1D4ED8", lw=2.2, rad=0.0, alpha=0.98, z=8, mscale=14)
    txt(ax, 0.664, 0.545, "context -> G_t", size=10.2, color="#1D4ED8")

    draw_arrow(ax, (0.752, 0.545), (0.812, 0.545), color="#2B5EA8", lw=1.6, rad=0.0, alpha=0.95)
    draw_arrow(ax, (0.857, 0.545), (0.907, 0.545), color="#2B5EA8", lw=1.6, rad=0.0, alpha=0.95)
    txt(ax, 0.782, 0.566, "ΔT_L, ΔT_R, g_L, g_R", size=9.3, color=MUTED)
    txt(ax, 0.883, 0.566, "...", size=12, color=MUTED)

    # Legend + contract notes
    draw_edge_legend(ax, x0=0.635, y0=0.105)

    png = OUT_DIR / "bimanual_model_diagram.png"
    svg = OUT_DIR / "bimanual_model_diagram.svg"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    print(f"wrote: {png}")
    print(f"wrote: {svg}")


if __name__ == "__main__":
    main()
