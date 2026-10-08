"""Plots of a closed-loop run, built around what the estimator is doing.

The point of these figures is not that the vehicle follows the path -- it is *how far the
controller's picture of the vehicle is from the vehicle*, and whether that picture describes
a machine that could exist at all.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SEGMENT_COLOURS = ["#ffb74d", "#4fc3f7", "#81d4fa", "#b3e5fc"]
TRUE, EST, LIMIT = "#66bb6a", "#ef5350", "#b71c1c"


def draw_vehicle(ax, poses, geo, alpha=1.0, zorder=5):
    """Every segment as a rectangle, with the hitch links between them."""
    out = []
    for i in range(poses.shape[1]):
        x, y, th = poses[:, i]
        L, W = geo.body_length[i], geo.body_width[i]
        r = Rectangle((-L / 2, -W / 2), L, W, facecolor=SEGMENT_COLOURS[min(i, 3)],
                      edgecolor="k", lw=0.6, alpha=alpha, zorder=zorder)
        r.set_transform(matplotlib.transforms.Affine2D().rotate(th).translate(x, y) + ax.transData)
        out.append(ax.add_patch(r))
    out.append(ax.plot(poses[0], poses[1], "-", color="0.35", lw=1.2, alpha=alpha,
                       zorder=zorder - 1)[0])
    return out


def plot_run(res, path_points, geo, model, ax=None, show_every=40, title=None):
    """The path, where the vehicle went, and where the estimator thought it was."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))
    p = np.asarray(path_points)
    ax.plot(p[:, 0], p[:, 1], "-", color="0.8", lw=6, zorder=1, label="path")

    last = geo.n_trailers
    true_xy = np.array([model.pose_array(x)[:, last][:2] for x in res.states])
    est_xy = np.array([model.pose_array(x)[:, last][:2] for x in res.estimates])
    ax.plot(true_xy[:, 0], true_xy[:, 1], "-", color=TRUE, lw=2, zorder=3,
            label="last trailer, true")
    ax.plot(est_xy[:, 0], est_xy[:, 1], "--", color=EST, lw=1.6, zorder=3,
            label="last trailer, estimated")

    for k in range(0, len(res.states), max(1, show_every)):
        draw_vehicle(ax, res.poses[k], geo, alpha=0.45 if k else 0.9)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if title:
        ax.set_title(title)
    return ax


def plot_diagnostics(res, ts, beta_limit, axes=None, title=None):
    """Three panels: how wrong the estimate is, whether it is even possible, and the angles."""
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    t = res.time

    axes[0].plot(t, res.estimation_error, color=EST, lw=1.5)
    axes[0].set_ylabel("estimation\nerror [m]")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, res.infeasibility, color="#8a94a6", lw=1.5)
    axes[1].set_ylabel("infeasibility [m]")
    axes[1].grid(alpha=0.25)
    axes[1].annotate("distance between the estimated last-trailer pose\n"
                     "and the one the chain actually allows",
                     xy=(0.02, 0.86), xycoords="axes fraction", fontsize=8, va="top",
                     color="0.35")

    for i in range(res.joint_angles.shape[1]):
        axes[2].plot(t, res.joint_angles[:len(t), i], lw=1.4,
                     color=SEGMENT_COLOURS[min(i + 1, 3)], label=rf"$\beta_{i + 1}$")
    for s in (1, -1):
        axes[2].axhline(s * beta_limit, color=LIMIT, lw=2)
    axes[2].set_ylabel("joint angles [rad]")
    axes[2].set_xlabel("t [s]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=9, ncol=2)
    if title:
        axes[0].set_title(title)
    return axes


def compare_bars(summaries, keys=("estimation_mean", "infeasibility_mean", "off_track_mean"),
                 labels=None, ax=None):
    """Side-by-side bars for several runs, e.g. one per estimator."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    names = list(summaries)
    width = 0.8 / len(keys)
    labels = labels or {k: k.replace("_", " ") for k in keys}
    for j, k in enumerate(keys):
        xs = np.arange(len(names)) + j * width - 0.4 + width / 2
        ax.bar(xs, [summaries[n][k] for n in names], width=width, label=labels[k])
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("metres")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    return ax
