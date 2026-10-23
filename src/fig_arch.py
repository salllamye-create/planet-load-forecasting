"""Figure 1: what the phase-locked basis is, and how the three tiers fit together."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as fs
from data import build_basis

BLUE, ORANGE, GREEN, PURPLE = "#0B5FA5", "#C44601", "#00916B", "#7D3C98"
# column layout of build_basis(n_day=2, n_week=1, n_year=0, trend=False, daytype=True):
# 0 const | 1-4 daily k=1,2 | 5-6 weekly | 7 Sat +4 interactions | 12 Sun +4 interactions
I_SIN_D1, I_SIN_D2, I_SIN_W1, I_SAT, I_SUN = 1, 3, 5, 7, 12


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=fs.MUTED, fontsize=6.3, lw=0.7, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.010",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, color=tc or fs.INK, linespacing=1.4)


def arrow(ax, p, q, color=None, lw=0.75, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=color or fs.MUTED, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1.0, shrinkB=1.0))


def main():
    fs.setup()
    fig = plt.figure(figsize=(fs.DBL, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.62], wspace=0.13)

    # ---------------------------------------------------------------- (a) the basis
    ax = fig.add_subplot(gs[0, 0])
    L, H = 168, 96
    t0 = 24.0 * (365 * 46 + 3)                     # arbitrary origin, hours since epoch
    tau = np.arange(t0 - L, t0 + H, dtype=float)
    Phi = build_basis(tau, n_day=2, n_week=1, n_year=0, trend=False, daytype=True)
    rel = np.arange(-L, H)

    rows = [(I_SIN_D1, "daily, $k{=}1$", BLUE),
            (I_SIN_D2, "daily, $k{=}2$", GREEN),
            (I_SIN_W1, "weekly, $k{=}1$", PURPLE)]
    for j, (idx, lab, c) in enumerate(rows):
        base = 3.0 - 1.15 * j
        ax.plot(rel, Phi[:, idx] * 0.42 + base, color=c, lw=1.05, zorder=3)
        ax.text(-L + 4, base + 0.50, lab, fontsize=6.0, color=c, va="bottom", zorder=6)
    wknd = Phi[:, I_SAT] + Phi[:, I_SUN]
    ax.fill_between(rel, -0.55, -0.55 + wknd * 0.42, color=ORANGE, alpha=0.8, lw=0,
                    zorder=3, step="mid")
    ax.text(-L + 4, -0.55 + 0.50, "weekend indicator", fontsize=6.0, color=ORANGE,
            va="bottom", zorder=6)

    ax.axvline(0, color=fs.INK, lw=0.9, ls="--", zorder=5)
    ax.annotate("", xy=(H, 3.90), xytext=(0, 3.90),
                arrowprops=dict(arrowstyle="->", color=fs.INK, lw=0.6))
    ax.text(H / 2, 3.99, "horizon", fontsize=6.1, color=fs.INK, ha="center", va="bottom")
    ax.text(-4, 3.99, "origin", fontsize=6.1, color=fs.INK, ha="right", va="bottom")
    ax.set_xlim(-L, H); ax.set_ylim(-1.15, 4.42)
    ax.set_yticks([]); ax.set_xticks([-168, -96, 0, 96])
    ax.set_xlabel("hours relative to forecast origin")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) Basis anchored to wall-clock time", loc="left", fontsize=7.5)

    # ---------------------------------------------------------------- (b) block diagram
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    box(ax, 0.00, 0.455, 0.135, 0.20, "lookback\n$\\mathbf{X}$\n$\\tau_{1:L+H}$", fc="#F3F5F8")
    box(ax, 0.175, 0.775, 0.235, 0.135, "climatological tier\n$g_c=\\psi(\\tau)^\\top\\mathbf{B}_c$",
        fc="#EAF1FA", ec=BLUE, tc=BLUE)
    box(ax, 0.175, 0.475, 0.235, 0.135, "instance norm\n$(\\tilde{\\mathbf{X}}-\\mu)/\\sigma$",
        fc="#F3F5F8")
    box(ax, 0.175, 0.185, 0.235, 0.135, "weighting net\n$w=L\\,\\mathrm{softmax}(f_\\theta)$",
        fc="#FDF0E8", ec=ORANGE, tc=ORANGE)

    box(ax, 0.455, 0.455, 0.36, 0.215,
        "phase-locked solver\n"
        "$\\beta_c=(\\Phi_L^{\\top}W_c\\Phi_L+\\Lambda_c)^{-1}$\n"
        "$\\qquad\\times\\;\\Phi_L^{\\top}W_c\\,\\hat{x}_c$",
        fc="#FDF0E8", ec=ORANGE, lw=1.2, tc=ORANGE, fontsize=6.2)
    box(ax, 0.455, 0.185, 0.36, 0.135,
        "residual encoder\npatch Transformer $+\\,\\beta_c$ token",
        fc="#E9F6F1", ec=GREEN, tc=GREEN)
    box(ax, 0.875, 0.455, 0.115, 0.215, "$\\hat{\\mathbf{Y}}$", fc="#F3F5F8", fontsize=9.0)

    arrow(ax, (0.135, 0.615), (0.175, 0.800))                  # X -> climatology
    arrow(ax, (0.135, 0.553), (0.175, 0.545))                  # X -> instance norm
    arrow(ax, (0.2925, 0.772), (0.2925, 0.614), color=BLUE)    # subtract g
    ax.text(0.302, 0.695, "$-\\,g$", fontsize=6.0, color=BLUE, va="center")
    arrow(ax, (0.2925, 0.472), (0.2925, 0.323))                # -> weighting net
    arrow(ax, (0.412, 0.545), (0.455, 0.560))                  # x_hat -> solver
    ax.text(0.417, 0.583, "$\\hat{x}$", fontsize=6.0, color=fs.MUTED)
    arrow(ax, (0.412, 0.268), (0.455, 0.478), color=ORANGE, rad=-0.22)
    ax.text(0.407, 0.355, "$w$", fontsize=6.0, color=ORANGE, ha="right")
    arrow(ax, (0.635, 0.452), (0.635, 0.323))                  # residual passed down
    ax.text(0.648, 0.388, "$\\mathbf{R}=\\hat{\\mathbf{X}}-\\Phi_L\\beta_c$",
            fontsize=5.9, color=fs.MUTED, va="center")
    arrow(ax, (0.815, 0.585), (0.875, 0.565), color=ORANGE)
    ax.text(0.845, 0.605, "$\\Phi_H\\beta_c$", fontsize=5.9, color=ORANGE, ha="center")
    arrow(ax, (0.815, 0.252), (0.875, 0.490), color=GREEN, rad=-0.25)
    ax.text(0.868, 0.245, "$\\mathbf{R}^{\\mathrm{fut}}$", fontsize=5.9, color=GREEN, ha="right")
    arrow(ax, (0.410, 0.880), (0.9325, 0.678), color=BLUE, rad=-0.26)
    ax.text(0.672, 0.945, "$g(\\tau_{L+1:L+H})$", fontsize=5.9, color=BLUE, ha="center")
    ax.set_title("(b) Three additive tiers", loc="left", fontsize=7.5)

    fs.save(fig, "fig_arch")


if __name__ == "__main__":
    main()
