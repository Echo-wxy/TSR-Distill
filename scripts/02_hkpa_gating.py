"""
02_hkpa_gating.py — Inspect the multi-scale decomposition and adaptive
gating produced by the Hierarchical Knowledge Projection Adapter (HKPA).

Run:
    python scripts/02_hkpa_gating.py

Output:
    figures/02_hkpa_gating.png

What it shows:
    Top row     — for one teacher feature Z_i^l, the three hierarchical
                  branches G^1, G^2, G^3 produced by 1-D convolutions with
                  kernels k_h in {3, 7, 15}. The receptive field grows
                  from local (k=3) to global (k=15) context, visible as
                  increasingly smoothed temporal profiles.
    Bottom-left — softmax gate weights g_h evolving across Stage 2 training.
                  The gates start uniform (each = 1/L_h) and migrate to
                  up-weight the granularity level that minimises the
                  combined feature-regression + domain-aware loss.
    Bottom-right — feature regression loss l_feat (Eq. 8) decreasing over
                  Stage 2 epochs.

For this controlled visualisation we construct a teacher feature whose
class-discriminative information lives at a coarse temporal scale (a
slow sinusoid whose frequency depends on class label) buried in fast
random noise. The k=3 branch picks up mostly noise; the k=15 branch
captures the slow class signal. A correctly working HKPA should learn
to up-weight the k=15 gate accordingly. This isolates the gating effect
in a way that the real-data tsr-distill pipeline (where all three
scales typically contribute non-trivially) does not.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tsr_distill import (
    HKPA,
    AuxClassifier,
    feature_regression_loss,
    domain_aware_loss,
)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- CONTROLLED EXPERIMENT: construct a teacher feature where the
    # class-discriminative information lives ONLY at a coarse temporal
    # granularity. Specifically, we build a synthetic (B, T, d) feature
    # whose class identity is encoded in a low-frequency component that
    # only the k=15 convolution can pick up cleanly; the k=3 convolution
    # sees mostly local noise. A well-trained HKPA should up-weight the
    # k=15 branch and down-weight the k=3 branch.
    B = 256
    T = 32
    d = 64
    num_classes = 10
    rng = np.random.RandomState(0)
    labels = rng.randint(0, num_classes, B).astype(np.int64)
    t_axis = np.linspace(0, 1, T)

    # Class-specific slow component (only visible to large kernels) +
    # class-independent fast noise (drowns out small kernels).
    Z = np.zeros((B, T, d), dtype=np.float32)
    for b in range(B):
        c = labels[b]
        slow = np.sin(2 * np.pi * (1 + c / 5.0) * t_axis)        # period ~ T/2
        fast = rng.randn(T) * 1.5
        Z[b] = (slow[:, None] + fast[:, None]).astype(np.float32)
    Z_tea = torch.from_numpy(Z)
    y = torch.from_numpy(labels)

    # ---- BUILD TWO STAGE-2-STYLE TARGETS. We need a template student
    # feature for the l_feat term. A simple choice: the slow class-
    # specific component, exposed at T_t = 9 positions.
    T_t = 9
    Z_stu = np.zeros((B, T_t, d), dtype=np.float32)
    t_stu_axis = np.linspace(0, 1, T_t)
    for b in range(B):
        c = labels[b]
        slow = np.sin(2 * np.pi * (1 + c / 5.0) * t_stu_axis)
        Z_stu[b] = np.tile(slow[:, None], (1, d))
    Z_stu = torch.from_numpy(Z_stu)

    # Mini-batch the controlled data through a DataLoader.
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(Z_tea, Z_stu, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    # HKPA with three kernel sizes.
    kernel_sizes = (3, 7, 15)
    hkpa = HKPA(d_in=d, d_mid=64, d_out=d, target_T=T_t, kernel_sizes=kernel_sizes)
    aux_per_branch = torch.nn.ModuleList(
        [AuxClassifier(64, num_classes) for _ in kernel_sizes]
    )
    params_h = list(hkpa.parameters()) + list(aux_per_branch.parameters())
    opt_h = torch.optim.Adam(params_h, lr=5e-3)

    gate_history = []
    loss_history = []
    epochs = 20
    for ep in range(epochs):
        hkpa.train()
        ep_loss = 0.0; n = 0
        for f_tea, f_stu, yy in loader:
            z_tilde = hkpa(f_tea)
            l_feat = feature_regression_loss(z_tilde, f_stu)
            # Per-branch CE: each branch must classify on its own. The
            # gate then weights the most discriminative branch higher.
            l_branch = 0.0
            for h, head in enumerate(aux_per_branch):
                g_h = hkpa.last_G[h]
                l_branch = l_branch + domain_aware_loss(
                    head(g_h.mean(dim=1)), yy
                )
            loss = l_feat + 0.5 * l_branch
            opt_h.zero_grad(); loss.backward(); opt_h.step()
            ep_loss += l_feat.item(); n += 1
        loss_history.append(ep_loss / max(1, n))
        gate_history.append(hkpa.last_gate.cpu().numpy().copy())

    # ---- Capture the per-branch hierarchical features G^h for one batch.
    hkpa.eval()
    with torch.no_grad():
        f_tea_batch, _, _ = next(iter(loader))
        _ = hkpa(f_tea_batch)
        G_branches = [g[0].cpu().numpy() for g in hkpa.last_G]   # each (T_t, d_mid)

    # ---- PLOT
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.35)

    # Top row: three branches as heatmaps (T_t x d_mid, but show only first
    # 24 channels for visual clarity).
    vmax = max(np.abs(g).max() for g in G_branches)
    for i, (g, k) in enumerate(zip(G_branches, kernel_sizes)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(g[:, :24].T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"$G^{i + 1}$ — kernel $k_h = {k}$\n"
                     f"{'local' if k <= 5 else 'mid-range' if k <= 9 else 'global'} context",
                     fontsize=10)
        ax.set_xlabel("temporal position $t$")
        if i == 0:
            ax.set_ylabel("channel (first 24 of $d_{mid}$)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom-left: gate evolution.
    ax_g = fig.add_subplot(gs[1, 0:2])
    gates = np.stack(gate_history, axis=0)            # (epochs, L_h)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for h in range(gates.shape[1]):
        ax_g.plot(range(1, gates.shape[0] + 1), gates[:, h],
                  marker="o", linewidth=2,
                  label=f"$g_{h + 1}$  ($k_h = {kernel_sizes[h]}$)",
                  color=colors[h])
    ax_g.set_xlabel("Stage 2 epoch")
    ax_g.set_ylabel("softmax gate weight $g_h$")
    ax_g.set_title("Adaptive gates $g_h$ depart from uniform and "
                   "settle on a data-driven mixture", fontsize=11)
    ax_g.axhline(1.0 / 3, linestyle="--", color="grey", alpha=0.6,
                 label="uniform 1/$L_h$")
    ax_g.legend(loc="best", fontsize=9)
    ax_g.grid(True, alpha=0.3)
    ax_g.set_ylim(0, 1)

    # Bottom-right: regression loss.
    ax_l = fig.add_subplot(gs[1, 2])
    ax_l.plot(range(1, len(loss_history) + 1), loss_history,
              marker="s", color="#d62728", linewidth=2)
    ax_l.set_xlabel("Stage 2 epoch")
    ax_l.set_ylabel(r"$\ell_{\mathrm{feat}}$ (Eq. 8)")
    ax_l.set_title("Feature regression loss\ndecreases monotonically",
                   fontsize=11)
    ax_l.grid(True, alpha=0.3)

    fig.suptitle("Hierarchical Knowledge Projection Adapter (HKPA): "
                 "multi-granularity decomposition + adaptive gating",
                 fontsize=12, fontweight="bold")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "02_hkpa_gating.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved: {out_path}")
    print("final gate weights:", gates[-1].round(3).tolist())
    print(f"feat loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
