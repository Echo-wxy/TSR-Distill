"""
make_paper_figures.py — Publication-quality versions of the paper's
Figures 2, 3, and 4, generated from the SAME computation as the
reference scripts:

    Figure 2  <-  scripts/01_cta_alignment.py      (computation copied verbatim)
    Figure 3  <-  scripts/02_hkpa_gating.py        (computation copied verbatim)
    Figure 4  <-  scripts/03_car_routing.py::_run_diagnostic_car (verbatim)

Only the plotting differs: 300 dpi, PDF + TIFF, Times New Roman,
no figure titles, top-mounted legends, (a)(b)(c) panel tags.

Place this file in the repo's  scripts/  folder, then run:

    python scripts/make_paper_figures.py            # all three
    python scripts/make_paper_figures.py 2          # only Figure 2

Outputs: figures/Figure2.{pdf,tiff}, Figure3.{pdf,tiff}, Figure4.{pdf,tiff}
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.family": "serif",
    # Times New Roman first; metric-compatible fallbacks if unavailable.
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})
BLUE, TAN, GREEN, RED = "#4C6A92", "#B0885E", "#6A8F5F", "#9A5B5B"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name + ".pdf"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, name + ".tiff"), dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(OUT_DIR, name + "_preview.png"), dpi=300)
    print(f"saved: {OUT_DIR}/{name}.pdf + .tiff (300 dpi)")


# =====================================================================
# Figure 2 — CTA alignment (computation verbatim from 01_cta_alignment)
# =====================================================================
def compute_fig2():
    from tsr_distill import (
        SyntheticAVDataset, ImageEncoder, AudioEncoder,
        soft_alignment_matrix, cta_loss, icc_loss,
    )
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    np.random.seed(0)

    dataset = SyntheticAVDataset(num_samples=512, seed=0)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    img_enc = ImageEncoder()
    aud_enc = AudioEncoder()

    img_enc.eval(); aud_enc.eval()
    with torch.no_grad():
        x_img, x_aud, _ = next(iter(loader))
        _, f_img = img_enc(x_img, return_feat=True)
        _, f_aud = aud_enc(x_aud, return_feat=True)
        S_before = soft_alignment_matrix(f_img, f_aud, kappa=0.1)[0].numpy()
        ent_before = (-(soft_alignment_matrix(f_img, f_aud, 0.1)
                        * torch.log(soft_alignment_matrix(f_img, f_aud, 0.1) + 1e-8)
                        ).sum(dim=-1)).mean().item()

    opt = torch.optim.Adam(
        list(img_enc.parameters()) + list(aud_enc.parameters()), lr=1e-3
    )
    img_enc.train(); aud_enc.train()
    entropy_curve = [ent_before]
    epochs = 8
    for ep in range(epochs):
        for x_img, x_aud, y in loader:
            li, f_img = img_enc(x_img, return_feat=True)
            la, f_aud = aud_enc(x_aud, return_feat=True)
            loss = (
                F.cross_entropy(li, y)
                + F.cross_entropy(la, y)
                + 0.5 * cta_loss(f_img, f_aud, kappa=0.1)
                + 1.0 * icc_loss(f_img.mean(dim=1), f_aud.mean(dim=1), tau=0.07)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        img_enc.eval(); aud_enc.eval()
        with torch.no_grad():
            x_img, x_aud, _ = next(iter(loader))
            _, f_img = img_enc(x_img, return_feat=True)
            _, f_aud = aud_enc(x_aud, return_feat=True)
            S = soft_alignment_matrix(f_img, f_aud, kappa=0.1)
            ent = (-(S * torch.log(S + 1e-8)).sum(dim=-1)).mean().item()
            entropy_curve.append(ent)
        img_enc.train(); aud_enc.train()

    img_enc.eval(); aud_enc.eval()
    torch.manual_seed(0); np.random.seed(0)
    loader_eval = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    with torch.no_grad():
        x_img, x_aud, _ = next(iter(loader_eval))
        _, f_img = img_enc(x_img, return_feat=True)
        _, f_aud = aud_enc(x_aud, return_feat=True)
        S_after = soft_alignment_matrix(f_img, f_aud, kappa=0.1)[0].numpy()

    return S_before, S_after, np.asarray(entropy_curve)


def plot_fig2(S_before, S_after, entropy_curve):
    fig = plt.figure(figsize=(7.0, 3.9), dpi=300)
    gs = GridSpec(2, 2, height_ratios=[1.25, 1.0], hspace=0.62, wspace=0.28,
                  left=0.075, right=0.955, top=0.90, bottom=0.115)

    panels = [(S_before, "Before Stage 1 (random initialization)", "(a)"),
              (S_after,  "After Stage 1 (alignment loss minimized)", "(b)")]
    vmax = float(S_after.max())
    for k, (S, sub, tag) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(S, aspect="auto", cmap="cividis", vmin=0.0, vmax=vmax)
        ax.text(-0.075, 1.30, tag, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
        ax.set_title(sub, pad=3)
        ax.set_xlabel("Audio position", labelpad=1.5)
        ax.set_ylabel("Image position", labelpad=1.5)
        ax.tick_params(length=2, pad=1.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6.5, length=2)
        cb.set_label("Alignment weight", fontsize=7, labelpad=2)

    ax = fig.add_subplot(gs[1, :])
    ax.text(-0.032, 1.34, "(c)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    ep = np.arange(len(entropy_curve))
    ax.plot(ep, entropy_curve, color=BLUE, lw=1.4, marker="o", ms=3.2,
            mew=0, label="Mean per-row entropy")
    ax.axhline(entropy_curve[-1], color=TAN, lw=1.0, ls="--",
               label="Converged level")
    ax.set_xlabel("Stage 1 epoch", labelpad=2)
    ax.set_ylabel("Entropy (nats)", labelpad=2)
    ax.set_xlim(0, ep[-1])
    ax.set_xticks(ep)
    ax.tick_params(length=2, pad=1.5)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, borderaxespad=0, handlelength=1.8)
    _save(fig, "Figure2")
    plt.close(fig)


# =====================================================================
# Figure 3 — HKPA gating (computation verbatim from 02_hkpa_gating)
# =====================================================================
def compute_fig3():
    from tsr_distill import (
        HKPA, AuxClassifier, feature_regression_loss, domain_aware_loss,
    )
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(0)
    np.random.seed(0)

    B = 256
    T = 32
    d = 64
    num_classes = 10
    rng = np.random.RandomState(0)
    labels = rng.randint(0, num_classes, B).astype(np.int64)
    t_axis = np.linspace(0, 1, T)

    Z = np.zeros((B, T, d), dtype=np.float32)
    for b in range(B):
        c = labels[b]
        slow = np.sin(2 * np.pi * (1 + c / 5.0) * t_axis)
        fast = rng.randn(T) * 1.5
        Z[b] = (slow[:, None] + fast[:, None]).astype(np.float32)
    Z_tea = torch.from_numpy(Z)
    y = torch.from_numpy(labels)

    T_t = 9
    Z_stu = np.zeros((B, T_t, d), dtype=np.float32)
    t_stu_axis = np.linspace(0, 1, T_t)
    for b in range(B):
        c = labels[b]
        slow = np.sin(2 * np.pi * (1 + c / 5.0) * t_stu_axis)
        Z_stu[b] = np.tile(slow[:, None], (1, d))
    Z_stu = torch.from_numpy(Z_stu)

    ds = TensorDataset(Z_tea, Z_stu, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

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

    hkpa.eval()
    with torch.no_grad():
        f_tea_batch, _, _ = next(iter(loader))
        _ = hkpa(f_tea_batch)
        G_branches = [g[0].cpu().numpy() for g in hkpa.last_G]

    gates = np.stack(gate_history, axis=0)
    return G_branches, gates, np.asarray(loss_history), kernel_sizes


def plot_fig3(G_branches, gates, loss_history, kernel_sizes):
    fig = plt.figure(figsize=(7.0, 4.5), dpi=300)
    gs = GridSpec(2, 6, height_ratios=[1.0, 1.05], hspace=0.66, wspace=1.05,
                  left=0.07, right=0.945, top=0.88, bottom=0.10)

    ctx = {3: "local context", 7: "mid-range context", 15: "global context"}
    vmax = max(float(np.abs(g).max()) for g in G_branches)
    for k, (G, ks) in enumerate(zip(G_branches, kernel_sizes)):
        ax = fig.add_subplot(gs[0, 2*k:2*k+2])
        # (T_t, d_mid) -> show first 24 channels, channels on the y axis
        im = ax.imshow(G[:, :24].T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.text(-0.14, 1.30, f"({chr(97+k)})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.set_title(f"Kernel size {ks} ({ctx.get(ks, 'context')})", pad=3)
        ax.set_xlabel("Temporal position", labelpad=1.5)
        if k == 0:
            ax.set_ylabel("Channel", labelpad=1.5)
        ax.tick_params(length=2, pad=1.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.03)
        cb.ax.tick_params(labelsize=6.5, length=2)

    ax = fig.add_subplot(gs[1, 0:4])
    ax.text(-0.07, 1.30, "(d)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    ep = np.arange(1, gates.shape[0] + 1)
    for h, (c, m) in enumerate(zip([BLUE, TAN, GREEN], ["o", "s", "^"])):
        ax.plot(ep, gates[:, h], color=c, lw=1.4, marker=m, ms=2.8, mew=0,
                label=f"Gate {h+1} (kernel {kernel_sizes[h]})")
    ax.axhline(1.0 / gates.shape[1], color="0.55", lw=1.0, ls="--",
               label="Uniform initialization")
    ax.set_xlabel("Stage 2 epoch", labelpad=2)
    ax.set_ylabel("Gate weight", labelpad=2)
    ax.set_xlim(1, ep[-1])
    ax.set_ylim(0.0, max(0.6, float(gates.max()) + 0.1))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.tick_params(length=2, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, borderaxespad=0, handlelength=1.8,
              columnspacing=1.2)

    ax = fig.add_subplot(gs[1, 4:6])
    ax.text(-0.22, 1.30, "(e)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    ax.plot(ep, loss_history, color=RED, lw=1.4, marker="s", ms=2.8, mew=0,
            label="Feature regression loss")
    ax.set_xlabel("Stage 2 epoch", labelpad=2)
    ax.set_ylabel("Loss", labelpad=2)
    ax.set_xlim(1, ep[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.tick_params(length=2, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False,
              borderaxespad=0, handlelength=1.8)
    _save(fig, "Figure3")
    plt.close(fig)


# =====================================================================
# Figure 4 — CAR diagnostic (computation verbatim from 03_car_routing)
# =====================================================================
def compute_fig4():
    from tsr_distill import ContextAwareRouter

    torch.manual_seed(0)
    np.random.seed(0)

    B = 600
    d_stu = 16
    d_phi = 4
    L_h = 3
    N = 4

    mode = np.random.randint(0, 2, B)
    phi = np.zeros((B, d_phi), dtype=np.float32)
    phi[mode == 0] = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32) \
        + 0.1 * np.random.randn((mode == 0).sum(), d_phi).astype(np.float32)
    phi[mode == 1] = np.array([0.2, 1.0, 1.0, 0.2], dtype=np.float32) \
        + 0.1 * np.random.randn((mode == 1).sum(), d_phi).astype(np.float32)

    stu = np.random.randn(B, d_stu).astype(np.float32) * 0.3
    sigma = np.random.randn(B, L_h).astype(np.float32) * 0.1 + 1.5

    target = np.zeros((B, N), dtype=np.float32)
    target[mode == 0] = np.array([0.45, 0.35, 0.10, 0.10], dtype=np.float32)
    target[mode == 1] = np.array([0.10, 0.10, 0.35, 0.45], dtype=np.float32)

    car = ContextAwareRouter(d_stu=d_stu, d_phi=d_phi,
                             num_levels=L_h, num_teachers=N, temperature=1.0)
    opt = torch.optim.Adam(car.parameters(), lr=5e-3)
    stu_t = torch.from_numpy(stu)
    phi_t = torch.from_numpy(phi)
    sig_t = torch.from_numpy(sigma)
    tgt_t = torch.from_numpy(target)
    for ep in range(200):
        w, _ = car(stu_t, phi_t, sig_t)
        loss = ((w - tgt_t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        w_final, _ = car(stu_t, phi_t, sig_t)
    W = w_final.numpy()
    return W, mode


def plot_fig4(W, mode):
    order = np.argsort(mode, kind="stable")
    Ws = W[order]
    nA = int((mode == 0).sum())
    B, N = W.shape

    fig = plt.figure(figsize=(7.0, 2.75), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.42,
                  left=0.075, right=0.965, top=0.82, bottom=0.165)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(Ws, aspect="auto", cmap="cividis",
                   extent=[-0.5, N - 0.5, B, 0])
    ax.axhline(nA, color="white", lw=1.4)
    ax.text(-0.115, 1.24, "(a)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    ax.set_title("Per-sample routing weights (sorted by mode)", pad=3)
    ax.set_xlabel("Teacher index", labelpad=1.5)
    ax.set_ylabel("Sample", labelpad=1.5)
    ax.set_xticks(range(N))
    ax.tick_params(length=2, pad=1.5)
    ax.text(N - 0.38, nA / 2, "Mode A", rotation=270, va="center", fontsize=7)
    ax.text(N - 0.38, nA + (B - nA) / 2, "Mode B", rotation=270, va="center",
            fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.10)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("Routing weight", fontsize=7, labelpad=2)
    cb.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5])
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.135, 1.24, "(b)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    x = np.arange(N); wbar = 0.36
    ax.bar(x - wbar / 2, W[mode == 0].mean(axis=0), wbar, color=BLUE,
           label="Mode A", edgecolor="white", lw=0.4)
    ax.bar(x + wbar / 2, W[mode == 1].mean(axis=0), wbar, color=TAN,
           label="Mode B", edgecolor="white", lw=0.4)
    ax.set_xlabel("Teacher index", labelpad=1.5)
    ax.set_ylabel("Mean routing weight", labelpad=1.5)
    ax.set_xticks(x)
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    ax.tick_params(length=2, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=False, borderaxespad=0, handlelength=1.4)
    _save(fig, "Figure4")
    plt.close(fig)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "2"):
        plot_fig2(*compute_fig2())
    if which in ("all", "3"):
        plot_fig3(*compute_fig3())
    if which in ("all", "4"):
        plot_fig4(*compute_fig4())
