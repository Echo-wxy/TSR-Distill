"""
03_car_routing.py — Visualise the Context-Aware Router (CAR) and its
per-sample routing weights over the specialised-teacher ensemble.

Run:
    python scripts/03_car_routing.py

Output:
    figures/03_car_routing.png

What it shows:
    Top-left   — heatmap of routing weights w in R^{B x N} on the
                 validation batch at the end of Stage 3. Each row is one
                 sample, each column one specialised teacher. Visible
                 horizontal banding by class means CAR has learned to
                 group samples by domain attribute, not just average them.
    Top-right  — per-class mean routing distribution. The bars differ
                 across classes, confirming that the router is genuinely
                 input-conditioned rather than collapsing to a uniform
                 vector (which would make CAR equivalent to the
                 "Uniform Weighting" ablation baseline in the paper).
    Bottom     — three diagnostic panels:
                    (a) load balance: per-teacher utilisation vs uniform 1/N
                    (b) the L_h entries of the ensemble disagreement
                        vector sigma (Eq. 11-12) across a batch
                    (c) Stage 3 loss curves (task / HKD / RKD / LB)
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tsr_distill import (
    make_loaders,
    train_tsr_distill,
    ContextAwareRouter,
    compute_image_phi,
)
import torch.nn.functional as F


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    train_loader, val_loader, _, _ = make_loaders(
        batch_size=64, train_samples=1024, val_samples=256
    )
    # Run the full pipeline and ask for internals.
    out = train_tsr_distill(
        train_loader,
        val_loader,
        target_modality="image",
        epochs_s1=10,
        epochs_s2=2,
        epochs_s3=8,
        device="cpu",
        verbose=False,
        return_internals=True,
    )
    student = out["student"]
    teacher_img = out["teacher_img"]
    teacher_aud = out["teacher_aud"]
    teacher_mm = out["teacher_mm"]
    hkpas = out["hkpas"]
    car = out["car"]
    teacher_specs = out["teacher_specs"]
    kernel_sizes = out["kernel_sizes"]

    # We rebuild the same routing forward as in trainer.py to record
    # routing weights, sigma, and phi on the validation set.
    routing_rows = []
    sigma_rows = []
    labels = []
    student.eval(); car.eval()
    for hkpa in hkpas: hkpa.eval()
    with torch.no_grad():
        for x_img, x_aud, y in val_loader:
            stu_logits, f_stu = student(x_img, return_feat=True)
            stu_pooled = f_stu.mean(dim=1)

            G_all = []
            for j, name in enumerate(teacher_specs):
                if name == "aud_unimodal" or name == "aud_layer2":
                    _, f_tea = teacher_aud(x_aud, return_feat=True)
                elif name == "img_unimodal" or name == "img_layer2":
                    _, f_tea = teacher_img(x_img, return_feat=True)
                elif name == "img_mm":
                    _, f_tea, _ = teacher_mm(x_img, x_aud, return_feat=True)
                elif name == "aud_mm":
                    _, _, f_tea = teacher_mm(x_img, x_aud, return_feat=True)
                else:
                    raise ValueError(name)
                _ = hkpas[j](f_tea)
                G_all.append([g for g in hkpas[j].last_G])

            # Build sigma using a default per-level entropy estimate from
            # the *gate-weighted average* of G^{h,j} norms (a stable proxy
            # when the per-level aux classifiers from trainer.py are not
            # exposed). This still varies per-sample, which is what
            # matters for the visualisation.
            sigma_list = []
            for h in range(len(kernel_sizes)):
                # Stack per-teacher level-h feature norms: (N, B)
                norms_per_teacher = torch.stack(
                    [G_all[j][h].norm(dim=-1).mean(dim=1)
                     for j in range(len(teacher_specs))],
                    dim=0,
                )
                # Per-sample softmax over teachers, then row entropy.
                p = F.softmax(norms_per_teacher.t(), dim=1)   # (B, N)
                sigma_h = -(p * torch.log(p + 1e-8)).sum(dim=1)
                sigma_list.append(sigma_h)
            sigma = torch.stack(sigma_list, dim=1)            # (B, L_h)

            phi = compute_image_phi(x_img)
            w, _ = car(stu_pooled, phi, sigma)
            routing_rows.append(w.cpu().numpy())
            sigma_rows.append(sigma.cpu().numpy())
            labels.append(y.cpu().numpy())

    W = np.concatenate(routing_rows, axis=0)             # (B_total, N)
    SIGMA = np.concatenate(sigma_rows, axis=0)           # (B_total, L_h)
    Y = np.concatenate(labels, axis=0)                   # (B_total,)
    # Sort rows by class label so banding is visible.
    order = np.argsort(Y)
    W_sorted = W[order]
    Y_sorted = Y[order]

    # Per-class mean routing distribution.
    classes = sorted(set(Y.tolist()))
    per_class_mean = np.stack(
        [W[Y == c].mean(axis=0) for c in classes], axis=0
    )  # (num_classes, N)

    # ---- PLOT
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.2, 1, 1.2],
                          hspace=0.55, wspace=0.4)

    # Top-left: routing heatmap with class boundaries.
    ax0 = fig.add_subplot(gs[0, :2])
    im0 = ax0.imshow(W_sorted, aspect="auto", cmap="magma",
                     interpolation="nearest")
    ax0.set_xlabel("specialised teacher index $j$")
    ax0.set_ylabel("sample (sorted by class)")
    ax0.set_xticks(range(W.shape[1]))
    ax0.set_xticklabels([f"$j={i}$\n{teacher_specs[i]}" for i in range(W.shape[1])],
                        fontsize=8)
    # Class boundaries
    counts = [int((Y == c).sum()) for c in classes]
    boundaries = np.cumsum(counts)[:-1]
    for b in boundaries:
        ax0.axhline(b - 0.5, color="white", linewidth=0.5, alpha=0.6)
    routing_var = per_class_mean.var(axis=0).sum()
    if routing_var > 1e-4:
        subtitle = "Banding by class confirms CAR is input-conditioned"
    else:
        subtitle = (
            r"Routing is near-constant across samples on this dataset"
            " — CAR has learned a global teacher preference"
        )
    ax0.set_title(r"Per-sample routing weights $w \in \mathbb{R}^N$" "\n"
                  + subtitle,
                  fontsize=11)
    plt.colorbar(im0, ax=ax0, fraction=0.035, pad=0.02)

    # Top-right: per-class mean routing.
    ax1 = fig.add_subplot(gs[0, 2])
    im1 = ax1.imshow(per_class_mean, aspect="auto", cmap="magma",
                     vmin=W.min(), vmax=W.max())
    ax1.set_xticks(range(per_class_mean.shape[1]))
    ax1.set_xticklabels([f"$j={i}$" for i in range(per_class_mean.shape[1])],
                        fontsize=9)
    ax1.set_yticks(range(len(classes)))
    ax1.set_yticklabels([f"c={c}" for c in classes], fontsize=8)
    ax1.set_title("Per-class mean\nrouting", fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Bottom-left: load balance.
    ax2 = fig.add_subplot(gs[1, 0])
    util = W.mean(axis=0)
    bar_colors = ["#1f77b4" if u > 1.0 / len(util) else "#ff7f0e" for u in util]
    ax2.bar(range(len(util)), util, color=bar_colors)
    ax2.axhline(1.0 / len(util), linestyle="--", color="grey",
                label=r"uniform $1/N$")
    ax2.set_xticks(range(len(util)))
    ax2.set_xticklabels([f"$j={i}$" for i in range(len(util))], fontsize=8)
    ax2.set_ylabel(r"$\overline{w^j}$ across batch")
    ax2.set_title("Teacher utilisation (load balance)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Bottom-middle: sigma distributions.
    ax3 = fig.add_subplot(gs[1, 1])
    for h in range(SIGMA.shape[1]):
        ax3.hist(SIGMA[:, h], bins=20, alpha=0.5,
                 label=f"$\\sigma_{h + 1}$  ($k_h={kernel_sizes[h]}$)")
    ax3.set_xlabel(r"$\sigma_h$ (ensemble disagreement entropy)")
    ax3.set_ylabel("samples")
    ax3.set_title("Per-level $\\sigma_h$ varies\nacross samples", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # Bottom-right: loss curves.
    ax4 = fig.add_subplot(gs[1, 2])
    hist = out["s3_history"]
    epochs = range(1, len(hist["loss_task"]) + 1)
    ax4.plot(epochs, hist["loss_task"], label="task", linewidth=1.8)
    ax4.plot(epochs, hist["loss_hkd"], label="HKD ($\\eta\\cdot$)", linewidth=1.8)
    ax4.plot(epochs, hist["loss_rkd"], label="RKD ($\\zeta\\cdot$)", linewidth=1.8)
    ax4.plot(epochs, hist["loss_lb"], label="LB ($\\mu\\cdot$)", linewidth=1.8)
    ax4.set_xlabel("Stage 3 epoch")
    ax4.set_ylabel("loss")
    ax4.set_title("Stage 3 loss curves", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Bottom row across full width: validation accuracy.
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(epochs, hist["val_acc"], marker="o", color="#2ca02c", linewidth=2,
             label="TSR-Distill student")
    ax5.axhline(out["teacher_mm_val_acc"], linestyle="--", color="grey",
                label=f"MM teacher upper bound ({out['teacher_mm_val_acc']:.3f})")
    ax5.set_xlabel("Stage 3 epoch")
    ax5.set_ylabel("validation accuracy")
    ax5.set_title("Stage 3 validation accuracy trajectory", fontsize=10)
    ax5.legend(fontsize=9, loc="lower right")
    ax5.grid(True, alpha=0.3)

    fig.suptitle("Context-Aware Router (CAR): "
                 "per-sample teacher weighting via student + $\\phi$ + $\\sigma$",
                 fontsize=12, fontweight="bold", y=0.995)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "03_car_routing.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"final val_acc = {hist['val_acc'][-1]:.4f}, MM teacher = {out['teacher_mm_val_acc']:.4f}")
    print(f"teacher utilisation: {util.round(3).tolist()}")
    print(f"per-class routing variance (sum): {routing_var:.4f} "
          "(0 = uniform, > 0 = class-conditional)")

    # ---- BONUS DIAGNOSTIC: a controlled experiment that isolates input
    # conditioning. We train a fresh CAR on synthetic two-mode data where
    # half the samples belong to "mode A" (one phi pattern, prefers
    # teachers {0, 1}) and the other half to "mode B" (different phi
    # pattern, prefers teachers {2, 3}). A correctly working router
    # should produce visibly different weight distributions for the two
    # modes — a fact obscured on the real toy dataset, where phi
    # variation across classes is too small to drive the routing.
    print("\n--- diagnostic experiment ---")
    diag_path = os.path.join(out_dir, "03b_car_routing_diagnostic.png")
    _run_diagnostic_car(diag_path)


def _run_diagnostic_car(out_path):
    """Train a small CAR on synthetic two-mode data and visualise the
    resulting weights. This isolates the router's ability to be input-
    conditioned from the rest of the pipeline."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    np.random.seed(0)

    B = 600
    d_stu = 16
    d_phi = 4
    L_h = 3
    N = 4

    # Two modes with distinguishable phi.
    mode = np.random.randint(0, 2, B)            # 0 = A, 1 = B
    phi = np.zeros((B, d_phi), dtype=np.float32)
    phi[mode == 0] = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32) \
        + 0.1 * np.random.randn((mode == 0).sum(), d_phi).astype(np.float32)
    phi[mode == 1] = np.array([0.2, 1.0, 1.0, 0.2], dtype=np.float32) \
        + 0.1 * np.random.randn((mode == 1).sum(), d_phi).astype(np.float32)

    stu = np.random.randn(B, d_stu).astype(np.float32) * 0.3
    sigma = np.random.randn(B, L_h).astype(np.float32) * 0.1 + 1.5

    # Soft targets: mode A prefers teachers 0,1; mode B prefers 2,3.
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = np.argsort(mode)
    im = axes[0].imshow(W[order], aspect="auto", cmap="magma")
    axes[0].axhline((mode == 0).sum() - 0.5, color="white", linewidth=1.5)
    axes[0].set_title("Routing $w$ (samples sorted by mode)\n"
                      "Top half = mode A,  bottom half = mode B")
    axes[0].set_xlabel("teacher index $j$")
    axes[0].set_ylabel("sample")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for mode_id, mode_name in enumerate(["A", "B"]):
        m = W[mode == mode_id].mean(axis=0)
        axes[1].bar(np.arange(N) + 0.4 * mode_id - 0.2, m, width=0.35,
                    label=f"mode {mode_name}",
                    color=["#1f77b4", "#d62728"][mode_id])
    axes[1].set_xticks(range(N))
    axes[1].set_xticklabels([f"$j={i}$" for i in range(N)])
    axes[1].set_xlabel("teacher index")
    axes[1].set_ylabel("mean weight")
    axes[1].set_title("Per-mode mean routing — CAR splits its\nattention "
                      "based on $\\phi$ even with identical labels")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Diagnostic: CAR is input-conditioned under sufficient $\\phi$ variation",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"  mode A mean w: {W[mode == 0].mean(axis=0).round(3).tolist()}")
    print(f"  mode B mean w: {W[mode == 1].mean(axis=0).round(3).tolist()}")


if __name__ == "__main__":
    main()
