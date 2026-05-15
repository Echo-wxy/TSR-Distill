"""
00_quickstart.py — End-to-end TSR-Distill training, plus comparison
against (a) the no-distillation baseline and (b) the standard
response-based KD baseline. The fastest single script to verify that
the codebase runs correctly on a new machine.

Run:
    python scripts/00_quickstart.py

Output:
    figures/00_quickstart.png

What it shows:
    Left panel  — validation-accuracy learning curves for the three
                  methods, with the MM teacher upper bound marked as a
                  horizontal dashed line.
    Right panel — bar chart of the three final accuracies plus the MM
                  teacher reference. The "gap recovery" annotation
                  reports the fraction of the teacher-student gap that
                  each KD method recovers.

Typical wall-clock on a single CPU core: ~2 minutes.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tsr_distill import (
    make_loaders,
    train_unimodal_baseline,
    train_kd_baseline,
    train_tsr_distill,
    MultimodalTeacher,
)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    train_loader, val_loader, _, _ = make_loaders(
        batch_size=64, train_samples=1024, val_samples=256, seed=0
    )

    # MM teacher (upper bound reference).
    print("training MM teacher (upper bound) ...")
    torch.manual_seed(0)
    tm = MultimodalTeacher()
    opt = torch.optim.Adam(tm.parameters(), lr=1e-3)
    history_mm = []
    for ep in range(15):
        tm.train()
        for x_img, x_aud, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(tm(x_img, x_aud), y).backward()
            opt.step()
        # Per-epoch validation.
        tm.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_img, x_aud, y in val_loader:
                correct += (tm(x_img, x_aud).argmax(dim=1) == y).sum().item()
                total += y.size(0)
        history_mm.append(correct / total)
    print(f"  MM teacher final accuracy: {history_mm[-1]:.4f}")

    # No-KD baseline.
    print("training: no-KD baseline ...")
    torch.manual_seed(0)
    base = train_unimodal_baseline(
        train_loader, val_loader, target_modality="image", epochs=12,
        device="cpu", verbose=False,
    )
    print(f"  baseline final accuracy: {base['history']['val_acc'][-1]:.4f}")

    # Response-based KD.
    print("training: response-based KD ...")
    torch.manual_seed(0)
    kd = train_kd_baseline(
        train_loader, val_loader, teacher=tm, target_modality="image",
        epochs=12, device="cpu", verbose=False,
    )
    print(f"  KD final accuracy: {kd['history']['val_acc'][-1]:.4f}")

    # Full TSR-Distill.
    print("training: TSR-Distill (full 3-stage pipeline) ...")
    torch.manual_seed(0)
    tsr = train_tsr_distill(
        train_loader, val_loader, target_modality="image",
        epochs_s1=12, epochs_s2=2, epochs_s3=8, device="cpu", verbose=False,
    )
    print(f"  TSR-Distill final accuracy: {tsr['final_val_acc']:.4f}")

    acc_mm = tsr["teacher_mm_val_acc"]
    acc_base = base["history"]["val_acc"][-1]
    acc_kd = kd["history"]["val_acc"][-1]
    acc_tsr = tsr["final_val_acc"]

    # ---- PLOT
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # Learning curves.
    ax0 = axes[0]
    ax0.plot(range(1, len(base["history"]["val_acc"]) + 1),
             base["history"]["val_acc"], marker="o",
             label="No KD (baseline)", linewidth=2, color="#7f7f7f")
    ax0.plot(range(1, len(kd["history"]["val_acc"]) + 1),
             kd["history"]["val_acc"], marker="s",
             label="Response KD", linewidth=2, color="#1f77b4")
    s3_history = tsr["s3_history"]["val_acc"]
    # TSR Stage 3 epochs follow on the same x-axis for comparison.
    ax0.plot(range(1, len(s3_history) + 1), s3_history,
             marker="^", label="TSR-Distill (Stage 3 epochs)",
             linewidth=2.5, color="#2ca02c")
    ax0.axhline(acc_mm, linestyle="--", color="#d62728",
                label=f"MM teacher upper bound ({acc_mm:.3f})", alpha=0.7)
    ax0.set_xlabel("epoch (within the configuration's primary training phase)")
    ax0.set_ylabel("validation accuracy")
    ax0.set_title("Validation accuracy across configurations", fontsize=11)
    ax0.legend(loc="lower right", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # Bar comparison.
    ax1 = axes[1]
    bars = ["baseline", "KD", "TSR-Distill", "MM teacher"]
    vals = [acc_base, acc_kd, acc_tsr, acc_mm]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    b = ax1.bar(bars, vals, color=colors, edgecolor="black")
    for bar, v in zip(b, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    # Gap-recovery annotation.
    if acc_mm > acc_base:
        rec_kd = (acc_kd - acc_base) / (acc_mm - acc_base) * 100
        rec_tsr = (acc_tsr - acc_base) / (acc_mm - acc_base) * 100
        ax1.set_xlabel(
            f"gap recovery rate:\n"
            f"  KD: {rec_kd:+.1f}%   TSR-Distill: {rec_tsr:+.1f}%"
        )
    ax1.set_ylabel("validation accuracy")
    ax1.set_title("Final accuracy comparison", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(min(vals) - 0.05, acc_mm + 0.08)

    fig.suptitle("TSR-Distill quickstart — synthetic AV-MNIST-style dataset",
                 fontsize=12, fontweight="bold")
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "00_quickstart.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
