"""
05_tsne_features.py — t-SNE feature visualisation and silhouette-score
comparison across three configurations (no KD, response KD, full
TSR-Distill).

Run:
    python scripts/05_tsne_features.py

Output:
    figures/05_tsne_features.png

What it shows:
    Three side-by-side t-SNE projections of the student's pooled feature
    on the validation set, one per configuration. The expected qualitative
    result (matching Fig. 6 in the paper) is that TSR-Distill produces
    visibly tighter intra-class clusters with more inter-class margin
    than either no-KD or vanilla KD.

    The silhouette score (sklearn) is annotated above each panel. It is
    a numerical proxy for cluster quality: higher values indicate
    samples that are more similar to their own class than to other
    classes.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tsr_distill import (
    make_loaders,
    train_unimodal_baseline,
    train_kd_baseline,
    train_tsr_distill,
    MultimodalTeacher,
)


def _extract_features(model, val_loader, target_modality):
    """Return (N, d) pooled features and (N,) labels."""
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x_img, x_aud, y in val_loader:
            x = x_img if target_modality == "image" else x_aud
            _, f = model(x, return_feat=True)            # (B, T, d)
            feats.append(f.mean(dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    train_loader, val_loader, _, _ = make_loaders(
        batch_size=64, train_samples=1024, val_samples=400, seed=0
    )

    # ---- Three configurations.
    print("training: baseline (no KD) ...")
    torch.manual_seed(0)
    base = train_unimodal_baseline(
        train_loader, val_loader, target_modality="image", epochs=12,
        device="cpu", verbose=False,
    )

    print("training: MM teacher + response-based KD ...")
    torch.manual_seed(0)
    tm = MultimodalTeacher()
    opt = torch.optim.Adam(tm.parameters(), lr=1e-3)
    for ep in range(12):
        for x_img, x_aud, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(tm(x_img, x_aud), y).backward()
            opt.step()
    tm.eval()
    torch.manual_seed(0)
    kd = train_kd_baseline(
        train_loader, val_loader, teacher=tm, target_modality="image",
        epochs=12, device="cpu", verbose=False,
    )

    print("training: TSR-Distill ...")
    torch.manual_seed(0)
    tsr = train_tsr_distill(
        train_loader, val_loader, target_modality="image",
        epochs_s1=12, epochs_s2=2, epochs_s3=8, device="cpu",
        verbose=False,
    )

    print(
        f"\nFinal val accuracies: "
        f"baseline={base['history']['val_acc'][-1]:.4f}  "
        f"KD={kd['history']['val_acc'][-1]:.4f}  "
        f"TSR={tsr['final_val_acc']:.4f}"
    )

    # ---- Extract features and run t-SNE.
    configs = [
        ("No KD", base["student"]),
        ("Response KD", kd["student"]),
        ("TSR-Distill (Ours)", tsr["student"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    panel_colors = plt.cm.tab10(np.arange(10))

    for ax, (name, model) in zip(axes, configs):
        feats, labels = _extract_features(model, val_loader, "image")
        # Silhouette on the raw pooled feature (more meaningful than on
        # t-SNE coordinates, which already optimise for separation).
        sil = silhouette_score(feats, labels, metric="euclidean")
        # sklearn 1.5+ renamed `n_iter` -> `max_iter`; support both.
        import inspect
        tsne_kwargs = dict(
            n_components=2, perplexity=30, init="pca", random_state=0,
        )
        sig = inspect.signature(TSNE.__init__)
        if "max_iter" in sig.parameters:
            tsne_kwargs["max_iter"] = 600
        else:
            tsne_kwargs["n_iter"] = 600
        emb = TSNE(**tsne_kwargs).fit_transform(feats)
        for c in range(10):
            m = labels == c
            ax.scatter(
                emb[m, 0], emb[m, 1], s=18, alpha=0.7,
                color=panel_colors[c], label=f"{c}" if name == configs[0][0] else None,
                edgecolor="white", linewidth=0.4,
            )
        ax.set_title(f"{name}\nsilhouette = {sil:.3f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].legend(loc="lower left", title="class", fontsize=7, ncol=2,
                   bbox_to_anchor=(0.0, -0.15))

    fig.suptitle(
        "t-SNE of student pooled features on the validation set\n"
        "Silhouette score quantifies intra-class compactness + "
        "inter-class margin (higher = better)",
        fontsize=12, fontweight="bold",
    )
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "05_tsne_features.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
