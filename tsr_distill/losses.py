"""
Loss functions for the three stages of TSR-Distill.

Stage 1 (S1): Temporal-Intrinsic Collaborative Bootstrapping
    - cta_loss      : Eq. (1)-(2), cross-modal temporal alignment.
    - icc_loss      : Eq. (3)-(4), intrinsic consistency contrastive loss.

Stage 2 (S2): Hierarchical Knowledge Projection Adaptation
    - feature_regression_loss : Eq. (8), normalised MSE between projected
                                teacher feature and student feature.
    - domain_aware_loss       : Eq. (9), auxiliary classification regulariser.

Stage 3 (S3): Context-Aware Hierarchical Routed Distillation
    - hierarchical_kd_loss    : Eq. (14), router-weighted MSE across HKPA
                                levels between teacher G^{h,j} and projected
                                student feature Psi_h(Z_t^{l'}).
    - load_balancing_loss     : Symmetric KL between mean routing distribution
                                and uniform, as in MST-Distill, used to keep
                                router utilisation diverse.

All losses operate on per-sample tensors and return scalars unless stated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def _row_entropy(p, eps=1e-8):
    """Shannon entropy of each row of a probability matrix p of shape (..., K)."""
    return -(p * torch.log(p + eps)).sum(dim=-1)


def soft_alignment_matrix(z_i, z_j, kappa=0.1):
    """Directed soft-alignment matrix S^{i->j} (Eq. 1).

    Parameters
    ----------
    z_i : (B, T_i, d) tensor
    z_j : (B, T_j, d) tensor
    kappa : temperature

    Returns
    -------
    S : (B, T_i, T_j) tensor where S[b, t_i, :] is a probability over t_j.
    """
    # Cosine similarity between every (t_i, t_j) pair.
    z_i_n = F.normalize(z_i, dim=-1)
    z_j_n = F.normalize(z_j, dim=-1)
    sim = torch.bmm(z_i_n, z_j_n.transpose(1, 2)) / kappa
    return F.softmax(sim, dim=-1)


def cta_loss(z_i, z_j, kappa=0.1):
    """Cross-modal temporal alignment loss (Eq. 2).

    Encourages each row of S^{i->j} and S^{j->i} to be peaked, i.e. each
    position in one modality should align confidently to a small subset of
    positions in the other.

    The 1/2 factor and per-direction sequence-length normalisation give equal
    weight to both directions even when T_i != T_j, matching the paper.
    """
    s_ij = soft_alignment_matrix(z_i, z_j, kappa)         # (B, T_i, T_j)
    s_ji = soft_alignment_matrix(z_j, z_i, kappa)         # (B, T_j, T_i)
    h_ij = _row_entropy(s_ij).mean(dim=1)                 # (B,)
    h_ji = _row_entropy(s_ji).mean(dim=1)                 # (B,)
    return 0.5 * (h_ij + h_ji).mean()


def icc_loss(h_i, h_j, tau=0.07):
    """Intrinsic consistency contrastive loss (Eq. 3-4).

    SimCLR-style: pull together the two modalities of the same sample,
    push apart pairs from different samples. The denominator includes the
    self-pair term so the loss is well-defined even for B=1.

    Parameters
    ----------
    h_i, h_j : (B, d) clip-level pooled features from two modalities.
    tau : contrastive temperature.
    """
    h_i = F.normalize(h_i, dim=-1)
    h_j = F.normalize(h_j, dim=-1)
    logits_ij = torch.mm(h_i, h_j.t()) / tau              # (B, B)
    logits_ji = logits_ij.t()
    labels = torch.arange(h_i.size(0), device=h_i.device)
    loss_ij = F.cross_entropy(logits_ij, labels)
    loss_ji = F.cross_entropy(logits_ji, labels)
    return 0.5 * (loss_ij + loss_ji)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def feature_regression_loss(z_tilde, z_student, eps=1e-8):
    """Normalised MSE between projected teacher feature and student feature
    (Eq. 8). Both inputs are expected in shape (B, T_t, d_stu) and on the
    same device.
    """
    a = z_tilde / (z_tilde.norm(dim=(-2, -1), keepdim=True) + eps)
    b = z_student / (z_student.norm(dim=(-2, -1), keepdim=True) + eps)
    return ((a - b) ** 2).flatten(1).sum(dim=1).mean()


def domain_aware_loss(logits_aux, labels):
    """Cross-entropy on the auxiliary task (Eq. 9)."""
    return F.cross_entropy(logits_aux, labels)


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def hierarchical_kd_loss(G_per_teacher, student_projections, weights):
    """Router-weighted hierarchical distillation loss (Eq. 14).

    Parameters
    ----------
    G_per_teacher : list of length N (number of specialised teachers).
        Each element is itself a list of L_h tensors of shape
        (B, T_t, d_mid), i.e. the stored G^{h,j} from each HKPA.
    student_projections : list of L_h tensors of shape (B, T_t, d_mid),
        i.e. Psi_h(Z_t^{l'}) -- the per-level linear projection of the
        student's intermediate feature.
    weights : (B, N) tensor, router weights w^j.

    Returns
    -------
    Scalar tensor; lambda_h = 1/L_h uniform weighting across levels.
    """
    n = len(G_per_teacher)
    l_h = len(student_projections)
    lambda_h = 1.0 / l_h

    total = 0.0
    for j in range(n):
        w_j = weights[:, j].view(-1, 1, 1)                # (B, 1, 1)
        per_j = 0.0
        for h in range(l_h):
            diff = (G_per_teacher[j][h] - student_projections[h]) ** 2
            # Per-sample MSE, weighted by router weight w_j[b].
            per_j = per_j + lambda_h * (w_j * diff).mean(dim=(1, 2))
        total = total + per_j
    return total.mean()


def load_balancing_loss(w):
    """KL(mean(w) || uniform) load-balancing regulariser.

    Parameters
    ----------
    w : (B, N) router output. Each row already sums to 1.
    """
    mean_w = w.mean(dim=0)                                # (N,)
    n = mean_w.size(0)
    uniform = torch.full_like(mean_w, 1.0 / n)
    return F.kl_div(torch.log(mean_w + 1e-8), uniform, reduction="batchmean")


# ---------------------------------------------------------------------------
# Convenience: response-based KD (used as a baseline comparator)
# ---------------------------------------------------------------------------

def kd_loss(logits_student, logits_teacher, T=4.0):
    """Standard Hinton-style KD loss for comparison baselines."""
    log_p_s = F.log_softmax(logits_student / T, dim=1)
    p_t = F.softmax(logits_teacher.detach() / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
