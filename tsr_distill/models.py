"""
Network components used in TSR-Distill.

All backbones here are intentionally lightweight (each well below 1M
parameters) so that the full three-stage pipeline trains in minutes on a
single CPU core or a modest GPU. They are functional analogues of the
LeNet / ResNet-style architectures used in the paper's experiments,
preserving the structural property that matters for distillation: an
intermediate feature with a temporal axis $T$ and a channel axis $d$.

Naming convention for intermediate features:
    Z_l       : the layer-$l$ feature exposed by a teacher / student.
                Shape: (B, T, d) where T is the sequence length and d is
                the channel dimension. Image teachers reshape spatial
                features into a (B, H*W, C) sequence to apply the same
                temporal-style alignment uniformly across modalities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base unimodal teachers (also reused as student backbones)
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """A small CNN for 1-channel images.

    The encoder exposes a sequence-form intermediate feature `Z_l` of shape
    (B, T_img, d_z) where T_img = H' * W' after spatial pooling, plus a
    classification logit.
    """

    def __init__(self, num_classes=10, d_z=64, img_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, d_z, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        # After three 2x downsamplings: img_size // 8
        self.spatial = max(1, img_size // 8)
        self.fc = nn.Linear(d_z * self.spatial * self.spatial, num_classes)
        self.d_z = d_z

    def forward(self, x, return_feat=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))            # (B, d_z, h, w)
        b, c, h, w = x.shape
        feat = x.flatten(2).transpose(1, 2)             # (B, T=h*w, d_z)
        logit = self.fc(x.flatten(1))
        if return_feat:
            return logit, feat
        return logit


class AudioEncoder(nn.Module):
    """A small CNN for 1-channel log-Mel spectrograms.

    The temporal axis $T$ in the exposed feature corresponds to the
    spectrogram's time dimension after spatial pooling. The frequency axis
    is folded into the channel dimension.
    """

    def __init__(self, num_classes=10, d_z=64, audio_t=32, audio_f=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, d_z, 3, padding=1)
        self.pool_f = nn.AvgPool2d((2, 1))  # only pool along frequency
        self.pool_t = nn.AvgPool2d((1, 2))  # only pool along time
        # After two (2,1) freq-poolings and two (1,2) time-poolings:
        self.t_out = max(1, audio_t // 4)
        self.f_out = max(1, audio_f // 4)
        self.fc = nn.Linear(d_z * self.t_out * self.f_out, num_classes)
        self.d_z = d_z

    def forward(self, x, return_feat=False):
        x = self.pool_t(self.pool_f(F.relu(self.conv1(x))))
        x = self.pool_t(self.pool_f(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))                       # (B, d_z, f, t)
        # Collapse frequency into a channel-like dim, keep time as T.
        b, c, f, t = x.shape
        feat = x.mean(dim=2).transpose(1, 2)            # (B, T=t, d_z)
        logit = self.fc(x.flatten(1))
        if return_feat:
            return logit, feat
        return logit


class MultimodalTeacher(nn.Module):
    """Fusion teacher combining the image and audio encoders.

    Mid-level fusion: features from each modality are pooled over their
    sequence axis and concatenated, then passed through a small MLP head.
    The fusion teacher serves as the upper-bound reference in our gap
    recovery analysis (Table V of the paper).
    """

    def __init__(self, num_classes=10, d_z=64, img_size=28, audio_t=32, audio_f=32):
        super().__init__()
        self.img = ImageEncoder(num_classes, d_z, img_size)
        self.aud = AudioEncoder(num_classes, d_z, audio_t, audio_f)
        self.head = nn.Sequential(
            nn.Linear(2 * d_z, 2 * d_z),
            nn.ReLU(inplace=True),
            nn.Linear(2 * d_z, num_classes),
        )

    def forward(self, x_img, x_aud, return_feat=False):
        _, f_img = self.img(x_img, return_feat=True)
        _, f_aud = self.aud(x_aud, return_feat=True)
        pooled = torch.cat([f_img.mean(dim=1), f_aud.mean(dim=1)], dim=1)
        logit = self.head(pooled)
        if return_feat:
            return logit, f_img, f_aud
        return logit


# ---------------------------------------------------------------------------
# Stage 2: Hierarchical Knowledge Projection Adapter (HKPA)
# ---------------------------------------------------------------------------

class HKPA(nn.Module):
    """Hierarchical Knowledge Projection Adapter.

    Implements Eq.(6)-(7) of the paper:
        G^h = GN(ReLU(Conv1D_{k_h}(Z_i^l)))          [Eq. 6]
        Z_tilde = Proj(sum_h g_h * G^h)              [Eq. 7]

    The adapter takes a teacher feature `Z` of shape (B, T_i, d_tea), applies
    L_h parallel 1D convolutions with different kernel sizes to extract
    multi-granularity hierarchical features `G^h`, fuses them by adaptive
    softmax gating, and finally re-projects to the student's (T_t, d_stu).

    The intermediate features `G^h` are stored on the module after the
    forward pass so that Stage 3's hierarchical distillation loss can
    reuse them without an extra forward.
    """

    def __init__(
        self,
        d_in,
        d_mid=64,
        d_out=64,
        target_T=49,
        kernel_sizes=(3, 7, 15),
        num_groups=8,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.target_T = target_T
        self.d_mid = d_mid

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(d_in, d_mid, kernel_size=k, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.GroupNorm(num_groups, d_mid),
                )
            )
        # Learnable gate logits, initialised to zero so the softmax is uniform.
        self.gate_logits = nn.Parameter(torch.zeros(len(kernel_sizes)))
        # 1x1 conv to map d_mid -> d_out
        self.proj = nn.Conv1d(d_mid, d_out, kernel_size=1)

        # Buffers to expose intermediate features for Stage 3.
        self.last_G = None        # list of (B, T_t, d_mid) tensors
        self.last_gate = None     # (L_h,) softmax weights

    def forward(self, Z):
        """Project a teacher feature into the student-compatible space.

        Parameters
        ----------
        Z : Tensor of shape (B, T_i, d_in)
            Teacher intermediate feature.

        Returns
        -------
        Z_tilde : Tensor of shape (B, T_t, d_out)
            Student-compatible projected feature.
        """
        # Conv1d expects (B, C, T)
        x = Z.transpose(1, 2)                             # (B, d_in, T_i)
        gate = F.softmax(self.gate_logits, dim=0)
        self.last_gate = gate.detach()
        self.last_G = []

        # Hierarchical abstraction at multiple kernel sizes.
        fused = 0
        for h, branch in enumerate(self.branches):
            g = branch(x)                                 # (B, d_mid, T_i)
            # Resample temporal axis to T_t via adaptive average pooling /
            # linear interpolation, matching the paper's Proj operator.
            g_t = F.adaptive_avg_pool1d(g, self.target_T) # (B, d_mid, T_t)
            self.last_G.append(g_t.transpose(1, 2))       # store as (B, T_t, d_mid)
            fused = fused + gate[h] * g_t

        # 1x1 channel projection.
        out = self.proj(fused)                            # (B, d_out, T_t)
        return out.transpose(1, 2)                        # (B, T_t, d_out)


class AuxClassifier(nn.Module):
    """A small MLP used as the auxiliary domain-aware classifier.

    Used both inside Stage 2 (Eq. 9, the $\\ell_{DA}$ regulariser) and inside
    Stage 3's CAR (Eq. 12, the per-level disagreement probability $p_j^h$).
    Architectures are independent across hierarchical levels.
    """

    def __init__(self, d_in, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Stage 3: Context-Aware Router (CAR)
# ---------------------------------------------------------------------------

class ContextAwareRouter(nn.Module):
    """Sample-level soft router over N specialised teachers (Eq. 10-13).

    The router consumes three context signals:
        (1) Pool(Z_t^{l'})    --- pooled student feature (d_stu)
        (2) phi(X_t)           --- non-learnable input statistics (d_phi)
        (3) sigma              --- ensemble disagreement entropy per HKPA level (L_h)
    and emits a softmax weight w in R^N over the specialised teachers.

    The softmax `temperature` controls how peaked the routing distribution
    is allowed to become. A temperature > 1 keeps the weights soft and
    prevents the common failure mode where the router collapses to a
    single teacher under joint optimisation with the student.
    """

    def __init__(self, d_stu, d_phi, num_levels, num_teachers, hidden=256,
                 temperature=2.0):
        super().__init__()
        d_ctx = d_stu + d_phi + num_levels
        self.mlp = nn.Sequential(
            nn.Linear(d_ctx, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_teachers),
        )
        self.temperature = temperature

    def forward(self, student_pooled, phi, sigma):
        ctx = torch.cat([student_pooled, phi, sigma], dim=1)
        logits = self.mlp(ctx)
        w = F.softmax(logits / self.temperature, dim=1)
        return w, logits


def compute_image_phi(x_img):
    """Non-learnable image descriptor: per-channel mean/var + Sobel edge density.

    Matches the paper's image branch of phi(X_t) (Section III-C).
    Returns a tensor of shape (B, 7) for 3-channel input or (B, 3) for 1-channel.
    For our synthetic 1-channel images we use mean, var, and edge density.
    """
    b = x_img.size(0)
    mean = x_img.mean(dim=(2, 3))
    var = x_img.var(dim=(2, 3))
    # Sobel-like edge density via simple finite differences.
    gx = (x_img[:, :, :, 1:] - x_img[:, :, :, :-1]).abs().mean(dim=(2, 3))
    gy = (x_img[:, :, 1:, :] - x_img[:, :, :-1, :]).abs().mean(dim=(2, 3))
    edge = gx + gy
    return torch.cat([mean, var, edge], dim=1)  # (B, 3 * num_channels)


def compute_audio_phi(x_aud):
    """Non-learnable audio descriptor: spectrum mean / var per frequency bin
    plus spectral centroid. Matches the paper's audio branch of phi(X_t).
    """
    b = x_aud.size(0)
    # Average over time -> per-frequency profile (B, C, F).
    profile = x_aud.mean(dim=3)                           # (B, 1, F)
    spec_mean = profile.squeeze(1)                        # (B, F)
    spec_var = x_aud.var(dim=3).squeeze(1)                # (B, F)
    # Spectral centroid: weighted average frequency index, normalised to [0,1].
    f = x_aud.size(2)
    freq_axis = torch.linspace(0, 1, f, device=x_aud.device).view(1, 1, f)
    energy = x_aud.mean(dim=3)                            # (B, 1, F)
    centroid = (energy * freq_axis).sum(dim=2) / (energy.sum(dim=2) + 1e-8)
    return torch.cat([spec_mean, spec_var, centroid], dim=1)


def get_phi_dim(modality, audio_f=32, img_channels=1):
    if modality == "image":
        return 3 * img_channels
    if modality == "audio":
        return 2 * audio_f + 1
    raise ValueError(modality)
