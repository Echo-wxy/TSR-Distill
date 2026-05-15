"""
Synthetic multimodal dataset for reproducible experiments.

To keep the codebase self-contained and runnable on any machine, we provide a
synthetic image-audio classification dataset that mimics the structural
properties of AV-MNIST. Each sample is a (image, audio_spectrogram, label)
triple where the two modalities share a common latent class but are corrupted
by independent noise. This setup preserves the key property exploited by
TSR-Distill: complementary cross-modal evidence under controllable difficulty.

For users who wish to run the framework on real benchmarks, drop-in adapters
for AV-MNIST, RAVDESS, VGGSound-50k, CrisisMMD-V2 and NYU-Depth-V2 can be
written following the same `(x_image, x_audio, y)` interface.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def _make_class_prototypes(num_classes, img_size, audio_t, audio_f, rng):
    """Create per-class prototype tensors for both modalities.

    To ensure that the multimodal teacher genuinely outperforms each
    unimodal model (so that cross-modal distillation has something
    meaningful to transfer), we factorise the class label into two
    orthogonal latent attributes:

        class id  c  =  i_img * K_aud + i_aud,    0 <= c < K_img * K_aud

    The image prototype depends only on `i_img` and the audio prototype
    depends only on `i_aud`. Therefore:
        - looking only at the image, the model can at best identify `i_img`
          (and hence collapse `K_aud` classes into the same group),
        - looking only at the audio, the model can at best identify `i_aud`,
        - looking at both modalities, the full class label is recoverable.
    This factorisation mirrors the structure of AV-MNIST in the limit
    where audio carries digit identity but its style overlaps across digits.

    For `num_classes = 10` we use a 5 x 2 factorisation (K_img=5, K_aud=2),
    so the chance ceiling on either modality alone is 50%.
    """
    # Pick factorisation factors closest to a 5 x 2 layout for the default
    # num_classes=10. For other counts we fall back to a roughly balanced
    # rectangle factorisation.
    K_img = max(2, int(round(num_classes ** 0.5)))
    while num_classes % K_img != 0 and K_img > 1:
        K_img -= 1
    K_aud = num_classes // K_img
    # Slight preference for K_img > K_aud (image is usually the stronger
    # modality in AV-MNIST-style benchmarks).
    if K_img < K_aud:
        K_img, K_aud = K_aud, K_img

    img_protos = np.zeros((num_classes, 1, img_size, img_size), dtype=np.float32)
    aud_protos = np.zeros((num_classes, 1, audio_f, audio_t), dtype=np.float32)

    xs = np.linspace(-1.0, 1.0, img_size)
    ys = np.linspace(-1.0, 1.0, img_size)
    xx, yy = np.meshgrid(xs, ys)

    ts = np.linspace(0.0, 1.0, audio_t)
    fs = np.linspace(0.0, 1.0, audio_f)
    tt, ff = np.meshgrid(ts, fs)

    # Build K_img distinct image patterns and K_aud distinct audio patterns.
    img_patterns = []
    for i in range(K_img):
        cx = np.cos(2 * np.pi * i / K_img) * 0.6
        cy = np.sin(2 * np.pi * i / K_img) * 0.6
        img_patterns.append(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.15))

    aud_patterns = []
    for j in range(K_aud):
        freq = 1.5 + 2.0 * j
        aud_patterns.append(
            np.sin(2 * np.pi * freq * tt) * np.exp(-((ff - 0.5) ** 2) / 0.08)
        )

    for c in range(num_classes):
        i_img = c // K_aud
        i_aud = c % K_aud
        img_protos[c, 0] = img_patterns[i_img]
        aud_protos[c, 0] = aud_patterns[i_aud]

    return img_protos, aud_protos


class SyntheticAVDataset(Dataset):
    """A lightweight synthetic image-audio dataset.

    Parameters
    ----------
    num_samples : int
        Total number of samples to generate.
    num_classes : int
        Number of class labels (default 10, matching AV-MNIST).
    img_size : int
        Spatial size of the image modality.
    audio_t : int
        Number of audio time steps (treated as the temporal axis $T$).
    audio_f : int
        Number of audio frequency bins.
    noise_image : float
        Standard deviation of additive Gaussian noise on the image.
    noise_audio : float
        Standard deviation of additive Gaussian noise on the spectrogram.
    seed : int
        Random seed for full reproducibility.
    """

    def __init__(
        self,
        num_samples=2000,
        num_classes=10,
        img_size=28,
        audio_t=32,
        audio_f=32,
        noise_image=0.5,
        noise_audio=0.5,
        seed=0,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.audio_t = audio_t
        self.audio_f = audio_f

        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=num_samples).astype(np.int64)

        img_protos, aud_protos = _make_class_prototypes(
            num_classes, img_size, audio_t, audio_f, rng
        )

        # Build per-sample tensors by adding noise to the class prototype.
        self.images = (
            img_protos[self.labels]
            + rng.randn(num_samples, 1, img_size, img_size).astype(np.float32)
            * noise_image
        )
        self.audios = (
            aud_protos[self.labels]
            + rng.randn(num_samples, 1, audio_f, audio_t).astype(np.float32)
            * noise_audio
        )

        # Keep prototypes available for visualisation scripts.
        self.image_prototypes = img_protos
        self.audio_prototypes = aud_protos

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        aud = torch.from_numpy(self.audios[idx])
        y = int(self.labels[idx])
        return img, aud, y


def make_loaders(batch_size=64, train_samples=2000, val_samples=500, seed=0,
                 num_classes=10):
    """Convenience helper returning train and validation DataLoaders."""
    from torch.utils.data import DataLoader

    train_set = SyntheticAVDataset(
        num_samples=train_samples, num_classes=num_classes, seed=seed
    )
    val_set = SyntheticAVDataset(
        num_samples=val_samples, num_classes=num_classes, seed=seed + 1
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_set, val_set
