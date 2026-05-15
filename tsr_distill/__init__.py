"""TSR-Distill: Domain-Specific Cross-Modal Distillation via Temporal,
Structural, and Intrinsic-Aware Knowledge Transfer.

Reference implementation accompanying the IEEE Access manuscript.
"""

from .dataset import SyntheticAVDataset, make_loaders
from .losses import (
    cta_loss,
    icc_loss,
    soft_alignment_matrix,
    feature_regression_loss,
    domain_aware_loss,
    hierarchical_kd_loss,
    load_balancing_loss,
    kd_loss,
)
from .models import (
    ImageEncoder,
    AudioEncoder,
    MultimodalTeacher,
    HKPA,
    AuxClassifier,
    ContextAwareRouter,
    compute_image_phi,
    compute_audio_phi,
)
from .trainer import (
    train_unimodal_baseline,
    train_kd_baseline,
    train_tsr_distill,
)

__version__ = "1.0.0"

__all__ = [
    "SyntheticAVDataset",
    "make_loaders",
    "cta_loss",
    "icc_loss",
    "soft_alignment_matrix",
    "feature_regression_loss",
    "domain_aware_loss",
    "hierarchical_kd_loss",
    "load_balancing_loss",
    "kd_loss",
    "ImageEncoder",
    "AudioEncoder",
    "MultimodalTeacher",
    "HKPA",
    "AuxClassifier",
    "ContextAwareRouter",
    "compute_image_phi",
    "compute_audio_phi",
    "train_unimodal_baseline",
    "train_kd_baseline",
    "train_tsr_distill",
]
