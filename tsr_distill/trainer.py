"""
Training pipelines for the TSR-Distill framework and its baselines.

Three trainers are exposed:

    train_unimodal_baseline
        Trains a student from scratch on its own modality, no distillation.

    train_kd_baseline
        Trains a student from a frozen multimodal fusion teacher using the
        standard Hinton response-based KD loss.

    train_tsr_distill
        The full three-stage TSR-Distill pipeline.

All trainers return a dictionary containing per-epoch metrics and the
trained student state-dict, making it easy for visualisation scripts to
plot learning curves and inspect internals.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import (
    ImageEncoder,
    AudioEncoder,
    MultimodalTeacher,
    HKPA,
    AuxClassifier,
    ContextAwareRouter,
    compute_image_phi,
    compute_audio_phi,
    get_phi_dim,
)
from .losses import (
    cta_loss,
    icc_loss,
    feature_regression_loss,
    domain_aware_loss,
    hierarchical_kd_loss,
    load_balancing_loss,
    kd_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def _evaluate(model, loader, target_modality, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_img, x_aud, y in loader:
            x = (x_img if target_modality == "image" else x_aud).to(device)
            y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _evaluate_mm(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_img, x_aud, y in loader:
            logits = model(x_img.to(device), x_aud.to(device))
            correct += (logits.argmax(dim=1) == y.to(device)).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _build_student(target_modality, num_classes, img_size, audio_t, audio_f, d_z):
    if target_modality == "image":
        return ImageEncoder(num_classes, d_z, img_size)
    elif target_modality == "audio":
        return AudioEncoder(num_classes, d_z, audio_t, audio_f)
    raise ValueError(target_modality)


# ---------------------------------------------------------------------------
# Baseline 1: train a student from scratch (no KD)
# ---------------------------------------------------------------------------

def train_unimodal_baseline(
    train_loader,
    val_loader,
    target_modality="image",
    num_classes=10,
    img_size=28,
    audio_t=32,
    audio_f=32,
    d_z=64,
    epochs=10,
    lr=1e-3,
    device="cpu",
    verbose=True,
):
    student = _build_student(
        target_modality, num_classes, img_size, audio_t, audio_f, d_z
    ).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}

    for ep in range(epochs):
        student.train()
        total = 0.0
        for x_img, x_aud, y in train_loader:
            x = (x_img if target_modality == "image" else x_aud).to(device)
            y = y.to(device)
            logits = student(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg_loss = total / max(1, len(train_loader))
        acc = _evaluate(student, val_loader, target_modality, device)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)
        if verbose:
            print(
                f"[baseline/no-KD][{target_modality}] epoch {ep + 1}/{epochs} "
                f"loss={avg_loss:.4f} val_acc={acc:.4f}"
            )

    return {"student": student, "history": history}


# ---------------------------------------------------------------------------
# Baseline 2: response-based KD from a frozen multimodal teacher
# ---------------------------------------------------------------------------

def train_kd_baseline(
    train_loader,
    val_loader,
    teacher,
    target_modality="image",
    num_classes=10,
    img_size=28,
    audio_t=32,
    audio_f=32,
    d_z=64,
    epochs=10,
    lr=1e-3,
    T=4.0,
    alpha=0.5,
    device="cpu",
    verbose=True,
):
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = _build_student(
        target_modality, num_classes, img_size, audio_t, audio_f, d_z
    ).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    history = {"train_loss": [], "val_acc": []}

    for ep in range(epochs):
        student.train()
        total = 0.0
        for x_img, x_aud, y in train_loader:
            x_img = x_img.to(device)
            x_aud = x_aud.to(device)
            y = y.to(device)
            with torch.no_grad():
                t_logits = teacher(x_img, x_aud)
            x = x_img if target_modality == "image" else x_aud
            s_logits = student(x)
            loss = (1 - alpha) * F.cross_entropy(s_logits, y) + alpha * kd_loss(
                s_logits, t_logits, T=T
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg_loss = total / max(1, len(train_loader))
        acc = _evaluate(student, val_loader, target_modality, device)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)
        if verbose:
            print(
                f"[baseline/KD][{target_modality}] epoch {ep + 1}/{epochs} "
                f"loss={avg_loss:.4f} val_acc={acc:.4f}"
            )

    return {"student": student, "history": history}


# ---------------------------------------------------------------------------
# Helper used by Stage 1 to obtain sequence-form teacher features
# ---------------------------------------------------------------------------

def _teacher_features(teacher_mm, teacher_img, teacher_aud, x_img, x_aud):
    """Return per-teacher sequence features (B, T, d_z) for use in S1.

    We use the unimodal teachers' intermediate features `feat` directly,
    and for the multimodal teacher we expose its image and audio branch
    features. All three live in the same channel dimension `d_z` by design.
    """
    _, f_img_t = teacher_img(x_img, return_feat=True)
    _, f_aud_t = teacher_aud(x_aud, return_feat=True)
    _, f_img_mm, f_aud_mm = teacher_mm(x_img, x_aud, return_feat=True)
    return f_img_t, f_aud_t, f_img_mm, f_aud_mm


# ---------------------------------------------------------------------------
# Main entry point: full three-stage TSR-Distill training
# ---------------------------------------------------------------------------

def train_tsr_distill(
    train_loader,
    val_loader,
    target_modality="image",
    num_classes=10,
    img_size=28,
    audio_t=32,
    audio_f=32,
    d_z=64,
    d_mid=64,
    kernel_sizes=(3, 7, 15),
    # Stage budgets (small by default for fast demonstration runs).
    epochs_s1=15,
    epochs_s2=3,
    epochs_s3=10,
    lr=1e-3,
    # Loss coefficients (defaults from the paper, Section IV-A.1).
    alpha=0.5,        # weight on l_CTA
    beta=1.0,         # weight on l_ICC
    gamma=0.2,        # weight on l_DA
    eta=1.0,          # weight on l_HKD (feature-level hierarchical KD)
    zeta=1.0,         # weight on l_RKD (router-weighted logit-level KD)
    mu=2.0,           # weight on load-balancing (substantial to prevent
                      # winner-take-all routing collapse; the paper's
                      # mu=0.01 reflects much larger batches and a longer
                      # joint training horizon than the demo here)
    kappa=0.1,        # CTA temperature
    tau_c=0.07,       # ICC temperature
    T_kd=4.0,         # softmax temperature for the logit-level KD term
    lr_s3_mult=0.3,   # multiplier on `lr` for the Stage 3 student/CAR updates
    device="cpu",
    verbose=True,
    return_internals=False,
):
    """Run the full three-stage TSR-Distill pipeline on `train_loader`.

    Parameters
    ----------
    return_internals : bool
        If True, the returned dictionary also includes the trained HKPA
        modules, the CAR router, the teachers, and the gate weights — used
        by visualisation scripts to inspect internal behaviour.
    """
    # ---------- Stage 0: build modality-specific base teachers + MM teacher.
    teacher_img = ImageEncoder(num_classes, d_z, img_size).to(device)
    teacher_aud = AudioEncoder(num_classes, d_z, audio_t, audio_f).to(device)
    teacher_mm = MultimodalTeacher(
        num_classes, d_z, img_size, audio_t, audio_f
    ).to(device)

    # ----------- Stage 1: joint training of all base teachers with l_CTA + l_ICC.
    if verbose:
        print("=== Stage 1: Temporal-Intrinsic Collaborative Bootstrapping ===")
    params_s1 = (
        list(teacher_img.parameters())
        + list(teacher_aud.parameters())
        + list(teacher_mm.parameters())
    )
    opt_s1 = torch.optim.Adam(params_s1, lr=lr)
    s1_history = {"loss_task": [], "loss_cta": [], "loss_icc": []}

    for ep in range(epochs_s1):
        teacher_img.train(); teacher_aud.train(); teacher_mm.train()
        l_t, l_c, l_i = 0.0, 0.0, 0.0
        for x_img, x_aud, y in train_loader:
            x_img = x_img.to(device); x_aud = x_aud.to(device); y = y.to(device)

            logit_img, f_img = teacher_img(x_img, return_feat=True)
            logit_aud, f_aud = teacher_aud(x_aud, return_feat=True)
            logit_mm, f_img_mm, f_aud_mm = teacher_mm(x_img, x_aud, return_feat=True)

            # Cross-entropy on every base teacher and on the MM teacher
            # (Eq. 5: M+1 task terms with unit weight).
            l_task = (
                F.cross_entropy(logit_img, y)
                + F.cross_entropy(logit_aud, y)
                + F.cross_entropy(logit_mm, y)
            )
            # Apply CTA / ICC to both the unimodal pair AND the MM teacher's
            # internal pair, so all teachers benefit from temporal alignment.
            l_cta = (
                cta_loss(f_img, f_aud, kappa=kappa)
                + cta_loss(f_img_mm, f_aud_mm, kappa=kappa)
            )
            l_icc = (
                icc_loss(f_img.mean(dim=1), f_aud.mean(dim=1), tau=tau_c)
                + icc_loss(f_img_mm.mean(dim=1), f_aud_mm.mean(dim=1), tau=tau_c)
            )
            loss = l_task + alpha * l_cta + beta * l_icc

            opt_s1.zero_grad()
            loss.backward()
            opt_s1.step()
            l_t += l_task.item(); l_c += l_cta.item(); l_i += l_icc.item()

        n = max(1, len(train_loader))
        s1_history["loss_task"].append(l_t / n)
        s1_history["loss_cta"].append(l_c / n)
        s1_history["loss_icc"].append(l_i / n)
        if verbose:
            print(
                f"  S1 epoch {ep + 1}/{epochs_s1}  task={l_t/n:.4f}  "
                f"CTA={l_c/n:.4f}  ICC={l_i/n:.4f}"
            )

    # Freeze base teachers from here on.
    for m in (teacher_img, teacher_aud, teacher_mm):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    # ----------- Stage 2: build N specialised teachers via HKPA + AuxMLP.
    if verbose:
        print("=== Stage 2: Hierarchical Knowledge Projection Adaptation ===")
    # Sequence length of the student's intermediate layer.
    if target_modality == "image":
        # ImageEncoder exposes a (B, (img_size/8)^2, d_z) feature.
        T_t = max(1, img_size // 8) ** 2
    else:
        T_t = max(1, audio_t // 4)

    # First, train a reference student with task loss only. This student
    # provides a frozen feature template Z_t^{l'} for the HKPA regression
    # target (Eq. 8) — kept task-only so the template captures a stable
    # modality-specific structure rather than being biased by any
    # particular distillation signal.
    ref_student = _build_student(
        target_modality, num_classes, img_size, audio_t, audio_f, d_z
    ).to(device)
    opt_ref = torch.optim.Adam(ref_student.parameters(), lr=lr)
    for ep in range(max(2, epochs_s2)):
        ref_student.train()
        for x_img, x_aud, y in train_loader:
            x_stu = (x_img if target_modality == "image" else x_aud).to(device)
            logits = ref_student(x_stu)
            loss = F.cross_entropy(logits, y.to(device))
            opt_ref.zero_grad(); loss.backward(); opt_ref.step()
    for p in ref_student.parameters():
        p.requires_grad = False
    ref_student.eval()

    # In parallel, train a KD-warmed student that will serve as the
    # Stage 3 initialisation. This lifts the starting point of Stage 3
    # close to the response-based KD baseline, and lets HKD + RKD refine
    # the feature representation from there.
    init_student = _build_student(
        target_modality, num_classes, img_size, audio_t, audio_f, d_z
    ).to(device)
    opt_init = torch.optim.Adam(init_student.parameters(), lr=lr)
    for ep in range(max(2, epochs_s2)):
        init_student.train()
        for x_img, x_aud, y in train_loader:
            x_img = x_img.to(device); x_aud = x_aud.to(device); y = y.to(device)
            x_stu = x_img if target_modality == "image" else x_aud
            stu_logits = init_student(x_stu)
            with torch.no_grad():
                t_logits = teacher_mm(x_img, x_aud)
            loss = 0.5 * F.cross_entropy(stu_logits, y) + 0.5 * kd_loss(
                stu_logits, t_logits, T=T_kd
            )
            opt_init.zero_grad(); loss.backward(); opt_init.step()

    # Each HKPA bridges from a teacher feature to the student-compatible space.
    # We instantiate N = M_eff * L_attach specialised teachers; here we use
    # M_eff = 2 base teachers (image + audio + MM provides 3, but for clarity
    # we attach to the *opposite-modality* unimodal teacher and to the MM
    # branch matching the source modality) and L_attach = 1 layer per teacher,
    # giving N = 4 specialised teachers. This stays faithful to the paper's
    # N = M_eff * L_attach formula while keeping the demo light.
    if target_modality == "image":
        # Source teachers we distil from: opposite modality + MM branches.
        teacher_specs = [
            ("aud_unimodal", teacher_aud, audio_t // 4),
            ("img_mm",        teacher_mm,  None),     # use mm image branch
            ("aud_mm",        teacher_mm,  None),     # use mm audio branch
            ("aud_layer2",    teacher_aud, audio_t // 4),
        ]
    else:
        teacher_specs = [
            ("img_unimodal", teacher_img, max(1, img_size // 8) ** 2),
            ("img_mm",        teacher_mm,  None),
            ("aud_mm",        teacher_mm,  None),
            ("img_layer2",    teacher_img, max(1, img_size // 8) ** 2),
        ]
    num_teachers = len(teacher_specs)

    hkpas = nn.ModuleList(
        [
            HKPA(
                d_in=d_z,
                d_mid=d_mid,
                d_out=d_z,
                target_T=T_t,
                kernel_sizes=kernel_sizes,
            ).to(device)
            for _ in range(num_teachers)
        ]
    )
    # One auxiliary classifier per specialised teacher (for l_DA in Stage 2).
    aux_da = nn.ModuleList(
        [AuxClassifier(d_z, num_classes).to(device) for _ in range(num_teachers)]
    )

    def get_teacher_seq_feat(spec, x_img, x_aud):
        """Return a (B, T, d_z) sequence feature from the named teacher."""
        name, model, _ = spec
        if name == "aud_unimodal" or name == "aud_layer2":
            _, f = model(x_aud, return_feat=True)
        elif name == "img_unimodal" or name == "img_layer2":
            _, f = model(x_img, return_feat=True)
        elif name == "img_mm":
            _, f_img_mm, _ = model(x_img, x_aud, return_feat=True)
            f = f_img_mm
        elif name == "aud_mm":
            _, _, f_aud_mm = model(x_img, x_aud, return_feat=True)
            f = f_aud_mm
        else:
            raise ValueError(name)
        return f

    params_s2 = list(hkpas.parameters()) + list(aux_da.parameters())
    opt_s2 = torch.optim.Adam(params_s2, lr=lr)
    s2_history = {"loss_feat": [], "loss_da": []}

    for ep in range(epochs_s2):
        for m in hkpas: m.train()
        for m in aux_da: m.train()
        l_f, l_d = 0.0, 0.0
        for x_img, x_aud, y in train_loader:
            x_img = x_img.to(device); x_aud = x_aud.to(device); y = y.to(device)
            x_stu = x_img if target_modality == "image" else x_aud
            with torch.no_grad():
                _, f_stu = ref_student(x_stu, return_feat=True)

            total_feat, total_da = 0.0, 0.0
            for j, spec in enumerate(teacher_specs):
                with torch.no_grad():
                    f_tea = get_teacher_seq_feat(spec, x_img, x_aud)
                z_tilde = hkpas[j](f_tea)                  # (B, T_t, d_z)
                total_feat = total_feat + feature_regression_loss(z_tilde, f_stu)
                logits_aux = aux_da[j](z_tilde.mean(dim=1))
                total_da = total_da + domain_aware_loss(logits_aux, y)

            loss = total_feat + gamma * total_da
            opt_s2.zero_grad()
            loss.backward()
            opt_s2.step()
            l_f += total_feat.item()
            l_d += total_da.item()

        n = max(1, len(train_loader))
        s2_history["loss_feat"].append(l_f / n)
        s2_history["loss_da"].append(l_d / n)
        if verbose:
            print(
                f"  S2 epoch {ep + 1}/{epochs_s2}  feat={l_f/n:.4f}  "
                f"DA={l_d/n:.4f}"
            )

    # Freeze HKPAs.
    for m in hkpas:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    # ----------- Stage 3: train the student under CAR-routed hierarchical KD.
    if verbose:
        print("=== Stage 3: Context-Aware Hierarchical Routed Distillation ===")

    # Build a fresh, trainable student for Stage 3. We warm-start its
    # parameters from `init_student` (a short KD-pretrained snapshot) so
    # the HKD + RKD signal refines an already cross-modally informed
    # feature space, rather than competing with cold-start optimisation.
    student = _build_student(
        target_modality, num_classes, img_size, audio_t, audio_f, d_z
    ).to(device)
    student.load_state_dict(init_student.state_dict())

    # Per-level auxiliary classifiers used to compute sigma (Eq. 12).
    aux_sigma = nn.ModuleList(
        [AuxClassifier(d_mid, num_classes).to(device) for _ in kernel_sizes]
    )
    # Per-level linear projection Psi_h: d_z -> d_mid (Eq. 14).
    psi = nn.ModuleList(
        [nn.Linear(d_z, d_mid).to(device) for _ in kernel_sizes]
    )
    # Per-teacher logit heads on the HKPA-projected feature. These produce
    # the soft targets used by the router-weighted logit-level KD term.
    # Trained jointly with the student in Stage 3; the underlying HKPAs
    # remain frozen, so each head only learns a thin readout on top of the
    # already-aligned representation Z_tilde.
    teacher_logit_heads = nn.ModuleList(
        [nn.Linear(d_z, num_classes).to(device) for _ in range(num_teachers)]
    )

    phi_dim = get_phi_dim(target_modality, audio_f=audio_f, img_channels=1)
    car = ContextAwareRouter(
        d_stu=d_z,
        d_phi=phi_dim,
        num_levels=len(kernel_sizes),
        num_teachers=num_teachers,
    ).to(device)

    params_s3 = (
        list(student.parameters())
        + list(car.parameters())
        + list(psi.parameters())
        + list(aux_sigma.parameters())
        + list(teacher_logit_heads.parameters())
    )
    opt_s3 = torch.optim.Adam(params_s3, lr=lr * lr_s3_mult)
    s3_history = {
        "loss_task": [], "loss_hkd": [], "loss_rkd": [],
        "loss_lb": [], "val_acc": [],
    }

    # Storage of last-batch routing weights, used by visualisation scripts.
    last_routing_weights = None

    for ep in range(epochs_s3):
        student.train(); car.train()
        for m in psi: m.train()
        for m in aux_sigma: m.train()
        for m in teacher_logit_heads: m.train()

        l_task_tot, l_hkd_tot, l_rkd_tot, l_lb_tot = 0.0, 0.0, 0.0, 0.0
        for x_img, x_aud, y in train_loader:
            x_img = x_img.to(device); x_aud = x_aud.to(device); y = y.to(device)
            x_stu = x_img if target_modality == "image" else x_aud

            # Forward through student.
            stu_logits, f_stu = student(x_stu, return_feat=True)   # (B, T_t, d_z)
            stu_pooled = f_stu.mean(dim=1)                         # (B, d_z)

            # Forward through every HKPA (no grad on HKPAs) and collect both
            # the hierarchical features G^{h,j} and the final projected
            # feature Z_tilde^j for the logit-level term.
            G_all = []     # list[N] of list[L_h] of (B, T_t, d_mid)
            Z_tilde_all = []   # list[N] of (B, T_t, d_z)
            with torch.no_grad():
                for j, spec in enumerate(teacher_specs):
                    f_tea = get_teacher_seq_feat(spec, x_img, x_aud)
                    z_t = hkpas[j](f_tea)                          # (B, T_t, d_z)
                    G_all.append([g.detach() for g in hkpas[j].last_G])
                    Z_tilde_all.append(z_t.detach())

            # Per-level disagreement entropy sigma (Eq. 11-12).
            sigma_list = []
            for h in range(len(kernel_sizes)):
                p_h = []
                for j in range(num_teachers):
                    p_jh = F.softmax(
                        aux_sigma[h](G_all[j][h].mean(dim=1)), dim=1
                    )
                    p_h.append(p_jh)
                p_mean = torch.stack(p_h, dim=0).mean(dim=0)        # (B, C)
                sigma_h = -(p_mean * torch.log(p_mean + 1e-8)).sum(dim=1)
                sigma_list.append(sigma_h)                          # (B,)
            sigma = torch.stack(sigma_list, dim=1)                  # (B, L_h)

            # phi: non-learnable input statistics (Eq. 10).
            with torch.no_grad():
                if target_modality == "image":
                    phi = compute_image_phi(x_img)
                else:
                    phi = compute_audio_phi(x_aud)

            # Router output.
            w, _ = car(stu_pooled.detach(), phi, sigma.detach())   # (B, N)

            # Student-side per-level projections.
            stu_projs = [psi[h](f_stu) for h in range(len(kernel_sizes))]

            # Feature-level hierarchical KD loss (Eq. 14).
            l_hkd = hierarchical_kd_loss(G_all, stu_projs, w)

            # Router-weighted logit-level KD loss: for each specialised
            # teacher j, compute soft targets from its Z_tilde feature and
            # KL-divergence against the student logits, then sum weighted
            # by w^j. This is exactly the routed analogue of Hinton KD.
            log_p_s = F.log_softmax(stu_logits / T_kd, dim=1)
            l_rkd = 0.0
            for j in range(num_teachers):
                tlogits = teacher_logit_heads[j](Z_tilde_all[j].mean(dim=1))
                p_t = F.softmax(tlogits.detach() / T_kd, dim=1)
                kl_per_sample = (p_t * (torch.log(p_t + 1e-8) - log_p_s)).sum(dim=1)
                l_rkd = l_rkd + (w[:, j] * kl_per_sample).mean() * (T_kd * T_kd)
            # Train the teacher heads with their own CE so soft targets
            # are meaningful; this gradient does not flow through HKPAs.
            l_head_ce = 0.0
            for j in range(num_teachers):
                tlogits = teacher_logit_heads[j](Z_tilde_all[j].mean(dim=1))
                l_head_ce = l_head_ce + F.cross_entropy(tlogits, y)

            l_task = F.cross_entropy(stu_logits, y)
            l_lb = load_balancing_loss(w)
            loss = (
                l_task
                + eta * l_hkd
                + zeta * l_rkd
                + l_head_ce
                + mu * l_lb
            )

            opt_s3.zero_grad()
            loss.backward()
            opt_s3.step()
            l_task_tot += l_task.item()
            l_hkd_tot += l_hkd.item()
            l_rkd_tot += float(l_rkd) if isinstance(l_rkd, float) else l_rkd.item()
            l_lb_tot += l_lb.item()
            last_routing_weights = w.detach().cpu()

        n = max(1, len(train_loader))
        val_acc = _evaluate(student, val_loader, target_modality, device)
        s3_history["loss_task"].append(l_task_tot / n)
        s3_history["loss_hkd"].append(l_hkd_tot / n)
        s3_history["loss_rkd"].append(l_rkd_tot / n)
        s3_history["loss_lb"].append(l_lb_tot / n)
        s3_history["val_acc"].append(val_acc)
        if verbose:
            print(
                f"  S3 epoch {ep + 1}/{epochs_s3}  task={l_task_tot/n:.4f}  "
                f"HKD={l_hkd_tot/n:.4f}  RKD={l_rkd_tot/n:.4f}  "
                f"LB={l_lb_tot/n:.4f}  val_acc={val_acc:.4f}"
            )

    out = {
        "student": student,
        "s1_history": s1_history,
        "s2_history": s2_history,
        "s3_history": s3_history,
        "final_val_acc": s3_history["val_acc"][-1] if s3_history["val_acc"] else None,
        "teacher_mm_val_acc": _evaluate_mm(teacher_mm, val_loader, device),
    }
    if return_internals:
        out.update(
            {
                "teacher_img": teacher_img,
                "teacher_aud": teacher_aud,
                "teacher_mm": teacher_mm,
                "hkpas": hkpas,
                "car": car,
                "psi": psi,
                "last_routing_weights": last_routing_weights,
                "teacher_specs": [s[0] for s in teacher_specs],
                "kernel_sizes": kernel_sizes,
            }
        )
    return out
