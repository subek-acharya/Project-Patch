"""
apgd_l2_patchguard.py

APGD L2 attack adapted for PatchGuard anomaly detection.
Uses PatchGuard's score-based loss instead of DLR/CE classification loss.
"""

import time
import math
import torch
import torch.nn as nn


def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


class APGD_L2_PatchGuard:
    """
    APGD L2 attack for PatchGuard anomaly detection.
    
    Key differences from classification APGD:
    - Uses score-based loss (normal_avg - anomalous_avg) instead of DLR/CE
    - No "misclassification" check (anomaly detection uses AUROC)
    - Tracks best loss instead of flipped predictions
    - Masks (ground truth) used instead of class labels
    """

    def __init__(
        self,
        model,
        eps,
        n_iter=200,
        n_restarts=1,
        seed=0,
        rho=0.75,
        verbose=False,
        device=None,
    ):
        self.model = model
        self.eps = eps
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.seed = seed
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device

        # Checkpoint parameters (adaptive step size schedule)
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def normalize(self, x):
        """Normalize to unit L2 norm."""
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return x / (t.view(-1, *([1] * (len(x.shape) - 1))) + 1e-12)

    def patchguard_loss(self, scores, masks):
        """
        PatchGuard's attack loss function.
        
        Args:
            scores: model output [batch, num_patches] (256 patch scores)
            masks: ground truth [batch, num_patches] (0=normal, 1=anomalous)
        
        Returns:
            loss_indiv: per-sample loss [batch]
            loss: total loss (scalar)
        """
        zeros_count = (masks == 0).sum(dim=1)
        non_zeros_count = (masks != 0).sum(dim=1)

        anomalous_loss = (masks * scores).sum(dim=1) / (non_zeros_count + 1e-8)
        normal_loss = ((1 - masks) * scores).sum(dim=1) / (zeros_count + 1e-8)

        loss_indiv = normal_loss - anomalous_loss
        loss = loss_indiv.sum()

        return loss_indiv, loss

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        """Check if loss is oscillating (not improving steadily)."""
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()
        return (t <= k * k3 * torch.ones_like(t)).float()

    def attack_single_run(self, x, masks, x_init=None):
        """
        Single run of APGD L2 attack.
        
        Args:
            x: clean images [batch, C, H, W]
            masks: ground truth patch masks [batch, num_patches]
            x_init: optional initial perturbation
            
        Returns:
            x_best: best adversarial found (highest loss)
            loss_best: best loss values per sample
            x_best_adv: adversarial examples at best loss
        """
        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        ndims = len(x.shape) - 1

        # ─── Initialize adversarial image ───
        if x_init is not None:
            x_adv = x_init.clone()
        else:
            # L2: start at clean image (no random init)
            x_adv = x.clone()

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()

        # Storage for tracking
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]).to(self.device)

        # ─── Initial gradient computation ───
        x_adv.requires_grad_()
        with torch.enable_grad():
            scores = self.model(x_adv)
            loss_indiv, loss = self.patchguard_loss(scores, masks)

        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        grad_best = grad.clone()
        loss_best = loss_indiv.detach().clone()
        loss_best_steps[0] = loss_best

        # ─── Step size initialization ───
        alpha = 2.0
        step_size = alpha * self.eps * torch.ones(
            [x.shape[0], *([1] * ndims)]
        ).to(self.device).detach()

        x_adv_old = x_adv.clone().detach()
        counter3 = 0
        k = self.n_iter_2 + 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        # ─── Main APGD loop ───
        for i in range(self.n_iter):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old  # momentum (previous direction)
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                # ─── L2 step: normalized gradient ───
                x_adv_1 = x_adv + step_size * self.normalize(grad)

                # ─── L2 projection: ensure ||delta||_2 <= eps ───
                x_adv_1 = torch.clamp(
                    x + self.normalize(x_adv_1 - x) * torch.min(
                        self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)
                    ),
                    0.0, 1.0
                )

                # ─── Apply momentum ───
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)

                # ─── Project again after momentum ───
                x_adv_1 = torch.clamp(
                    x + self.normalize(x_adv_1 - x) * torch.min(
                        self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)
                    ),
                    0.0, 1.0
                )

                x_adv = x_adv_1 + 0.

            # ─── Compute gradient ───
            x_adv.requires_grad_()
            with torch.enable_grad():
                scores = self.model(x_adv)
                loss_indiv, loss = self.patchguard_loss(scores, masks)

            grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            if self.verbose and i % 50 == 0:
                print(f'[APGD-L2] iter: {i} - best loss: {loss_best.sum():.6f} '
                      f'- step size: {step_size.mean():.5f}')

            # ─── Track best results ───
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1

                # Update best
                ind = (y1 > loss_best).nonzero().squeeze()
                if ind.numel() > 0:
                    x_best[ind] = x_adv[ind].clone()
                    grad_best[ind] = grad[ind].clone()
                    loss_best[ind] = y1[ind]

                loss_best_steps[i + 1] = loss_best

                # ─── Adaptive step size check ───
                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps, i, k, loss_best, k3=self.thr_decr
                    )
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                        loss_best_last_check >= loss_best
                    ).float()
                    fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0
                        n_reduced = fl_oscillation.sum()

                        # Reset to best found so far
                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - self.size_decr, self.n_iter_min)
                    counter3 = 0

        return x_best, loss_best, x_best_adv

    def attack(self, x, masks):
        """
        Run APGD L2 attack with restarts.
        
        Args:
            x: clean images [batch, C, H, W]
            masks: ground truth patch masks [batch, num_patches]
            
        Returns:
            x_adv: best adversarial examples found
        """
        if self.device is None:
            self.device = x.device

        x = x.detach().clone().float().to(self.device)
        masks = masks.detach().clone().float().to(self.device)

        x_adv_best = x.clone()
        loss_best = torch.ones(x.shape[0]).to(self.device) * (-float('inf'))

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        for restart in range(self.n_restarts):
            # Random init for restarts > 0
            if restart > 0:
                t = torch.randn(x.shape).to(self.device).detach()
                x_init = x + self.eps * self.normalize(t)
                x_init = x_init.clamp(0., 1.)
            else:
                x_init = None

            x_best_curr, loss_best_curr, _ = self.attack_single_run(
                x, masks, x_init=x_init
            )

            # Update overall best
            ind = (loss_best_curr > loss_best).nonzero().squeeze()
            if ind.numel() > 0:
                x_adv_best[ind] = x_best_curr[ind].clone()
                loss_best[ind] = loss_best_curr[ind]

            if self.verbose:
                print(f'[APGD-L2] restart {restart} - best loss: {loss_best.sum():.5f}')

        return x_adv_best


def apgd_l2_attack_patchguard(model, images, masks, epsilon, n_iter=200, n_restarts=1, device=None):
    """
    Convenience function to run APGD L2 attack on PatchGuard.
    
    Args:
        model: PatchGuard model
        images: clean images [batch, C, H, W]
        masks: ground truth patch masks [batch, num_patches]
        epsilon: L2 perturbation budget
        n_iter: number of iterations (default: 200)
        n_restarts: number of random restarts (default: 1)
        device: torch device
        
    Returns:
        x_adv: adversarial images [batch, C, H, W]
    """
    if device is None:
        device = images.device

    attacker = APGD_L2_PatchGuard(
        model=model,
        eps=epsilon,
        n_iter=n_iter,
        n_restarts=n_restarts,
        seed=0,
        rho=0.75,
        verbose=True,
        device=device,
    )

    x_adv = attacker.attack(images, masks)

    return x_adv