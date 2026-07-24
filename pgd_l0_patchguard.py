"""
pgd_l0_patchguard.py

PGD L0 attack adapted for PatchGuard anomaly detection.
Uses PatchGuard's score-based loss instead of DLR/CE classification loss.
L0 norm: at most k pixels change, each by up to 1.0 (sparse, large changes).
"""

import math
import torch
import torch.nn as nn
import numpy as np


def project_L0_box_torch(delta, k, x_nat):
    """
    Project perturbation onto L0 ball: keep only top-k most important pixels.
    All operations in PyTorch (GPU).
    
    Args:
        delta: current perturbation [batch, C, H, W]
        k: maximum number of pixels to change (sparsity)
        x_nat: original image [batch, C, H, W]
        
    Returns:
        projected delta [batch, C, H, W]
    """
    batch_size = delta.shape[0]
    
    # Score each pixel location by magnitude of change (sum across channels)
    # p1 = sum of squared perturbation per spatial location
    p1 = (delta ** 2).sum(dim=1)  # [batch, H, W]
    
    # Penalty for being outside [0, 1] box
    x_adv = x_nat + delta
    lb_violation = torch.clamp(0.0 - x_adv, min=0)  # how much below 0
    ub_violation = torch.clamp(x_adv - 1.0, min=0)  # how much above 1
    p2 = (lb_violation ** 2 + ub_violation ** 2).sum(dim=1)  # [batch, H, W]
    
    # Score = importance - penalty
    scores = (p1 - p2).view(batch_size, -1)  # [batch, H*W]
    
    # Find threshold: k-th largest score per sample
    num_pixels = scores.shape[1]
    k_clamped = min(k, num_pixels)
    topk_vals, _ = torch.topk(scores, k_clamped, dim=1)
    threshold = topk_vals[:, -1].view(batch_size, 1, 1)  # [batch, 1, 1]
    
    # Create mask: keep only top-k pixel locations
    scores_2d = (p1 - p2)  # [batch, H, W]
    mask = (scores_2d >= threshold).unsqueeze(1).float()  # [batch, 1, H, W] → broadcast over channels
    
    # Apply mask and clip to box constraints
    delta_projected = delta * mask
    x_projected = x_nat + delta_projected
    x_projected = torch.clamp(x_projected, 0.0, 1.0)
    delta_projected = x_projected - x_nat
    
    return delta_projected


class PGD_L0_PatchGuard:
    """
    PGD L0 attack for PatchGuard anomaly detection.
    
    Key features:
    - At most k pixels change (sparsity constraint)
    - Each changed pixel can change by up to 1.0 (box constraint)
    - PatchGuard's score-based loss
    - Multiple restarts for robustness
    """

    def __init__(
        self,
        model,
        sparsity,
        num_steps=30,
        step_size=20.0,
        n_restarts=10,
        random_start=False,
        seed=0,
        verbose=False,
        device=None,
    ):
        self.model = model
        self.sparsity = sparsity  # k: max number of pixels to change
        self.num_steps = num_steps
        self.step_size = step_size
        self.n_restarts = n_restarts
        self.random_start = random_start
        self.seed = seed
        self.verbose = verbose
        self.device = device

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

    def attack_single_run(self, x, masks, random_start=False):
        """
        Single run of PGD L0 attack.
        
        Args:
            x: clean images [batch, C, H, W]
            masks: ground truth patch masks [batch, num_patches]
            random_start: whether to start from random perturbation
            
        Returns:
            x_best: best adversarial found (highest loss)
            loss_best: best loss values per sample
        """
        x_nat = x.clone().detach()
        
        # ─── Initialize ───
        if random_start:
            # Random start within [0, 1] box
            delta = torch.zeros_like(x).uniform_(-1.0, 1.0)
            delta = project_L0_box_torch(delta, self.sparsity, x_nat)
            x_adv = x_nat + delta
        else:
            x_adv = x_nat.clone()

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()

        # Initial loss
        with torch.no_grad():
            scores = self.model(x_adv)
            loss_indiv, _ = self.patchguard_loss(scores, masks)
        loss_best = loss_indiv.clone()

        # ─── Main PGD L0 loop ───
        for i in range(self.num_steps):
            if i > 0:
                # ─── Compute gradient ───
                x_adv.requires_grad_()
                with torch.enable_grad():
                    scores = self.model(x_adv)
                    loss_indiv, loss = self.patchguard_loss(scores, masks)

                grad = torch.autograd.grad(loss, [x_adv])[0].detach()

                # Normalize gradient by L1 norm (per sample)
                grad_norm = grad.abs().view(x.shape[0], -1).sum(dim=1, keepdim=True)
                grad_norm = grad_norm.view(x.shape[0], 1, 1, 1)
                grad = grad / (grad_norm + 1e-10)

                # Take step with tiny noise for tie-breaking
                noise = (torch.rand_like(grad) - 0.5) * 1e-12
                x_adv = x_adv.detach() + self.step_size * grad + noise

                # Track best
                with torch.no_grad():
                    ind = (loss_indiv > loss_best).nonzero().squeeze()
                    if ind.numel() > 0:
                        x_best[ind] = x_adv[ind].clone()
                        loss_best[ind] = loss_indiv[ind]

            # ─── Project onto L0 ball ───
            delta = x_adv.detach() - x_nat
            delta_projected = project_L0_box_torch(delta, self.sparsity, x_nat)
            x_adv = (x_nat + delta_projected).detach()

        # Final evaluation
        with torch.no_grad():
            scores = self.model(x_adv)
            loss_indiv, _ = self.patchguard_loss(scores, masks)
            ind = (loss_indiv > loss_best).nonzero().squeeze()
            if ind.numel() > 0:
                x_best[ind] = x_adv[ind].clone()
                loss_best[ind] = loss_indiv[ind]

        return x_best, loss_best

    def attack(self, x, masks):
        """
        Run PGD L0 attack with multiple restarts.
        
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
            # First restart: no random start; subsequent: random start
            use_random = (restart > 0) or self.random_start

            x_best_curr, loss_best_curr = self.attack_single_run(
                x, masks, random_start=use_random
            )

            # Update overall best
            ind = (loss_best_curr > loss_best).nonzero().squeeze()
            if ind.numel() > 0:
                x_adv_best[ind] = x_best_curr[ind].clone()
                loss_best[ind] = loss_best_curr[ind]

            if self.verbose:
                # Count pixels changed
                pixels_changed = (x_adv_best - x).abs().sum(dim=1).gt(1e-10).sum(dim=(1, 2)).float().mean()
                print(f'[PGD-L0] restart {restart} - best loss: {loss_best.sum():.5f} '
                      f'- avg pixels changed: {pixels_changed:.1f}/{self.sparsity}')

        return x_adv_best


def pgd_l0_attack_patchguard(model, images, masks, sparsity, num_steps=30, step_size=20.0, n_restarts=10, random_start=False, device=None):
    """
    Convenience function to run PGD L0 attack on PatchGuard.
    
    Args:
        model: PatchGuard model
        images: clean images [batch, C, H, W]
        masks: ground truth patch masks [batch, num_patches]
        sparsity: maximum number of pixels to change (k)
        num_steps: number of PGD iterations (default: 30)
        step_size: gradient step size (default: 20.0)
        n_restarts: number of random restarts (default: 10)
        random_start: use random initialization (default: False)
        device: torch device
        
    Returns:
        x_adv: adversarial images [batch, C, H, W]
    """
    if device is None:
        device = images.device

    attacker = PGD_L0_PatchGuard(
        model=model,
        sparsity=sparsity,
        num_steps=num_steps,
        step_size=step_size,
        n_restarts=n_restarts,
        random_start=random_start,
        seed=0,
        verbose=True,
        device=device,
    )

    x_adv = attacker.attack(images, masks)

    return x_adv