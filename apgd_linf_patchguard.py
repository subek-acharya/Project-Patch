"""
apgd_linf_patchguard.py

APGD L∞ attack adapted for PatchGuard anomaly detection.
Uses PatchGuard's score-based loss instead of CE/DLR classification loss.
"""

import math
import torch
import torch.nn as nn


class APGD_Linf_PatchGuard:
    """
    APGD L∞ attack for PatchGuard anomaly detection.
    
    Key features:
    - Sign of gradient for step direction (uniform step per pixel)
    - Per-pixel L∞ constraint (clamp to ±ε)
    - Momentum (75% current + 25% previous direction)
    - Adaptive step size with checkpoint schedule
    - PatchGuard's score-based loss (not CE/DLR)
    """

    def __init__(
        self,
        model,
        eps,
        n_iter=200,
        n_restarts=1,
        seed=0,
        alpha=0.75,
        rho=0.75,
        verbose=False,
        device=None,
    ):
        self.model = model
        self.eps = eps
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.seed = seed
        self.alpha = alpha
        self.rho = rho
        self.verbose = verbose
        self.device = device

    def build_checkpoint_schedule(self):
        """Build adaptive checkpoint schedule W."""
        W = [0]
        p_prev2, p_prev1 = 0.0, 0.22
        W.append(int(p_prev1 * self.n_iter))
        while W[-1] < self.n_iter:
            delta = max(p_prev1 - p_prev2 - 0.03, 0.06)
            p_next = p_prev1 + delta
            W.append(int(p_next * self.n_iter))
            p_prev2, p_prev1 = p_prev1, p_next
        return W

    def projection_linf(self, x_adv, x_orig):
        """Project onto L∞ ball around x_orig."""
        return torch.max(
            torch.min(x_adv, x_orig + self.eps),
            x_orig - self.eps
        )

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

    def get_grad(self, x, masks):
        """Compute gradient of PatchGuard loss w.r.t. input."""
        x_var = x.clone().detach().requires_grad_(True)
        scores = self.model(x_var)
        _, loss = self.patchguard_loss(scores, masks)
        grad = torch.autograd.grad(loss, x_var)[0].detach()
        return grad

    def get_loss(self, x, masks):
        """Compute per-sample loss (no gradient)."""
        with torch.no_grad():
            scores = self.model(x)
            loss_indiv, _ = self.patchguard_loss(scores, masks)
        return loss_indiv

    def attack_single_run(self, x, masks, random_start=False):
        """
        Single run of APGD L∞ attack.
        
        Args:
            x: clean images [batch, C, H, W]
            masks: ground truth patch masks [batch, num_patches]
            random_start: whether to start from random perturbation
            
        Returns:
            x_max: best adversarial found (highest loss)
            loss_max: best loss values per sample
        """
        bs = x.shape[0]
        x_clean = x.clone().detach()

        # Build checkpoint schedule
        W = self.build_checkpoint_schedule()

        # ─── Initialize x_k ───
        if random_start:
            delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            x_k = torch.clamp(x_clean + delta, 0.0, 1.0).detach()
        else:
            x_k = x_clean.clone().detach()

        # ─── First step → x(1) ───
        eta_init = 2.0 * self.eps
        grad = self.get_grad(x_k, masks)
        z_next = torch.clamp(x_k + eta_init * grad.sign(), 0.0, 1.0)
        x_next = self.projection_linf(z_next, x_clean).detach()

        # ─── Initialize tracking (x_max, f_max) ───
        f_x0 = self.get_loss(x_k, masks)
        f_x1 = self.get_loss(x_next, masks)

        better = (f_x1 > f_x0)
        x_max = x_clean.clone()
        x_max[better] = x_next[better]
        f_max = f_x0.clone()
        f_max[better] = f_x1[better]

        # Initialize loop variables
        x_prev = x_k.clone()
        x_k = x_next.clone()

        # Per-sample step size
        eta = torch.full((bs, 1, 1, 1), eta_init, device=self.device, dtype=x.dtype)

        # Per-sample improvement counters
        improvement = torch.zeros(bs, device=self.device, dtype=torch.int32)

        checkpoint_ptr = 1
        prev_eta = eta.clone()
        prev_f_max = f_max.clone()

        # ─── Main APGD L∞ loop ───
        for k in range(1, self.n_iter):
            # ─── Compute gradient ───
            grad = self.get_grad(x_k, masks)

            # ─── Take step with sign(grad) ───
            z_next = torch.clamp(x_k + eta * grad.sign(), 0.0, 1.0)
            z_next = self.projection_linf(z_next, x_clean)

            # ─── Apply momentum ───
            x_next = x_k + self.alpha * (z_next - x_k) + (1 - self.alpha) * (x_k - x_prev)
            x_next = self.projection_linf(x_next, x_clean)
            x_next = torch.clamp(x_next, 0.0, 1.0).detach()

            # ─── Compute losses ───
            f_k = self.get_loss(x_k, masks)
            f_next = self.get_loss(x_next, masks)

            # ─── Track improvements ───
            improvement += (f_next > f_k).to(torch.int32)

            # Update x_max, f_max
            better2 = (f_next > f_max)
            x_max[better2] = x_next[better2]
            f_max[better2] = f_next[better2]

            # ─── Checkpoint: adaptive step size ───
            if checkpoint_ptr < len(W) and k == W[checkpoint_ptr]:
                interval = W[checkpoint_ptr] - W[checkpoint_ptr - 1]

                # Condition 1: not improving fast enough
                cond1 = improvement.to(torch.float32) < (self.rho * interval)

                # Condition 2: eta unchanged AND f_max unchanged (stuck)
                same_eta = (eta == prev_eta).all(dim=(1, 2, 3))
                cond2 = same_eta & (f_max == prev_f_max)

                # Apply: halve eta and reset to best
                reduce = (cond1 | cond2)
                if reduce.any():
                    eta[reduce] = eta[reduce] / 2.0
                    x_next[reduce] = x_max[reduce].clone()
                    x_k[reduce] = x_max[reduce].clone()

                # Reset counters for next interval
                improvement.zero_()
                prev_eta = eta.clone()
                prev_f_max = f_max.clone()
                checkpoint_ptr += 1

            if self.verbose and k % 50 == 0:
                print(f'[APGD-Linf] iter: {k} - best loss: {f_max.sum():.6f} '
                      f'- eta: {eta.mean():.6f}')

            # Shift for next iteration
            x_prev = x_k.clone()
            x_k = x_next.clone()

        return x_max, f_max

    def attack(self, x, masks):
        """
        Run APGD L∞ attack with restarts.
        
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
            use_random = (restart > 0)

            x_max_curr, f_max_curr = self.attack_single_run(
                x, masks, random_start=use_random
            )

            # Update overall best
            ind = (f_max_curr > loss_best).nonzero().squeeze()
            if ind.numel() > 0:
                x_adv_best[ind] = x_max_curr[ind].clone()
                loss_best[ind] = f_max_curr[ind]

            if self.verbose:
                print(f'[APGD-Linf] restart {restart} - best loss: {loss_best.sum():.5f}')

        return x_adv_best


def apgd_linf_attack_patchguard(model, images, masks, epsilon, n_iter=200, n_restarts=1, device=None):
    """
    Convenience function to run APGD L∞ attack on PatchGuard.
    
    Args:
        model: PatchGuard model
        images: clean images [batch, C, H, W]
        masks: ground truth patch masks [batch, num_patches]
        epsilon: L∞ perturbation budget (e.g., 8/255)
        n_iter: number of iterations (default: 200)
        n_restarts: number of random restarts (default: 1)
        device: torch device
        
    Returns:
        x_adv: adversarial images [batch, C, H, W]
    """
    if device is None:
        device = images.device

    attacker = APGD_Linf_PatchGuard(
        model=model,
        eps=epsilon,
        n_iter=n_iter,
        n_restarts=n_restarts,
        seed=0,
        alpha=0.75,
        rho=0.75,
        verbose=True,
        device=device,
    )

    x_adv = attacker.attack(images, masks)

    return x_adv