"""
apgd_l1_patchguard.py

APGD L1 attack adapted for PatchGuard anomaly detection.
Uses PatchGuard's score-based loss instead of DLR/CE classification loss.
L1 norm: sparse perturbation (few pixels change a lot).
"""

import time
import math
import torch
import torch.nn as nn


def L0_norm(x):
    """Count non-zero elements per sample."""
    return (x != 0.).view(x.shape[0], -1).sum(-1)


def L1_norm(x, keepdim=False):
    """L1 norm per sample."""
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1] * (len(x.shape) - 1))
    return z


def L1_projection(x2, y2, eps1):
    """
    Project onto L1 ball with box constraints [0, 1].
    
    x2: center of the L1 ball (original image) [bs x input_dim]
    y2: current perturbation (delta) [bs x input_dim]
    eps1: radius of the L1 ball
    
    output: correction delta_p such that ||y2 + delta_p||_1 <= eps1
            and 0 <= x2 + y2 + delta_p <= 1
    """
    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement != 0:
        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)


class APGD_L1_PatchGuard:
    """
    APGD L1 attack for PatchGuard anomaly detection.
    
    Key features:
    - Sparse perturbation (few pixels change significantly)
    - L1 projection via binary search
    - Sparsity-based adaptive step size
    - PatchGuard's score-based loss (not DLR/CE)
    """

    def __init__(
        self,
        model,
        eps,
        n_iter=500,
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

        # Checkpoint parameters
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):
        """Initialize hyperparameters based on input shape."""
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

    def normalize(self, x):
        """L1 normalize."""
        try:
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        except:
            t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

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

    def attack_single_run(self, x, masks, x_init=None):
        """
        Single run of APGD L1 attack.
        
        Args:
            x: clean images [batch, C, H, W]
            masks: ground truth patch masks [batch, num_patches]
            x_init: optional initial perturbation
            
        Returns:
            x_best: best adversarial found (highest loss)
            loss_best: best loss values per sample
        """
        if len(x.shape) < self.ndims + 1:
            x = x.unsqueeze(0)

        n_fts = math.prod(self.orig_dim)
        u = torch.arange(x.shape[0], device=self.device)

        # ─── Initialize adversarial image ───
        if x_init is not None:
            x_adv = x_init.clone()
        else:
            # L1: random initialization + projection onto L1 ball
            t = torch.randn(x.shape, device=self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()

        # Storage for tracking
        loss_steps = torch.zeros([self.n_iter, x.shape[0]], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]], device=self.device)

        # ─── Initial gradient computation ───
        x_adv.requires_grad_()
        with torch.enable_grad():
            scores = self.model(x_adv)
            loss_indiv, loss = self.patchguard_loss(scores, masks)

        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        grad_best = grad.clone()
        loss_best = loss_indiv.detach().clone()
        loss_best_steps[0] = loss_best

        # ─── Step size and sparsity initialization ───
        alpha = 1.0  # L1 uses alpha=1
        step_size = alpha * self.eps * torch.ones(
            [x.shape[0], *([1] * self.ndims)], device=self.device
        ).detach()

        # Sparsity parameters
        topk = .2 * torch.ones([x.shape[0]], device=self.device)
        sp_old = n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.

        x_adv_old = x_adv.clone().detach()
        counter3 = 0
        k = max(int(.04 * self.n_iter), 1)

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        # ─── Main APGD L1 loop ───
        for i in range(self.n_iter):
            # ─── Sparse gradient step ───
            with torch.no_grad():
                x_adv = x_adv.detach()
                x_adv_old = x_adv.clone()

                # Compute sparse gradient (keep only top-k% components)
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp(
                    (1. - topk) * n_fts, min=0, max=n_fts - 1
                ).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()

                # Take step in sparse sign direction, normalized by L1
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                    L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10
                )

                # ─── L1 projection ───
                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, self.eps)
                x_adv = x_adv_1 + delta_p

            # ─── Compute new gradient ───
            x_adv.requires_grad_()
            with torch.enable_grad():
                scores = self.model(x_adv)
                loss_indiv, loss = self.patchguard_loss(scores, masks)

            grad = torch.autograd.grad(loss, [x_adv])[0].detach()

            if self.verbose and i % 50 == 0:
                print(f'[APGD-L1] iter: {i} - best loss: {loss_best.sum():.6f} '
                      f'- step size: {step_size.mean():.5f} '
                      f'- topk: {topk.mean() * n_fts:.2f}')

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

                # ─── Adaptive step size (sparsity-based) ───
                counter3 += 1

                if counter3 == k:
                    sp_curr = L0_norm(x_best - x)  # current sparsity
                    fl_redtopk = (sp_curr / sp_old) < .95  # sparsity changed enough?
                    topk = sp_curr / n_fts / 1.5  # update sparsity parameter

                    # Reset step size for improved sparsity
                    step_size[fl_redtopk] = alpha * self.eps
                    # Reduce step size for unchanged sparsity
                    step_size[~fl_redtopk] /= adasp_redstep
                    # Clamp step size
                    step_size.clamp_(
                        alpha * self.eps / adasp_minstep,
                        alpha * self.eps
                    )
                    sp_old = sp_curr.clone()

                    # Reset to best for improved samples
                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

        return x_best, loss_best

    def attack(self, x, masks):
        """
        Run APGD L1 attack with restarts.
        
        Args:
            x: clean images [batch, C, H, W]
            masks: ground truth patch masks [batch, num_patches]
            
        Returns:
            x_adv: best adversarial examples found
        """
        if self.device is None:
            self.device = x.device

        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        masks = masks.detach().clone().float().to(self.device)

        x_adv_best = x.clone()
        loss_best = torch.ones(x.shape[0]).to(self.device) * (-float('inf'))

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        for restart in range(self.n_restarts):
            # Random init for restarts > 0
            if restart > 0:
                t = torch.randn(x.shape, device=self.device).detach()
                delta = L1_projection(x, t, self.eps)
                x_init = (x + t + delta).clamp(0., 1.)
            else:
                x_init = None

            x_best_curr, loss_best_curr = self.attack_single_run(
                x, masks, x_init=x_init
            )

            # Update overall best
            ind = (loss_best_curr > loss_best).nonzero().squeeze()
            if ind.numel() > 0:
                x_adv_best[ind] = x_best_curr[ind].clone()
                loss_best[ind] = loss_best_curr[ind]

            if self.verbose:
                print(f'[APGD-L1] restart {restart} - best loss: {loss_best.sum():.5f}')

        return x_adv_best

def apgd_l1_attack_patchguard(model, images, masks, epsilon, n_iter=500, n_restarts=1, device=None):
    """
    Convenience function to run APGD L1 attack on PatchGuard.
    
    Args:
        model: PatchGuard model
        images: clean images [batch, C, H, W]
        masks: ground truth patch masks [batch, num_patches]
        epsilon: L1 perturbation budget
        n_iter: number of iterations (default: 500)
        n_restarts: number of random restarts (default: 1)
        device: torch device
        
    Returns:
        x_adv: adversarial images [batch, C, H, W]
    """
    if device is None:
        device = images.device

    attacker = APGD_L1_PatchGuard(
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