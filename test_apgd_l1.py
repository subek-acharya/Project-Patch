"""
test_apgd_l1.py

Test PatchGuard with APGD L1 attack.

Usage:
    python test_apgd_l1.py --class_name bottle --dataset mvtec --dataset_dir ./datasets/mvtec --epsilon_l1 2000 --checkpoint_dir ./checkpoints/

    # Multiple epsilons:
    python test_apgd_l1.py --class_name bottle --epsilon_l1 500 1000 2000 5000 --checkpoint_dir ./checkpoints/
"""

import torch
from patchguard import PatchGuard
from utils import get_dataloader, load_model, patchify, label_patch, get_auc, display_results
from apgd_l1_patchguard import apgd_l1_attack_patchguard


def test_apgd_l1(model, test_loader, device, args, epsilon, n_iter, n_restarts):
    """Evaluate model under APGD L1 attack."""
    model.eval()

    test_scores = []
    test_labels = []
    test_masks = []

    for batch in test_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        masks_pixel = batch[2].to(device)

        # Convert pixel masks to patch masks for attack loss
        patch_masks = label_patch(patchify(masks_pixel, model.patch_size)).to(device)

        # Run APGD L1 attack
        with torch.set_grad_enabled(True):
            adv_images = apgd_l1_attack_patchguard(
                model=model,
                images=images,
                masks=patch_masks,
                epsilon=epsilon,
                n_iter=n_iter,
                n_restarts=n_restarts,
                device=device,
            )

        # Get scores on adversarial images
        with torch.no_grad():
            scores = model(adv_images)

        test_scores.append(scores.cpu())
        test_labels.append(labels.cpu())
        test_masks.append(masks_pixel.cpu())

    image_auc, pixel_auc = get_auc(
        test_scores, test_labels, test_masks,
        model.patches_per_side, args.smoothing_sigma, args.smoothing_radius, args.top_k
    )

    return image_auc, pixel_auc


def run_test_apgd_l1(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    model = PatchGuard(args, device).to(device)
    load_model(model, args.checkpoint_dir + f"patchguard_{args.dataset}_{args.class_name}.pth")
    _, test_loader = get_dataloader(
        args.image_size, args.dataset_dir, args.dataset, args.class_name,
        args.train_batch_size, args.test_batch_size, args.num_workers, args.seed
    )

    # Clean evaluation (no attack)
    from test_model import test
    image_auc, pixel_auc = test(model, test_loader, device, args, False)
    display_results({"Image-level AUC": image_auc, "Pixel-level AUC": pixel_auc}, "Clean Performance")

    # APGD L1 evaluation for each epsilon
    for epsilon in args.epsilon_l1:
        image_auc, pixel_auc = test_apgd_l1(
            model, test_loader, device, args, epsilon, args.n_iter_l1, args.n_restarts_l1
        )
        display_results(
            {"Image-level AUC": image_auc, "Pixel-level AUC": pixel_auc},
            f"APGD-L1 attack (eps={epsilon}, n_iter={args.n_iter_l1})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PatchGuard with APGD L1 attack")
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/mvtec")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--epsilon_l1", type=float, nargs='+', default=[2000.0])
    parser.add_argument("--n_iter_l1", type=int, default=500)
    parser.add_argument("--n_restarts_l1", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--smoothing_sigma", type=int, default=6)
    parser.add_argument("--smoothing_radius", type=int, default=7)

    # Feature extractor config (needed to build PatchGuard model)
    parser.add_argument("--hf_path", type=str, default='vit_small_patch14_dinov2.lvd142m')
    parser.add_argument("--feature_layers", type=int, nargs='+', default=[12])
    parser.add_argument("--reg_layers", type=int, nargs='+', default=[6, 9, 12])
    # Discriminator config
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dsc_layers", type=int, default=1)
    parser.add_argument("--dsc_heads", type=int, default=4)

    # These are needed for test_model.test() clean evaluation
    parser.add_argument("--attack_type", type=str, default="PGD")
    parser.add_argument("--adv_test", action="store_true", default=False)

    args = parser.parse_args()
    run_test_apgd_l1(args)