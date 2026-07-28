# PatchGuard Lacks Robustness: Breaking State-Of-The-Art Defect Detection Under New Norm Attacks

We provide code for evaluating the adversarial robustness of PatchGuard trained on MVTec AD and VisA anomaly detection datasets.
All attacks provided here are white-box attacks implemented in PyTorch and evaluated on both clean-trained and adversarially-trained PatchGuard models.

We provide attack code for  PGD-Linf, APGD-L1, APGD-L2, APGD-Linf, and PGD-L0 along with model architecture, training, and evaluation files.

---
## Setup

### Clone the Project
```bash
git clone https://github.com/subek-acharya/Project-Patch.git
cd PatchGuard
```

### Download Datasets and Foreground Masks
To prepare the datasets for training and evaluation, simply run the following command:
```bash
python download_data.py --dataset <DATASET_NAME>
```
The mask directory, named foreground_mask, will be placed alongside the training images folder. For example:
```swift
datasets/MVTec/toothbrush/train/
                            ├── good/
                            └── foreground_mask/
```

## Training

#### Train All MVTec Categories (Clean + Adversarial)
```code
bash train_all_mvtec.sh
```

#### Train Individual Category

```code
# Clean model
python main.py --mode train --class_name bottle --dataset mvtec --dataset_dir ./datasets/mvtec --checkpoint_dir ./checkpoints_clean/ --epochs 1000 --top_k 5 --no_adv_train

# Adversarial model
python main.py --mode train --class_name bottle --dataset mvtec --dataset_dir ./datasets/mvtec --checkpoint_dir ./checkpoints_adv/ --epochs 1000 --top_k 5 --step_train 50 --epsilon_train 8
```

## Evaluation

#### Run All Attacks on All Categories
```code
bash run_all_attacks.sh
```

#### Individual Attack Evaluation

### PGD ℓ∞ :
```code
python main.py --mode test --class_name bottle --dataset mvtec --dataset_dir ./datasets/mvtec --checkpoint_dir ./checkpoints_adv/ --step_test 1000 --epsilon_test 8
```

### APGD ℓ∞ :
```code
python test_apgd_linf.py --class_name bottle --epsilon_linf 8 --n_iter_linf 100 --checkpoint_dir ./checkpoints_adv/
```

### APGD ℓ₂ :
```code
python test_apgd_l2.py --class_name bottle --epsilon_l2 2.0 --n_iter_l2 100 --checkpoint_dir ./checkpoints_adv/
```

### APGD ℓ₁ :
```code
python test_apgd_l1.py --class_name bottle --epsilon_l1 75 --n_iter_l1 100 --checkpoint_dir ./checkpoints_adv/
```

### PGD ℓ₀ :
```code
python test_pgd_l0.py --class_name bottle --sparsity 200 --num_steps_l0 100 --step_size_l0 20.0 --n_restarts_l0 1 --checkpoint_dir ./checkpoints_adv/
```

