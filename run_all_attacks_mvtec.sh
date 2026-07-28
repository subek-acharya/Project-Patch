#!/bin/bash
# run_all_attacks_mvtec.sh
# Evaluate PatchGuard with all attack norms on both clean and adversarial models

CATEGORIES=(
    "bottle"
    "cable"
    "capsule"
    "carpet"
    "grid"
    "hazelnut"
    "leather"
    "metal_nut"
    "pill"
    "screw"
    "tile"
    "toothbrush"
    "transistor"
    "wood"
    "zipper"
)

DATASET_DIR="./datasets/mvtec"
RESULTS_DIR="./evaluation_results"
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  PatchGuard Multi-Norm Attack Evaluation"
echo "  Date: $(date)"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════
# 1. PGD L∞ (1000 steps, eps=8/255) - Original attack from paper
# ═══════════════════════════════════════════════════════════════

for MODEL_TYPE in "clean" "adv"; do
    CHECKPOINT_DIR="./checkpoints_${MODEL_TYPE}/"
    RESULTS_FILE="$RESULTS_DIR/results_pgd_linf_${MODEL_TYPE}.log"
    echo "PGD-Linf (1000 steps) Evaluation (${MODEL_TYPE} model) - $(date)" > "$RESULTS_FILE"

    for category in "${CATEGORIES[@]}"; do
        if [ ! -f "${CHECKPOINT_DIR}patchguard_mvtec_${category}.pth" ]; then
            echo "  ✗ Checkpoint not found for $category (${MODEL_TYPE}), skipping..." | tee -a "$RESULTS_FILE"
            continue
        fi

        echo "======================================" | tee -a "$RESULTS_FILE"
        echo "  PGD-Linf | ${MODEL_TYPE} | $category" | tee -a "$RESULTS_FILE"
        echo "  Start: $(date)" | tee -a "$RESULTS_FILE"
        echo "======================================" | tee -a "$RESULTS_FILE"

        python main.py --mode test --class_name "$category" --dataset mvtec \
            --dataset_dir "$DATASET_DIR" --checkpoint_dir "$CHECKPOINT_DIR" \
            --step_test 1000 --epsilon_test 8 2>&1 | tee -a "$RESULTS_FILE"

        echo "" | tee -a "$RESULTS_FILE"
    done
done

# ═══════════════════════════════════════════════════════════════
# 2. APGD L∞ (100 iterations, eps=8/255)
# ═══════════════════════════════════════════════════════════════

for MODEL_TYPE in "clean" "adv"; do
    CHECKPOINT_DIR="./checkpoints_${MODEL_TYPE}/"
    RESULTS_FILE="$RESULTS_DIR/results_apgd_linf_${MODEL_TYPE}.log"
    echo "APGD-Linf (100 iters) Evaluation (${MODEL_TYPE} model) - $(date)" > "$RESULTS_FILE"

    for category in "${CATEGORIES[@]}"; do
        if [ ! -f "${CHECKPOINT_DIR}patchguard_mvtec_${category}.pth" ]; then
            echo "  ✗ Checkpoint not found for $category (${MODEL_TYPE}), skipping..." | tee -a "$RESULTS_FILE"
            continue
        fi

        echo "======================================" | tee -a "$RESULTS_FILE"
        echo "  APGD-Linf | ${MODEL_TYPE} | $category" | tee -a "$RESULTS_FILE"
        echo "  Start: $(date)" | tee -a "$RESULTS_FILE"
        echo "======================================" | tee -a "$RESULTS_FILE"

        python test_apgd_linf.py \
            --class_name "$category" \
            --dataset mvtec \
            --dataset_dir "$DATASET_DIR" \
            --epsilon_linf 8 \
            --n_iter_linf 100 \
            --n_restarts_linf 1 \
            --checkpoint_dir "$CHECKPOINT_DIR" 2>&1 | tee -a "$RESULTS_FILE"

        echo "" | tee -a "$RESULTS_FILE"
    done
done

# ═══════════════════════════════════════════════════════════════
# 3. APGD L2 (100 iterations, eps=2.0)
# ═══════════════════════════════════════════════════════════════

for MODEL_TYPE in "clean" "adv"; do
    CHECKPOINT_DIR="./checkpoints_${MODEL_TYPE}/"
    RESULTS_FILE="$RESULTS_DIR/results_apgd_l2_${MODEL_TYPE}.log"
    echo "APGD-L2 (100 iters, eps=2.0) Evaluation (${MODEL_TYPE} model) - $(date)" > "$RESULTS_FILE"

    for category in "${CATEGORIES[@]}"; do
        if [ ! -f "${CHECKPOINT_DIR}patchguard_mvtec_${category}.pth" ]; then
            echo "  ✗ Checkpoint not found for $category (${MODEL_TYPE}), skipping..." | tee -a "$RESULTS_FILE"
            continue
        fi

        echo "======================================" | tee -a "$RESULTS_FILE"
        echo "  APGD-L2 | ${MODEL_TYPE} | $category" | tee -a "$RESULTS_FILE"
        echo "  Start: $(date)" | tee -a "$RESULTS_FILE"
        echo "======================================" | tee -a "$RESULTS_FILE"

        python test_apgd_l2.py \
            --class_name "$category" \
            --dataset mvtec \
            --dataset_dir "$DATASET_DIR" \
            --epsilon_l2 2.0 \
            --n_iter_l2 100 \
            --n_restarts_l2 1 \
            --checkpoint_dir "$CHECKPOINT_DIR" 2>&1 | tee -a "$RESULTS_FILE"

        echo "" | tee -a "$RESULTS_FILE"
    done
done

# ═══════════════════════════════════════════════════════════════
# 4. APGD L1 (100 iterations, eps=75)
# ═══════════════════════════════════════════════════════════════

for MODEL_TYPE in "clean" "adv"; do
    CHECKPOINT_DIR="./checkpoints_${MODEL_TYPE}/"
    RESULTS_FILE="$RESULTS_DIR/results_apgd_l1_${MODEL_TYPE}.log"
    echo "APGD-L1 (100 iters, eps=75) Evaluation (${MODEL_TYPE} model) - $(date)" > "$RESULTS_FILE"

    for category in "${CATEGORIES[@]}"; do
        if [ ! -f "${CHECKPOINT_DIR}patchguard_mvtec_${category}.pth" ]; then
            echo "  ✗ Checkpoint not found for $category (${MODEL_TYPE}), skipping..." | tee -a "$RESULTS_FILE"
            continue
        fi

        echo "======================================" | tee -a "$RESULTS_FILE"
        echo "  APGD-L1 | ${MODEL_TYPE} | $category" | tee -a "$RESULTS_FILE"
        echo "  Start: $(date)" | tee -a "$RESULTS_FILE"
        echo "======================================" | tee -a "$RESULTS_FILE"

        python test_apgd_l1.py \
            --class_name "$category" \
            --dataset mvtec \
            --dataset_dir "$DATASET_DIR" \
            --epsilon_l1 75 \
            --n_iter_l1 100 \
            --n_restarts_l1 1 \
            --checkpoint_dir "$CHECKPOINT_DIR" 2>&1 | tee -a "$RESULTS_FILE"

        echo "" | tee -a "$RESULTS_FILE"
    done
done

# ═══════════════════════════════════════════════════════════════
# 5. PGD L0 (100 steps, k=200, 10 restarts)
# ═══════════════════════════════════════════════════════════════

for MODEL_TYPE in "clean" "adv"; do
    CHECKPOINT_DIR="./checkpoints_${MODEL_TYPE}/"
    RESULTS_FILE="$RESULTS_DIR/results_pgd_l0_${MODEL_TYPE}.log"
    echo "PGD-L0 (100 steps, k=200) Evaluation (${MODEL_TYPE} model) - $(date)" > "$RESULTS_FILE"

    for category in "${CATEGORIES[@]}"; do
        if [ ! -f "${CHECKPOINT_DIR}patchguard_mvtec_${category}.pth" ]; then
            echo "  ✗ Checkpoint not found for $category (${MODEL_TYPE}), skipping..." | tee -a "$RESULTS_FILE"
            continue
        fi

        echo "======================================" | tee -a "$RESULTS_FILE"
        echo "  PGD-L0 | ${MODEL_TYPE} | $category" | tee -a "$RESULTS_FILE"
        echo "  Start: $(date)" | tee -a "$RESULTS_FILE"
        echo "======================================" | tee -a "$RESULTS_FILE"

        python test_pgd_l0.py \
            --class_name "$category" \
            --dataset mvtec \
            --dataset_dir "$DATASET_DIR" \
            --sparsity 200 \
            --num_steps_l0 100 \
            --step_size_l0 20.0 \
            --n_restarts_l0 10 \
            --checkpoint_dir "$CHECKPOINT_DIR" 2>&1 | tee -a "$RESULTS_FILE"

        echo "" | tee -a "$RESULTS_FILE"
    done
done

echo "═══════════════════════════════════════════════════════════════"
echo "  ALL EVALUATIONS COMPLETE!"
echo "  Finished: $(date)"
echo "  Results saved to: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════════════"