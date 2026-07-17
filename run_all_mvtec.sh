#!/bin/bash
# run_all_mvtec.sh
# Evaluate PatchGuard on all MVTec AD categories with PGD-1000 attack

RESULTS_FILE="evaluation_results.log"
echo "========================================================" > "$RESULTS_FILE"
echo "  PatchGuard MVTec AD Evaluation - PGD-1000 (eps=8/255)" >> "$RESULTS_FILE"
echo "  Date: $(date)" >> "$RESULTS_FILE"
echo "========================================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

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

DATASET_DIR="./datasets/MVTec"
CHECKPOINT_DIR="./checkpoints/"
STEP_TEST=1000
EPSILON_TEST=8

for category in "${CATEGORIES[@]}"; do
    echo "======================================" | tee -a "$RESULTS_FILE"
    echo "  Evaluating: $category" | tee -a "$RESULTS_FILE"
    echo "======================================" | tee -a "$RESULTS_FILE"
    
    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINT_DIR}patchguard_mvtec_${category}.pth" ]; then
        echo "  ✗ Checkpoint not found for $category, skipping..." | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
        continue
    fi
    
    # Check if dataset exists
    if [ ! -d "${DATASET_DIR}/${category}" ]; then
        echo "  ✗ Dataset not found for $category, skipping..." | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
        continue
    fi
    
    python main.py \
        --mode test \
        --class_name "$category" \
        --dataset mvtec \
        --dataset_dir "$DATASET_DIR" \
        --step_test $STEP_TEST \
        --epsilon_test $EPSILON_TEST \
        --checkpoint_dir "$CHECKPOINT_DIR" 2>&1 | tee -a "$RESULTS_FILE"
    
    echo "" | tee -a "$RESULTS_FILE"
done

echo "========================================================" | tee -a "$RESULTS_FILE"
echo "  All evaluations complete!" | tee -a "$RESULTS_FILE"
echo "  Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "========================================================" | tee -a "$RESULTS_FILE"