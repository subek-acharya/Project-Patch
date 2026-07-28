#!/bin/bash
# train_all_mvtec.sh

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
EPOCHS=1000
LOG_DIR="./training_logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/training_master.log"
echo "========================================================" > "$MASTER_LOG"
echo "  PatchGuard MVTec Training - All Categories" >> "$MASTER_LOG"
echo "  Date: $(date)" >> "$MASTER_LOG"
echo "  Epochs: $EPOCHS" >> "$MASTER_LOG"
echo "========================================================" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

for category in "${CATEGORIES[@]}"; do
    # ─── Clean Model Training ───
    echo "======================================" | tee -a "$MASTER_LOG"
    echo "  Training CLEAN model: $category" | tee -a "$MASTER_LOG"
    echo "  Start: $(date)" | tee -a "$MASTER_LOG"
    echo "======================================" | tee -a "$MASTER_LOG"
    
    CLEAN_LOG="$LOG_DIR/clean_${category}.log"
    
    python main.py \
        --mode train \
        --class_name "$category" \
        --dataset mvtec \
        --dataset_dir "$DATASET_DIR" \
        --checkpoint_dir ./checkpoints_clean/ \
        --epochs $EPOCHS \
        --top_k 5 \
        --no_adv_train 2>&1 | tee "$CLEAN_LOG"
    
    echo "  End: $(date)" | tee -a "$MASTER_LOG"
    echo "  Log: $CLEAN_LOG" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

    # ─── Adversarial Model Training ───
    echo "======================================" | tee -a "$MASTER_LOG"
    echo "  Training ADVERSARIAL model: $category" | tee -a "$MASTER_LOG"
    echo "  Start: $(date)" | tee -a "$MASTER_LOG"
    echo "======================================" | tee -a "$MASTER_LOG"
    
    ADV_LOG="$LOG_DIR/adv_${category}.log"
    
    python main.py \
        --mode train \
        --class_name "$category" \
        --dataset mvtec \
        --dataset_dir "$DATASET_DIR" \
        --checkpoint_dir ./checkpoints_adv/ \
        --epochs $EPOCHS \
        --top_k 5 \
        --step_train 50 \
        --epsilon_train 8 2>&1 | tee "$ADV_LOG"
    
    echo "  End: $(date)" | tee -a "$MASTER_LOG"
    echo "  Log: $ADV_LOG" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"

done

echo "========================================================" | tee -a "$MASTER_LOG"
echo "  All training complete!" | tee -a "$MASTER_LOG"
echo "  Finished: $(date)" | tee -a "$MASTER_LOG"
echo "========================================================" | tee -a "$MASTER_LOG"