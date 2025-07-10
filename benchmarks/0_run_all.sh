#!/bin/bash

set -e  # exit on first error

echo "==============================="
echo " AgoraLearn Benchmark Runner"
echo "==============================="

# Log the time
START_TIME=$(date)

################################################################################
# Experiment 1: Bias Term Estimation
################################################################################
echo "[Experiment 1] Estimating bias term..."
python 1_bias_term_estimation.py
echo "[✓] Finished Experiment 1"
echo

################################################################################
# Experiment 2: Criterion Comparison (Synthetic Data)
################################################################################
echo "[Experiment 2] Comparing collaboration criteria on synthetic data..."
python 2_criterions_comparison_synthetic_data.py
echo "[✓] Finished Experiment 2"
echo

################################################################################
# Experiment 3: Criterion Illustration (Real Data)
################################################################################
echo "[Experiment 3] Illustrating collaboration on real datasets..."

echo "→ Dataset: ccpp"
python 3_criterions_illustration_real_data.py --dataset_name ccpp

echo "→ Dataset: wine"
python 3_criterions_illustration_real_data.py --dataset_name wine

# echo "→ Dataset: clip"
# python 3_criterions_illustration_real_data.py --dataset_name clip

echo "[✓] Finished Experiment 3"
echo

# Summary
END_TIME=$(date)
echo "==============================="
echo "All experiments completed."
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
echo "==============================="
