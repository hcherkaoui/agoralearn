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
python 1_bias_term_estimation.py --d 20 --sigma 1.0 --T 1000 --min_T 100 --n_runs 10 --n_jobs -1 --joblib_verbose 100 --seed 0
echo "[✓] Finished Experiment 1"
echo

################################################################################
# Experiment 2: Criterion Comparison (Synthetic Data)
################################################################################
echo "[Experiment 2] Comparing collaboration criteria on synthetic data..."

echo "→ Homoskedastic case"
python 2_criterions_comparison_synthetic_data.py --d 20 --sigma_1 1.0 --sigma_2 1.0 --T 1000 --min_T 100  --n_runs 10 --n_jobs -1 --joblib_verbose 100 --seed 0

echo "→ Heteroskedastic case"
python 2_criterions_comparison_synthetic_data.py --d 20 --sigma_1 5.0 --sigma_2 0.5 --T 1000 --min_T 100  --n_runs 10 --n_jobs -1 --joblib_verbose 100 --seed 0

echo "[✓] Finished Experiment 2"
echo

################################################################################
# Experiment 3: Criterion Illustration (Real Data)
################################################################################
echo "[Experiment 3] Illustrating collaboration on real datasets..."

# echo "→ Dataset: ccpp"
# python 3_criterions_illustration_real_data.py --dataset_name ccpp --delta 0.99 --beta 0.01 --n_runs 10 --n_jobs -1 --joblib_verbose 100 --seed 0

echo "→ Dataset: wine"
python 3_criterions_illustration_real_data.py --dataset_name wine --delta 0.99 --beta 0.01 --n_runs 10 --n_jobs -1 --joblib_verbose 100 --seed 0

# echo "→ Dataset: clip"
# python 3_criterions_illustration_real_data.py --dataset_name clip --delta 0.99 --beta 0.01 --n_runs 10 --n_jobs -1 --joblib_verbose 100 --seed 0

echo "[✓] Finished Experiment 3"
echo

# Summary
END_TIME=$(date)
echo "==============================="
echo "All experiments completed."
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
echo "==============================="
