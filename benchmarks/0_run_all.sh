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
python 1_bias_term_estimation.py --T 100000 --steps 1000 --n_runs 5 --n_jobs 5 --joblib_verbose 100
echo "[✓] Finished Experiment 1"
echo

################################################################################
# Experiment 2: Criterion Comparison (Synthetic Data)
################################################################################
echo "[Experiment 2] Comparing collaboration criteria on synthetic data..."
python 2_criterions_comparison_synthetic_data.py --T 1000 --steps 100 --n_runs 5 --n_jobs 5 --joblib_verbose 100
echo "[✓] Finished Experiment 2"
echo

################################################################################
# Experiment 3: Criterion Illustration (Real Data)
################################################################################
echo "[Experiment 3] Illustrating collaboration on real datasets..."

echo "→ Dataset: ccpp"
python 3_criterions_illustration_real_data.py --dataset_name power --n_runs 5 --n_jobs 5 --joblib_verbose 100

echo "→ Dataset: wine"
python 3_criterions_illustration_real_data.py --dataset_name wine --n_runs 5 --n_jobs 5 --joblib_verbose 100

echo "→ Dataset: medical"
python 3_criterions_illustration_real_data.py --dataset_name medical --n_runs 5 --n_jobs 5 --joblib_verbose 100

echo "[✓] Finished Experiment 3"
echo

################################################################################
# Experiment 4: Criterion on gradients (Synthetic Data)
################################################################################
echo "[Experiment 4] Comparing collaboration criteria on gradient on synthetic data..."
python 4_criterion_on_gradient_synthetic_data.py --T 100000 --steps 1000 --nu 1e-5 --n_runs 5 --n_jobs 5 --joblib_verbose 100
echo "[✓] Finished Experiment 4"
echo

# Summary
END_TIME=$(date)
echo "==============================="
echo "All experiments completed."
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
echo "==============================="
