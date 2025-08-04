#!/bin/bash

set -e  # exit on first error

echo "==============================================================================="
echo " AgoraLearn Benchmark Runner"

# Log the time
START_TIME=$(date)

# ################################################################################
# # Experiment 1: Gradient investigation
# ################################################################################
# echo "==============================================================================="
# echo "[Experiment 1] Gradient investigation..."
# python bench_1_criterion_on_gradient_real_data.py --n_iters 150 --lr 1e-3
# echo "[✓] Finished Experiment 1"
# echo

###############################################################################
# Experiment 2: Learning comparison
###############################################################################
echo "==============================================================================="
echo "[Experiment 2] Learning comparison..."
python bench_2_launch_criterion_on_gradient_comparison_real_data.py --n_epochs 25 --gradient_mode no_additional_grad --train_size 7500 --test_size 100
python bench_2_launch_criterion_on_gradient_comparison_real_data.py --n_epochs 25 --gradient_mode additional_grad --train_size 7500 --test_size 100
python bench_2_launch_criterion_on_gradient_comparison_real_data.py --n_epochs 25 --gradient_mode dynamic_additional_grad --train_size 7500 --test_size 100
python bench_2_plot_criterion_on_gradient_comparison_real_data.py
echo "[✓] Finished Experiment 2"
echo

# Summary
END_TIME=$(date)
echo "==============================================================================="
echo "All experiments completed."
echo "Started at: $START_TIME"
echo "Finished at: $END_TIME"
