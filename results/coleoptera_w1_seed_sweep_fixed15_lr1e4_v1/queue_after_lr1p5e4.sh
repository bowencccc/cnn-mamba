#!/usr/bin/env bash
# Start the lr 1e-4 fixed-15 sweep once the lr 1.5e-4 sweep launcher (pid 204396) has exited
# (GPU memory cannot hold two cnn-mamba trainings next to OpenSpliceAI).
cd ~/cnn-mamba
while kill -0 204396 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] lr1.5e-4 sweep finished; starting lr1e-4 sweep"
SWEEP_TAG=coleoptera_w1_seed_sweep_fixed15_lr1e4_v1 SWEEP_CONFIG=configs/droso_coleoptera_w1_seeds_fixed15_lr1e4.json \
EVAL_CHECKPOINTS="last best" CNN_MAMBA_PYTHON=/home/bchen103/miniforge3/envs/cnn_mamba_k7/bin/python \
exec bash -x scripts/run_coleoptera_w1_seed_sweep.sh
