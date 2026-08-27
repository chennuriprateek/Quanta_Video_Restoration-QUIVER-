#!/bin/bash
#SBATCH --job-name=quiver_distill
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
#SBATCH --account=stanchan
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00

# ---------------------------------------------------------------------------
# Paths — edit these before submitting
# ---------------------------------------------------------------------------
REPO_ROOT="/scratch/gilbreth/chen4848/projects/Quanta_Video_Restoration-QUIVER-"
TRAIN_DATA_DIR="/scratch/gilbreth/chen4848/datasets/i2-2kfps_v1/train"
VAL_DATA_DIR="/scratch/gilbreth/chen4848/datasets/i2-2kfps_v1/test"
TEACHER_WEIGHTS="$REPO_ROOT/weights_teacher/quiver_best_p5f5_3.25PPP.pth"
WEIGHTS_DIR="$REPO_ROOT/weights_student_3.25PPP"
PLOT_DIR="$REPO_ROOT/plots_student_3.25PPP"
SPYNET_PATH=""  # QUIVER trains SpyNet from scratch; do not load mismatched VRT weights

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p logs "$WEIGHTS_DIR" "$PLOT_DIR"
export PYTHONUNBUFFERED=1

# Activate the QUIVER conda environment by prepending its bin to PATH
export PATH="/home/chen4848/.conda/envs/quiver/bin:$PATH"

# Install torch if missing (the nightly pinned in QUIVER_environment.yml is no longer hosted)
python -c "import torch" 2>/dev/null || \
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121 --quiet

# ---------------------------------------------------------------------------
# Student architecture  (reduce n_features / n_blocks to compress)
# ---------------------------------------------------------------------------
STUDENT_N_FEATURES=32
STUDENT_N_BLOCKS=6

# Teacher architecture  (must match the checkpoint being loaded)
TEACHER_N_FEATURES=64
TEACHER_N_BLOCKS=12

# ---------------------------------------------------------------------------
# Distillation loss weights
#   lambda_kd_hf3    — RDBCell bottleneck features   (Tier 1A, highest weight)
#   lambda_kd_att    — spatial_att output             (Tier 1B)
#   lambda_kd_hidden — recurrent hidden state s       (Tier 1C)
#   lambda_kd_warp   — aligned/fused warp features    (Tier 2D)
#   lambda_task      — standard task loss multiplier
# ---------------------------------------------------------------------------
LAMBDA_KD_HF3=1.0
LAMBDA_KD_ATT=0.5
LAMBDA_KD_HIDDEN=0.5
LAMBDA_KD_WARP=0.25
LAMBDA_TASK=1.0

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_ROOT/code/quiver"

python quiver_qis_distill_train.py \
    --gtdata_dir      "$TRAIN_DATA_DIR" \
    --valgtdata_dir   "$VAL_DATA_DIR" \
    --weights_dir     "$WEIGHTS_DIR" \
    --plotdir         "$PLOT_DIR" \
    --spynet_path     "$SPYNET_PATH" \
    --n_features      $STUDENT_N_FEATURES \
    --n_blocks        $STUDENT_N_BLOCKS \
    --teacher_weights     "$TEACHER_WEIGHTS" \
    --teacher_n_features  $TEACHER_N_FEATURES \
    --teacher_n_blocks    $TEACHER_N_BLOCKS \
    --lambda_kd_hf3   $LAMBDA_KD_HF3 \
    --lambda_kd_att   $LAMBDA_KD_ATT \
    --lambda_kd_hidden $LAMBDA_KD_HIDDEN \
    --lambda_kd_warp  $LAMBDA_KD_WARP \
    --lambda_task     $LAMBDA_TASK \
    --model_name      quiver \
    --loss_fun_name   L1_grad \
    --lr              0.0001 \
    -batch_size       8 \
    -total_epochs     1500 \
    -save_period      300 \
    --load_spynet_weights False \
    --visualize       True \
    -FWC 200 \
    -avg_PPP 3.25 \
    -gain 1.0769 \
    -Nbits 3 \
    -QE 0.8 \
    -theta_dark 1.6 \
    -sigma_read 0.2 \
    -clicks_per_frame 1
