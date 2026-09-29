#!/bin/bash
# Slurm template for examples/big on one node with several GPUs.
# Replace every <...> placeholder, then: sbatch examples/big/big.sbatch
#
# MODE=ddp   (default) one process per GPU, each with a full model copy;
#            needs the 4-bit model (~20 GB) to fit on every GPU.
# MODE=split one process, the model's layers spread over all GPUs
#            (--device_map unsloth_balanced); for models too large for
#            one GPU.
#
# DDP is validated for this model on 2x A100 40GB (~2.6 s per step,
# 29.7 GB peak); model splitting is validated with gpt-oss-120b
# (examples/gptoss).

#SBATCH --job-name=speftr-big
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

cd <path-to-your-clone>
# Models and datasets downloaded beforehand; compute nodes often have no
# internet. Offline, pass the unsloth/ model id or a local directory.
export HF_HOME=<shared-hf-cache>
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

MODE=${MODE:-ddp}
OUTPUT_DIR=./models/speftr-big-${SLURM_JOB_ID}

if [ "$MODE" = "ddp" ]; then
    # 2 GPUs x 4 rows x accumulation 2 = 16 rows per optimizer step. For
    # another GPU count, change --gres, --nproc_per_node and the
    # accumulation together to keep 16.
    uv run torchrun --standalone --nproc_per_node 2 \
        -m examples.big.big \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --output_dir "$OUTPUT_DIR"
else
    uv run python -m examples.big.big --device_map unsloth_balanced \
        --output_dir "$OUTPUT_DIR"
fi
