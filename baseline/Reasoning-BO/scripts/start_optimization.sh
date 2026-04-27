#!/bin/sh

echo "start"

export SWANLAB_API_KEY="tSGp3IsD7uFZHaBax6NC4"
export SWANLAB_WORKSPACE="ovo"
# experiment_name 名为 model_轮次_日期_taskid
export SWANLAB_EXP_NAME="qwq"  # 添加任务ID到实验名

conda init
conda activate bo

RESULT_DIR="./data/results"
MODEL_PATH="/home/bingxing2/ailab/group/ai4phys/our-qwen2.5-14b"

python scripts/experiment.py \
    --exp_config_path "./src/config/suzuki_config.json" \
    --metric_name "suzuki" \
    --model_type  "qwq-plus-32b"  \
    --reasoner  "qwq"  \
    --result_dir "${RESULT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --num_iterations 30  \
    --seed 42 \
    --ablation_mode "reasoning_first" \
    # --enable_notes
