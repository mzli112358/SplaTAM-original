#!/usr/bin/env bash

# 自动批量运行 Replica (V1) 所有 8 个场景的评估：
# 1) 训练集评估 (Train Split Eval)
# 2) 新视角合成评估 (Novel View Synthesis Eval)
#
# 使用方式（推荐在已经激活 isogs 的终端里运行）：
#   从项目根目录运行：bash bash_scripts/run_replica_eval_all.sh
#   或从任意目录运行：bash /path/to/SplaTAM-original/bash_scripts/run_replica_eval_all.sh
#
# 如果你希望脚本自己激活 conda，请根据你本机路径修改 CONDA_BASE，再取消注释相关几行。

# 不使用 set -e，允许单个任务失败时继续执行其他任务
# set -e

# 自动检测项目根目录（脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

########################################
# 自动激活 conda 环境
########################################
# 尝试多个可能的 conda 路径
CONDA_PATHS=(
    "$HOME/anaconda3"
    "$HOME/miniconda3"
    "/opt/conda"
)

CONDA_BASE=""
for path in "${CONDA_PATHS[@]}"; do
    if [ -f "$path/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$path"
        break
    fi
done

if [ -n "$CONDA_BASE" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    # 检查是否已经在 isogs 环境中
    if [ "$CONDA_DEFAULT_ENV" != "isogs" ]; then
        echo "[Info] Activating conda environment: isogs"
        conda activate isogs
        if [ $? -ne 0 ]; then
            echo "[Error] Failed to activate 'isogs' environment. Please activate it manually."
            exit 1
        fi
    else
        echo "[Info] Already in 'isogs' environment."
    fi
else
    echo "[Warn] conda.sh not found in common locations."
    echo "[Warn] Please activate 'isogs' environment manually before running."
    echo "[Warn] Trying to continue anyway..."
fi

cd "$PROJECT_ROOT"

# 8个场景 (Replica V1 命名：room0, office0 等)
SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")
NUM_SCENES=${#SCENES[@]}

# 两种评估模式
EVAL_MODES=("train-split" "nvs")
MODE_NAMES=("训练集评估" "新视角合成评估")

echo "================================================================"
echo "开始批量评估 Replica (V1) 数据集"
echo "场景数量: $NUM_SCENES"
echo "评估模式: 2 种"
echo "总计任务: $((NUM_SCENES * 2))"
echo "================================================================"
echo

TOTAL_TASKS=$((NUM_SCENES * 2))
CURRENT_TASK=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 遍历所有场景
for SCENE_IDX in "${!SCENES[@]}"; do
    SCENE_NAME="${SCENES[$SCENE_IDX]}"
    
    echo "================================================================"
    echo "[场景 $((SCENE_IDX + 1))/$NUM_SCENES] $SCENE_NAME"
    echo "================================================================"
    
    # 设置环境变量，让配置文件知道使用哪个场景
    export SCENE="$SCENE_IDX"
    
    # 遍历两种评估模式
    for MODE_IDX in "${!EVAL_MODES[@]}"; do
        MODE="${EVAL_MODES[$MODE_IDX]}"
        MODE_NAME="${MODE_NAMES[$MODE_IDX]}"
        CURRENT_TASK=$((CURRENT_TASK + 1))
        
        echo "------------------------------------------------------------"
        echo "[任务 $CURRENT_TASK/$TOTAL_TASKS] $SCENE_NAME - $MODE_NAME"
        echo "------------------------------------------------------------"
        
        # 根据模式选择参数
        if [ "$MODE" == "train-split" ]; then
            EVAL_ARG="--train-split"
        else
            EVAL_ARG="--nvs"
        fi
        
        # 运行评估
        echo "执行命令: python scripts/eval_novel_view.py configs/replica/replica_rendering_eval.py $EVAL_ARG"
        START_TIME=$(date +%s)
        if python scripts/eval_novel_view.py configs/replica/replica_rendering_eval.py $EVAL_ARG; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "✓ 成功完成: $SCENE_NAME - $MODE_NAME (耗时: ${ELAPSED}秒)"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "✗ 失败: $SCENE_NAME - $MODE_NAME (耗时: ${ELAPSED}秒)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "继续执行下一个任务..."
        fi
        
        echo
    done
    
    echo "[完成] 场景 $SCENE_NAME 的所有评估已完成"
    echo
done

echo "================================================================"
echo "所有 Replica (V1) 场景的评估已完成！"
echo "================================================================"
echo
echo "统计信息:"
echo "  总任务数: $TOTAL_TASKS"
echo "  成功: $SUCCESS_COUNT"
echo "  失败: $FAIL_COUNT"
echo
echo "结果保存在: $PROJECT_ROOT/experiments/Replica/<scene_name>_0/eval_*/"
echo "  - eval_train/ : 训练集评估结果"
echo "  - eval_nvs/   : 新视角合成评估结果"

