#!/usr/bin/env bash

# 自动批量运行 Replica 所有 8 个场景：
# 1) 运行 SLAM 到 800 帧
# 2) 导出高斯场景（PLY）
# 3) 从高斯场景提取 mesh（PLY/OBJ/TXT）
#
# 使用方式（推荐在已经激活 isogs 的终端里运行）：
#   从项目根目录运行：bash bash_scripts/run_replica_all_scenes.sh
#   或从任意目录运行：bash /path/to/SplaTAM-original/bash_scripts/run_replica_all_scenes.sh
#
# 如果你希望脚本自己激活 conda，请根据你本机路径修改 CONDA_BASE，再取消注释相关几行。

set -e

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

SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")

########################################
# 询问要执行的步骤
########################################
echo "================================================================"
echo "请选择要执行的步骤（可多选，用空格分隔，例如：1 2 3 或 3）："
echo "  1) 运行 SLAM 到 2000 帧"
echo "  2) 导出高斯场景（PLY）"
echo "  3) 从高斯场景提取 mesh"
echo "================================================================"
read -p "输入步骤编号: " selected_steps

# 检查用户输入
if [ -z "$selected_steps" ]; then
    echo "[Error] 未选择任何步骤，退出。"
    exit 1
fi

# 检查用户选择的步骤是否包含1、2、3
run_step1=false
run_step2=false
run_step3=false

for step in $selected_steps; do
    case $step in
        1)
            run_step1=true
            ;;
        2)
            run_step2=true
            ;;
        3)
            run_step3=true
            ;;
        *)
            echo "[Warn] 忽略无效的步骤编号: $step"
            ;;
    esac
done

if [ "$run_step1" = false ] && [ "$run_step2" = false ] && [ "$run_step3" = false ]; then
    echo "[Error] 没有选择任何有效的步骤，退出。"
    exit 1
fi

echo
echo "将执行以下步骤："
[ "$run_step1" = true ] && echo "  ✓ 步骤1: 运行 SLAM"
[ "$run_step2" = true ] && echo "  ✓ 步骤2: 导出 PLY"
[ "$run_step3" = true ] && echo "  ✓ 步骤3: 提取 mesh"
echo

########################################
# 步骤1: 对所有场景运行 SLAM
########################################
if [ "$run_step1" = true ]; then
echo "================================================================"
echo "[Step 1] Run SLAM for all scenes until frame 2000 ..."
echo "================================================================"
for IDX in "${!SCENES[@]}"; do
    SCENE_NAME="${SCENES[$IDX]}"
    echo "------------------------------------------------------------"
    echo "[Scene $((IDX + 1))/8] Index: $IDX, Name: $SCENE_NAME"
    echo "------------------------------------------------------------"
    
    # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
    export SPLATAM_SCENE_INDEX="$IDX"
    
    python scripts/splatam.py configs/replica/splatam.py --end-at 2000
    
    echo "[Done] SLAM finished for scene $SCENE_NAME (index $IDX)."
    echo
done
echo "[Step 1] All SLAM tasks completed."
echo
fi

########################################
# 步骤2: 对所有场景导出高斯 PLY
########################################
if [ "$run_step2" = true ]; then
echo "================================================================"
echo "[Step 2] Export Gaussian PLY for all scenes ..."
echo "================================================================"
for IDX in "${!SCENES[@]}"; do
    SCENE_NAME="${SCENES[$IDX]}"
    echo "------------------------------------------------------------"
    echo "[Scene $((IDX + 1))/8] Index: $IDX, Name: $SCENE_NAME"
    echo "------------------------------------------------------------"
    
    # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
    export SPLATAM_SCENE_INDEX="$IDX"
    
    python scripts/export_ply.py configs/replica/splatam.py
    
    echo "[Done] PLY export finished for scene $SCENE_NAME (index $IDX)."
    echo
done
echo "[Step 2] All PLY exports completed."
echo
fi

########################################
# 步骤3: 对所有场景提取 mesh（并行执行，每次2个场景）
########################################
if [ "$run_step3" = true ]; then
echo "================================================================"
echo "[Step 3] Extract mesh from Gaussian field for all scenes (parallel, 1 at a time) ..."
echo "================================================================"

# 定义并行执行mesh提取的函数
extract_mesh_for_scene() {
    local IDX=$1
    local SCENE_NAME="${SCENES[$IDX]}"
    local SCENE_NUM=$((IDX + 1))
    
    echo "[Scene $SCENE_NUM/8] Starting mesh extraction for: $SCENE_NAME (index $IDX)"
    
    # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
    export SPLATAM_SCENE_INDEX="$IDX"
    
    python scripts/extract_mesh_fast.py configs/replica/splatam.py \
        --voxel-size 0.015 \
        --iso-level 0.3 \
        --no-cleaning \
        --no-show
    
    echo "[Done] Mesh extraction finished for scene $SCENE_NAME (index $IDX)."
}

# 并行执行，每次最多2个任务
MAX_PARALLEL=1
pids=()

for IDX in "${!SCENES[@]}"; do
    # 如果当前运行的任务数达到最大值，等待一个任务完成
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        # 检查并移除已完成的进程
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                unset 'pids[$i]'
            fi
        done
        # 重新索引数组
        pids=("${pids[@]}")
        # 如果仍然达到最大值，等待一下
        if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
            sleep 1
        fi
    done
    
    # 在后台启动任务
    extract_mesh_for_scene "$IDX" &
    pids+=($!)
done

# 等待所有剩余的后台任务完成
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo
echo "[Step 3] All mesh extractions completed."
echo
fi

echo "================================================================"
echo "所有选定的步骤已完成。"
echo "================================================================"


