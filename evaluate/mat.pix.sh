# CUDA_VISIBLE_DEVICES=7 python demo.py \
#   --model gim_dkm \
#   --generated_root /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam01_gen_videos_recam \
#   --real_root /data1/home/liu_kai/RecamMaster_plucker/evaluate/real_videos \
#   --frame_glob "frame_*.jpg" \
#   --tau 0.5 \
#   --csv_out results/matpix_results.csv

#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=7

REAL_ROOT="/data1/home/liu_kai/RecamMaster_plucker/evaluate/real_videos"
FRAME_GLOB="frame_*.jpg"
TAU=0.5
OUT_DIR="results"

mkdir -p "${OUT_DIR}"

# 相机编号
for CAM in 01 03 05 07 09; do
    GEN_ROOT="/data1/home/liu_kai/RecamMaster_plucker/evaluate/cam${CAM}_gen_videos_recam"
    CSV_OUT="${OUT_DIR}/matpix_cam${CAM}.csv"

    echo "Running Mat.Pix for generated_root=${GEN_ROOT}"

    python demo.py \
      --model gim_dkm \
      --generated_root "${GEN_ROOT}" \
      --real_root "${REAL_ROOT}" \
      --frame_glob "${FRAME_GLOB}" \
      --tau "${TAU}" \
      --csv_out "${CSV_OUT}"
done

# 使用的权重是gim_dkm
# --generated_root是生成视频的根目录,目录下包含多个视频,每个视频是一个包含81帧视频帧的文件夹, 视频帧命名方式为--frame_glob
# --real_root同--generated_root的组织方式
# --tau表示特征匹配的置信度阈值
# --csv_out表示输出目录