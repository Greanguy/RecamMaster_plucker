# python ./cam_err/cam_err.py \
#   --gt ./cam_err/c2w_cam01_trans.json \
#   --est ./vipe/cam01_v0.json \
#   --key cam09 \
#   --out ./cam_err/cam09.csv \
#   --convention cam2world   # or world2cam, vipe提取的视频外参是c2w矩阵, 同时需要保证ground true的外参也是c2w矩阵并且坐标轴保持一致

# python ./cam_err/cam_err.py \
#   --gt /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/c2w_cam03_trans.json \
#   --est_dir /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/cam03 \
#   --key cam03 \
#   --convention cam2world \
#   --csv_out /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/cam03_all.csv \
#   --ignore_first

# 注意坐标轴序列和方向就行了,做cam_err估计的时候
# vipe估计出来的pose是c2w矩阵,旋转矩阵是单位阵的格式:[1,0,0][0,1,0][0,0,1] 位移向量应该都是[tx,ty,tz]应该不需要管,一般处理的时候不会调换它们的位置
# recammaster原始相机轨迹也是c2w矩阵(做了个转置)

#!/usr/bin/env bash
set -e

# 需要评估的相机编号
CAMS="01 03 05 07 09"

BASE_DIR="/data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err"

for CAM in $CAMS; do
    GT="${BASE_DIR}/c2w_cam${CAM}_trans.json"
    EST_DIR="${BASE_DIR}/plucker/cam${CAM}"
    KEY="cam${CAM}"
    CSV_OUT="${BASE_DIR}/plucker_cam${CAM}_all.csv"

    if [ ! -f "${GT}" ]; then
        echo "WARNING: GT json 不存在: ${GT}，跳过 cam${CAM}"
        continue
    fi
    if [ ! -d "${EST_DIR}" ]; then
        echo "WARNING: 估计结果目录不存在: ${EST_DIR}，跳过 cam${CAM}"
        continue
    fi

    echo "Running cam_err for ${KEY}"

    python ./cam_err/cam_err.py \
      --gt "${GT}" \
      --est_dir "${EST_DIR}" \
      --key "${KEY}" \
      --convention cam2world \
      --csv_out "${CSV_OUT}" \
      --ignore_first
done
