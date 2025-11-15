# python ./clip-t-f-v/clip-t-f.py \
#   --video-root /data1/home/liu_kai/ReCamMaster/recam_result/cam_type9 \
#   --prompts-csv /data1/home/liu_kai/ReCamMaster/example_test_data/metadata.csv \
#   --csv-video-col file_name --csv-text-col text \
#   --model ViT-L-14  \
#   --pretrained-path ./clip-t-f-v/open_clip_model.safetensors \
#   --num-frames 81 --batch-size 128 \
#   --out ./clip-t-f-v/clip-t-f/results.csv \
#   --pair-root /data1/home/liu_kai/RecamMaster_plucker/evaluate/fvd_v_view_frames_gen \
#   --clipv-out ./clip-t-f-v/clip-t-f/clipv_results.csv \

# 该命令可将输入的测试视频文件夹的前len(prompts.csv)个视频(prompt与video.mp4按顺序一一对应)取出, 然后计算每个video与prompt的CLIP-T以及单个视频的CLIP-F, 输出为results.csv的每一行
# 如clip-t-f-v/clip-t-f/results.csv示例, 其中选择的生成视频与prompts不对应(使用的是完整prompt, results.csv只显示一部分), 可看到CILP-F值依然很高, 而CLIP-T偏低, 但由于描述的都是跳舞场景所以仍有部分数值
# clip-v输出结果在--clipv-out参数下的文件中

# Tips:
# 读取prompt.csv文件的时候,表头名字通过--csv-video-col file_name --csv-text-col text这两个参数设置
# --every-n表示每搁多少帧取一次frame,--num-frames表示一共需要均匀取多少帧,二者的效果一样设置一个就行
# --dump-per-frame pre_frame.csv 设置这个参数可以生成逐帧的相似度
# --video用来设置单个视频的文件地址,并且此时需要搭配--prompt使用; 要使用--prompts-csv需要搭配--video-root提供视频文件夹的路径
# 使用的模型是https://huggingface.co/timm/vit_large_patch14_clip_224.openai/tree/main

# CLIP-V即对同一timestep的多视图帧计算相似度, 计算方式相同
# --pair-root是生成视频与原视频在同一timestep时提取的视频帧对（可以通过RecamMaster_plucker/evaluate/fid-metrics/extract_frames.py代码所示的那样提取出同一timestep的多视图帧从而计算clip-v）
# 结构为：pair_root/
  # video001/
  #   t0000/
  #     frame_0000.jpg   # 原视频某一帧
  #     frame_0001.jpg   # 生成视频对应帧
  #   t0001/
  #     ...
  # video002/
  #   ...

# 也可以只计算clip-v或clip-t-f:
# python your_script.py \
#   --model ViT-L-14 \
#   --pair-root  \
#   --clipv-out \


#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=7
# 想跑的 cam_type 序号
CAMS="1 3 5 7 9"

BASE_VIDEO_ROOT="/data1/home/liu_kai/ReCamMaster/recam_result"
PROMPTS_CSV="/data1/home/liu_kai/ReCamMaster/example_test_data/metadata.csv"
PAIR_ROOT="/data1/home/liu_kai/RecamMaster_plucker/evaluate/recam_fvd_v_view_frames_gen_5"

MODEL="ViT-L-14"
PRETRAINED_PATH="./clip-t-f-v/open_clip_model.safetensors"

OUT_DIR="./clip-t-f-v/clip-t-f"
CLIPV_OUT="${OUT_DIR}/clipv_results_recam_5.csv"

mkdir -p "${OUT_DIR}"

for CAM in $CAMS; do
    VIDEO_ROOT="${BASE_VIDEO_ROOT}/cam_type${CAM}"
    OUT="${OUT_DIR}/recam/results_cam_type${CAM}.csv"

    # 简单检查一下视频目录是否存在，避免直接报错
    if [ ! -d "${VIDEO_ROOT}" ]; then
        echo "WARNING: ${VIDEO_ROOT} 不存在，跳过 cam_type${CAM}"
        continue
    fi

    echo "Running CLIP-T/F/V for ${VIDEO_ROOT}"

    python ./clip-t-f-v/clip-t-f.py \
      --video-root "${VIDEO_ROOT}" \
      --prompts-csv "${PROMPTS_CSV}" \
      --csv-video-col file_name --csv-text-col text \
      --model "${MODEL}"  \
      --pretrained-path "${PRETRAINED_PATH}" \
      --num-frames 81 --batch-size 128 \
      --out "${OUT}" 
done

python ./clip-t-f-v/clip-t-f.py \
  --model ViT-L-14 \
  --pair-root "${PAIR_ROOT}" \
  --clipv-out "${CLIPV_OUT}"

BASE_VIDEO_ROOT="/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker"
PROMPTS_CSV="/data1/home/liu_kai/ReCamMaster/example_test_data/metadata.csv"
PAIR_ROOT="/data1/home/liu_kai/RecamMaster_plucker/evaluate/plucker_fvd_v_view_frames_gen_5"

MODEL="ViT-L-14"
PRETRAINED_PATH="./clip-t-f-v/open_clip_model.safetensors"

OUT_DIR="./clip-t-f-v/clip-t-f"
CLIPV_OUT="${OUT_DIR}/clipv_results_plucker_5.csv"

mkdir -p "${OUT_DIR}"

for CAM in $CAMS; do
    VIDEO_ROOT="${BASE_VIDEO_ROOT}/cam_type${CAM}"
    OUT="${OUT_DIR}/plucker/results_cam_type${CAM}.csv"

    # 简单检查一下视频目录是否存在，避免直接报错
    if [ ! -d "${VIDEO_ROOT}" ]; then
        echo "WARNING: ${VIDEO_ROOT} 不存在，跳过 cam_type${CAM}"
        continue
    fi

    echo "Running CLIP-T/F/V for ${VIDEO_ROOT}"

    python ./clip-t-f-v/clip-t-f.py \
      --video-root "${VIDEO_ROOT}" \
      --prompts-csv "${PROMPTS_CSV}" \
      --csv-video-col file_name --csv-text-col text \
      --model "${MODEL}"  \
      --pretrained-path "${PRETRAINED_PATH}" \
      --num-frames 81 --batch-size 128 \
      --out "${OUT}" 
done

python ./clip-t-f-v/clip-t-f.py \
  --model ViT-L-14 \
  --pair-root "${PAIR_ROOT}" \
  --clipv-out "${CLIPV_OUT}"
