CUDA_VISIBLE_DEVICES=7 python ./gim/demo.py \
  --model gim_dkm \
  --generated_root ./gim/generated_videos \
  --real_root ./gim/real_videos \
  --frame_glob "frame_*.jpg" \
  --tau 0.5 \
  --csv_out ./gim/results/matpix_results.csv

# 使用的权重是gim_dkm
# --generated_root是生成视频的根目录,目录下包含"*_frames"名字的视频,每个视频是一个包含81帧视频帧的文件夹, 视频帧命名方式为--frame_glob
# --real_root同--generated_root的组织方式
# --tau表示特征匹配的置信度阈值
# --csv_out表示输出目录