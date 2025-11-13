CUDA_VISIBLE_DEVICES=7 python demo.py \
  --model gim_dkm \
  --generated_root /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam09_gen_videos \
  --real_root /data1/home/liu_kai/RecamMaster_plucker/evaluate/real_videos \
  --frame_glob "frame_*.jpg" \
  --tau 0.3 \
  --csv_out results/matpix_results_tau03.csv

# 使用的权重是gim_dkm
# --generated_root是生成视频的根目录,目录下包含多个视频,每个视频是一个包含81帧视频帧的文件夹, 视频帧命名方式为--frame_glob
# --real_root同--generated_root的组织方式
# --tau表示特征匹配的置信度阈值
# --csv_out表示输出目录