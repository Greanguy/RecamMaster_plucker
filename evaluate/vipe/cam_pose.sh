# vipe infer .mp4 --output 
# 处理单个视频或目录（streams.base_path 可以是文件或文件夹）    # streams.base_path=YOUR_VIDEO_OR_DIR_PATH \

# CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
#     streams.base_path=/data1/home/liu_kai/ReCamMaster/recam_result/cam_type3 \
#     pipeline.post.depth_align_model=null \
#     pipeline.output.save_viz=false pipeline.slam.visualize=false \
#     pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam03 \
#     pipeline.output.save_artifacts=true
# 其提取的外参是c2w

# python process_video.py ./vipe_res_cam01/pose/video0.npz cam01_v0.json --cam cam01
# 批量处理.npz文件并在各自所在目录下生成同名 json 如果想要把所有 json 输出到指定目录 out_dir（脚本会自动创建）：
# python process_video.py ./vipe_res_cam01/pose out_dir --cam cam01 
# python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam03/pose --cam cam03 # --prefix frame --inv  
# 用以处理vipe提取出的相机外参文件.npz(生成视频都是81帧的,所以vipe提取出的相机外侧也默认是81帧)
# 参数--cam指定输出.json文件的前缀名; vipe提取的外参是c2w矩阵, 想得到w2c矩阵可在运行process_video.py是加--inv参数, 取个逆即可

# CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
#     streams.base_path=/data1/home/liu_kai/ReCamMaster/recam_result/cam_type1 \
#     pipeline.post.depth_align_model=null \
#     pipeline.output.save_viz=false pipeline.slam.visualize=false \
#     pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam01 \
#     pipeline.output.save_artifacts=true
# python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam01/pose \
#     /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/cam01 --cam cam01

# CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
#     streams.base_path=/data1/home/liu_kai/ReCamMaster/recam_result/cam_type5 \
#     pipeline.post.depth_align_model=null \
#     pipeline.output.save_viz=false pipeline.slam.visualize=false \
#     pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam05 \
#     pipeline.output.save_artifacts=true
# python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam05/pose \
#     /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/cam05 --cam cam05

CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=/data1/home/liu_kai/ReCamMaster/recam_result/cam_type7 \
    pipeline.post.depth_align_model=null \
    pipeline.output.save_viz=false pipeline.slam.visualize=false \
    pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam07 \
    pipeline.output.save_artifacts=true
python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/vipe_res_cam07/pose \
    /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/cam07 --cam cam07

CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker/cam_type7 \
    pipeline.post.depth_align_model=null \
    pipeline.output.save_viz=false pipeline.slam.visualize=false \
    pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam07 \
    pipeline.output.save_artifacts=true
python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam07/pose \
    /data1/home/liu_kai/RecamMaster_plucker/evaluate/plucker/cam_err/cam07 --cam cam07

CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker/cam_type9 \
    pipeline.post.depth_align_model=null \
    pipeline.output.save_viz=false pipeline.slam.visualize=false \
    pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam09 \
    pipeline.output.save_artifacts=true
python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam09/pose \
    /data1/home/liu_kai/RecamMaster_plucker/evaluate/plucker/cam_err/cam09 --cam cam09

CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker/cam_type1 \
    pipeline.post.depth_align_model=null \
    pipeline.output.save_viz=false pipeline.slam.visualize=false \
    pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam01 \
    pipeline.output.save_artifacts=true
python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam01/pose \
    /data1/home/liu_kai/RecamMaster_plucker/evaluate/plucker/cam_err/cam01 --cam cam01

CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker/cam_type3 \
    pipeline.post.depth_align_model=null \
    pipeline.output.save_viz=false pipeline.slam.visualize=false \
    pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam03 \
    pipeline.output.save_artifacts=true
python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam03/pose \
    /data1/home/liu_kai/RecamMaster_plucker/evaluate/plucker/cam_err/cam03 --cam cam03

CUDA_VISIBLE_DEVICES=0 python run.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker/cam_type5 \
    pipeline.post.depth_align_model=null \
    pipeline.output.save_viz=false pipeline.slam.visualize=false \
    pipeline.output.path=/data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam05 \
    pipeline.output.save_artifacts=true
python process_video.py /data1/home/liu_kai/RecamMaster_plucker/evaluate/vipe/plucker_res_cam05/pose \
    /data1/home/liu_kai/RecamMaster_plucker/evaluate/plucker/cam_err/cam05 --cam cam05