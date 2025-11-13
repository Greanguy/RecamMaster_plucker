# vipe infer .mp4 --output 
# 处理单个视频或目录（streams.base_path 可以是文件或文件夹）    # streams.base_path=YOUR_VIDEO_OR_DIR_PATH \

CUDA_VISIBLE_DEVICES=7 python run.py pipeline=default streams=raw_mp4_stream \
    streams.base_path=/data1/home/liu_kai/ReCamMaster/recam_result/cam_type9 \
    pipeline.post.depth_align_model=null \
    pipeline.output.save_viz=false pipeline.slam.visualize=false \
    pipeline.output.path=/data1/home/liu_kai/vipe/vipe_res_cam09 \
    pipeline.output.save_artifacts=true
# 其提取的外参是c2w

# python process_video.py ./vipe_res_cam01/pose/video0.npz cam01_v0.json --cam cam01
# python process_video.py /data1/home/liu_kai/vipe/vipe_res_cam09/pose --cam cam09 --prefix frame --inv  # 批量处理.npz文件并在各自所在目录下生成同名 json
# 用以处理vipe提取出的相机外参文件.npz(生成视频都是81帧的,所以vipe提取出的相机外侧也默认是81帧)
# 参数--cam指定输出.json文件的前缀名; vipe提取的外参是c2w矩阵, 想得到w2c矩阵可在运行process_video.py是加--inv参数, 取个逆即可