# cd到RecamMaster_plucker/evaluate/fid-metrics目录下之后运行python -m fid_metrics.main即可

# 参数设置在config.yaml : 调整sequence_length的值来设置视频序列长度
# 会按照config.yaml中metrics部分的type顺序依次计算数据集中的指标
# 输入视频图片帧集合
# 输出在hydra工作目录下的results.txt中


# FVD-V把同一timestep的多视图帧作为视频帧序列,计算多视图的一致性
# 计算FVD-V可以直接使用FVD部分的参数, 如将'real_video_frames'和'generated_video_frames'分别放入10个同一timestep的不同视角帧, 将FVD中的seq_len改为10即可


# 提取视频帧的脚本可参考fid-metrics中的extract_frames.py