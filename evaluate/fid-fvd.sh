# cd到RecamMaster_plucker/evaluate/fid-metrics目录下之后运行python -m fid_metrics.main即可

# 参数设置在config.yaml : 调整sequence_length的值来设置视频序列长度
# 会按照config.yaml中metrics部分的type顺序依次计算数据集中的指标
# 输入视频图片帧集合
# 输出在hydra工作目录下的results.txt中


# FVD-V把同一timestep的多视图帧作为视频帧序列,计算多视图的一致性
# 计算FVD-V可以直接使用FVD部分的参数, 如将'real_video_frames'和'generated_video_frames'分别放入10个同一timestep的不同视角帧, 将FVD中的seq_len改为10即可
'''
比如说按这样的组织方式：
output_root/
  video001/
    t0000/
      frame_0000.jpg  # view0: real(起始视角)
      frame_0001.jpg  # view1: gen_view1
      frame_0002.jpg  # view2: gen_view2
      ...
    t0001/
      frame_0000.jpg
      frame_0001.jpg
      ...
  video002/
    t0000/
    t0001/
    ...
但理想情况下计算FVD-V的话需要同一time的ground truth视角帧, 即按如上组织出生成视频的同一场景的多视角帧, 将其视为静态视频, 
与真实情况下的该场景的多视角帧的静态视频二者计算FVD;
但现在没有这样的数据集, 1.先将原视频一个time的单帧拷贝构造出ground truth; 2. 在生成集内部计算, 即将每个t0000/分成两份计算FVD
'''
# 提取视频帧的脚本可参考fid-metrics中的extract_frames.py