CUDA_VISIBLE_DEVICES=7 vbench evaluate \
    --videos_path /data1/home/liu_kai/ReCamMaster/recam_result/cam_type9 \
    --dimension "temporal_flickering" \
    --mode=custom_input 
# 评估自己的视频时即--mode=custom_input(不需要提示集), 只支持下面的六个维度(都是visual quality层面的):
    # Note: We support customized videos / prompts for the following dimensions: 
    # 'subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'
# --mode=vbench_category是使用按类别的标准提示集评估(8类:Animal, Architecture, Food, Human, Lifestyle, Plant, Scenery, and Vehicles.) 
# --mode=vbench_standard是按维度的标准提示集评估(包含16维,包括7个视频质量评估维度和9个的视频条件一致性评估维度)

# 使用标准提示集评估的时候需要先按标准提示集让模型生成视频,每个类别/维度都包含100个提示
# recammaster评估时只使用vbench的视觉质量的6个维度, 其中5个不需要提示集, 只有Temporal Flickering这个维度vbench要求需要按提示集生成视频
# vbench的标准提示集在/data1/home/liu_kai/vbench/VBench_full_info.json

# recammaster使用的验证数据集:https://github.com/m-bain/webvid, 现在不提供url和字幕csv了, 除非以前下载过还能提取其中的部分视频