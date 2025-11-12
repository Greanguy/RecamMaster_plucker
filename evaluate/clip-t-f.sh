python ./clip-t-f-v/clip-t-f.py \
  --video-root ./test_result/cam_type1 \
  --prompts-csv ./test_data/metadata.csv \
  --csv-video-col file_name --csv-text-col text \
  --model ViT-L-14  \
  --pretrained-path ./clip-t-f-v/open_clip_model.safetensors \
  --num-frames 81 --batch-size 128 \
  --out ./clip-t-f-v/clip-t-f/results.csv

# 该命令可将输入的测试视频文件夹的前len(prompts.csv)个视频(prompt与video.mp4按顺序一一对应)取出, 然后计算每个video与prompt的CLIP-T以及单个视频的CLIP-F, 输出为results.csv的每一行
# 如clip-t-f-v/clip-t-f/results.csv示例, 其中选择的生成视频与prompts不对应(使用的是完整prompt, results.csv只显示一部分), 可看到CILP-F值依然很高, 而CLIP-T偏低, 但由于描述的都是跳舞场景所以仍有部分数值


# Tips:
# 读取prompt.csv文件的时候,表头名字通过--csv-video-col file_name --csv-text-col text这两个参数设置
# --every-n表示每搁多少帧取一次frame,--num-frames表示一共需要均匀取多少帧,二者的效果一样设置一个就行
# --dump-per-frame pre_frame.csv 设置这个参数可以生成逐帧的相似度
# --video用来设置单个视频的文件地址,并且此时需要搭配--prompt使用; 要使用--prompts-csv需要搭配--video-root提供视频文件夹的路径
# 使用的模型是https://huggingface.co/timm/vit_large_patch14_clip_224.openai/tree/main

# CLIP-V即对同一timestep的多视图帧计算相似度, 计算方式相同