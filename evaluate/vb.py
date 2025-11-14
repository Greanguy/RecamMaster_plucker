from vbench import VBench

device = "cuda:0"  
full_info_json = "vbench/VBench_full_info.json"  # 只有在使用提示集生成的视频的时候需要用,即在mode为vbench_standard或者vbench_category的时候
save_dir = "/data1/home/liu_kai/RecamMaster_plucker/evaluate/vbench"                 

my_VBench = VBench(
    device=device,
    full_info_dir=full_info_json,
    output_path=save_dir,
)

my_VBench.evaluate(
    videos_path="/data1/home/liu_kai/ReCamMaster/recam_result/cam_type3",  
    name="cam03_eval",       # 这会决定输出文件名前缀
    dimension_list=[
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "temporal_flickering",
    ],
    mode="custom_input",               
)
