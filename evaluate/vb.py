from vbench import VBench

device = "cuda:7"  
full_info_json = "vbench/VBench_full_info.json"  
save_dir = "/data1/home/liu_kai/RecamMaster_plucker/evaluate/vbench"                 

my_VBench = VBench(
    device=device,
    full_info_dir=full_info_json,
    output_path=save_dir,
)

my_VBench.evaluate(
    videos_path="/data1/home/liu_kai/ReCamMaster/recam_result/cam_type9",  
    name="custom_eval",       # 这会决定输出文件名前缀
    dimension_list=[
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
    ],
    mode="custom_input",               
)
