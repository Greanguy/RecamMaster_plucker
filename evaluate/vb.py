from vbench import VBench

device = "cuda:0"  
full_info_json = "vbench/VBench_full_info.json"
save_dir = "/data1/home/liu_kai/RecamMaster_plucker/evaluate/vbench/plucker"                 

my_VBench = VBench(
    device=device,
    full_info_dir=full_info_json,
    output_path=save_dir,
)

cam_types = ["1", "3", "5", "7", "9"]

for cam_type in cam_types:
    videos_path = f"/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker/cam_type{cam_type}"
    name = f"cam{cam_type}_eval"
    
    print(f"eval-ing: {name}")
    
    my_VBench.evaluate(
        videos_path=videos_path,
        name=name,
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