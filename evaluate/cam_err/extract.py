import json

def extract_cam01(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cam01_data = {}
    for frame, cams in data.items():
        if 'cam01' in cams:
            cam01_data[frame] = {'cam01': cams['cam01']}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cam01_data, f, indent=4, ensure_ascii=False)
    

# 示例用法：
extract_cam01('cam01-10_trans.json', 'c2w_cam01_trans.json')

