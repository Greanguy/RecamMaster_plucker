import os
import cv2
import numpy as np

# 单个视频提取指定数量的帧，保存到 output_folder
def extract_frames(video_path, output_folder, num_frames=81):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1
    cap.release()

# 从一个文件夹中提取所有视频的帧
def extract_frames_from_dir(video_dir,output_root,num_frames=81,exts=(".mp4", ".avi", ".mov", ".mkv", ".webm"),):
    """
    video_dir: 输入视频所在文件夹
    output_root: 所有视频帧输出的根目录
    num_frames: 每个视频要提取的帧数
    exts: 视频文件的后缀
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for fname in os.listdir(video_dir):
        # 跳过非视频文件
        if not fname.lower().endswith(exts):
            continue
        video_path = os.path.join(video_dir, fname)
        video_name = os.path.splitext(fname)[0]  # 去掉后缀作为视频名
        output_folder = os.path.join(output_root, video_name)
        # print(f"Processing {video_path} -> {output_folder}")
        extract_frames(video_path, output_folder, num_frames=num_frames)


def _read_video_frames(video_path, max_frames=None):
    """
    读入一个视频前 max_frames 帧，返回 [frame0, frame1, ...] 的列表。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames

# 从真实视频目录和多个生成视频目录中，构造“同一 timestep 下不同视角”的伪视频（帧目录形式）
def build_view_pseudo_videos_as_frame_dirs(
    real_dir,
    gen_dirs,
    output_root,
    max_frames=81,
    num_timesteps=None,  # 想要的 timestep 数；None 表示用所有可用
    exts=(".mp4", ".avi", ".mov", ".mkv", ".webm"),
):
    """
    从 real_dir 和若干个 gen_dirs 中构造“同一 timestep 下不同视角”的伪视频，
    以“帧目录”的形式保存

    - real_dir 下的真实视频按文件名排序 -> real_files[0], real_files[1], ...
    - 对每个 gen_dir同样按文件名排序 -> gen_files_list[g_idx][i]
    - 第 i 个真实视频 real_files[i] 与每个 gen_dir 中的第 i 个生成视频一一对应

    输出目录结构示意：
    output_root/
      clip001/
        t0000/
          frame_0000.jpg  # 视角0：real
          frame_0001.jpg  # 视角1：gen_view1
          frame_0002.jpg  # 视角2：gen_view2
          ...
        t0001/
          frame_0000.jpg
          frame_0001.jpg
          ...
      clip002/
        t0000/
        t0001/
        ...

    这样，每个 tXXXX 目录就可以被当作一个“长度 = 视角数”的伪视频。
    """
    if isinstance(gen_dirs, str):
        # 允许传入单个目录字符串
        gen_dirs = [gen_dirs]

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 找出所有真实视频
    real_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(exts)])
    num_real = len(real_files)
    print(f"[INFO] Found {num_real} real videos in {real_dir}")
    
    # 对每个 gen_dir 也按名字排序收集视频列表
    gen_files_list = []  # 每个元素是该 gen_dir 下的 [f0, f1, ...]
    for gdir in gen_dirs:
        gfiles = sorted([f for f in os.listdir(gdir) if f.lower().endswith(exts)])
        print(f"[INFO] Found {len(gfiles)} generated videos in {gdir}")
        if len(gfiles) < num_real:
            print(
                f"[WARN] generated videos in {gdir} fewer than real videos: "
                f"{len(gfiles)} < {num_real}. Will only use first {len(gfiles)} pairs."
            )
        gen_files_list.append(gfiles)

    # 能够匹配的 clip 数量 = real_files 数量 和 每个 gen_dir 视频数 的最小值
    if gen_files_list:
        max_pairs = min(
            num_real,
            min(len(gfiles) for gfiles in gen_files_list)
        )
    else:
        # 没有 gen_dirs 的极端情况：只用真实视角
        max_pairs = num_real

    if max_pairs == 0:
        print("[WARN] No usable real/gen pairs, return.")
        return

    print(f"[INFO] Will build {max_pairs} clips (pairs by index).")
    for idx in range(max_pairs):
        real_fname = real_files[idx]
        real_path = os.path.join(real_dir, real_fname)
        video_name = os.path.splitext(real_fname)[0]  # 输出目录名，例：'1'、'2'、...

        # 组装此 clip 的各视角视频路径：视角0是真实视频
        view_video_paths = [real_path]
        for gdir, gfiles in zip(gen_dirs, gen_files_list):
            gen_fname = gfiles[idx]  # 按顺序匹配
            gen_path = os.path.join(gdir, gen_fname)
            view_video_paths.append(gen_path)

        # 读入所有视角的视频帧
        view_frames_list = []
        for vp in view_video_paths:
            frames = _read_video_frames(vp, max_frames=max_frames)
            if len(frames) == 0:
                print(f"[WARN] no frames in {vp}, skip this clip index {idx}")
                view_frames_list = None
                break
            view_frames_list.append(frames)

        if view_frames_list is None:
            continue

        # 该 clip 可用的 timestep 数：受所有视角长度和 num_frames 共同限制
        total_timesteps = min(len(frames) for frames in view_frames_list)
        if total_timesteps == 0:
            print(f"[WARN] no usable timesteps for clip index {idx}")
            continue
        # 均匀采样 num_timesteps 个 index
        if (num_timesteps is None) or (num_timesteps >= total_timesteps):
            chosen_indices = list(range(total_timesteps))
        else:
            chosen_indices = np.linspace(
                0, total_timesteps - 1, num_timesteps, dtype=int
            )
            chosen_indices = sorted(set(chosen_indices))
            
        # 输出目录：每个 clip 一个文件夹
        clip_out_dir = os.path.join(output_root, video_name)
        os.makedirs(clip_out_dir, exist_ok=True)

        num_views = len(view_frames_list)
        print(
            f"[INFO] Building frame-dirs for pair index {idx}: "
            f"real={real_fname}, views={num_views}, timesteps={len(chosen_indices)} (total {total_timesteps})"
        )

        # 对每个 timestep t，构造一个“视角帧目录”
        for t_idx in chosen_indices:
            pseudo_dir = os.path.join(clip_out_dir, f"t{t_idx:04d}")
            os.makedirs(pseudo_dir, exist_ok=True)

            for v_idx in range(num_views):
                frame = view_frames_list[v_idx][t_idx]
                frame_name = os.path.join(pseudo_dir, f"frame_{v_idx:04d}.jpg")
                cv2.imwrite(frame_name, frame)

        print(f"[INFO] Done: clip index {idx}, output to {clip_out_dir}")


# if __name__ == "__main__":
#     # ========== 示例：构造 FVD-V 视角伪视频（帧目录） ==========
#     real_dir = "/data1/home/liu_kai/ReCamMaster/example_test_data/videos"  # 真实视频目录 传视频目录.mp4
#     gen_dirs = [
#         "/data1/home/liu_kai/ReCamMaster/recam_result/cam_type1",
#         "/data1/home/liu_kai/ReCamMaster/recam_result/cam_type3",
#         "/data1/home/liu_kai/ReCamMaster/recam_result/cam_type5",
#         "/data1/home/liu_kai/ReCamMaster/recam_result/cam_type7",
#         "/data1/home/liu_kai/ReCamMaster/recam_result/cam_type9"
#         # 可以继续加更多视角
#     ]
#     fvdv_output_root = "/data1/home/liu_kai/RecamMaster_plucker/evaluate/recam_fvd_v_view_frames_gen_5"

#     build_view_pseudo_videos_as_frame_dirs(
#         real_dir=real_dir,
#         gen_dirs=gen_dirs,
#         output_root=fvdv_output_root,
#         max_frames=81,
#         num_timesteps=21,
#     )

if __name__ == "__main__":
    # video_dir = "/data1/home/liu_kai/ReCamMaster/recam_result/cam_type3"
    # output_root = "/data1/home/liu_kai/RecamMaster_plucker/evaluate/cam03_gen_videos_recam"

    # extract_frames_from_dir(
    #     video_dir=video_dir,
    #     output_root=output_root,
    #     num_frames=81,
    #     # exts 可以不传，使用默认 (mp4, avi, mov, mkv, webm)
    # )
    cam_types = ["1", "3", "5", "7", "9"]

    for cam_type in cam_types:
        video_dir = f"/data1/home/liu_kai/ReCamMaster/ckpt_20000_plucker/cam_type{cam_type}"
        output_root = f"/data1/home/liu_kai/RecamMaster_plucker/evaluate/cam0{cam_type}_gen_videos_plucker"
        extract_frames_from_dir(
            video_dir=video_dir,
            output_root=output_root,
            num_frames=81,
            # exts 可以不传，使用默认 (mp4, avi, mov, mkv, webm)
        )

