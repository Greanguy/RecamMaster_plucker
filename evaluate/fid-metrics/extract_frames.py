import os
import cv2
import numpy as np

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
        video_name = os.path.splitext(fname)[0]

        # 每个视频一个单独输出文件夹
        output_folder = os.path.join(output_root, video_name)
        print(f"Processing {video_path} -> {output_folder}")

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

    - real_dir 下有若干真实视频，如 clip001.mp4, clip002.mp4, ...
    - gen_dirs 是一个目录列表，每个目录里存放对应的生成视频：
        gen_dirs = [
            "/path/to/gen_view1",
            "/path/to/gen_view2",
            ...
        ]
      且每个生成目录中都存在与 real_dir 同名的视频文件。

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
    real_files = sorted(
        [
            f for f in os.listdir(real_dir)
            if f.lower().endswith(exts)
        ]
    )

    print(f"[INFO] Found {len(real_files)} real videos in {real_dir}")

    for fname in real_files:
        real_path = os.path.join(real_dir, fname)
        video_name = os.path.splitext(fname)[0]

        # 对应的生成视频路径（视角）
        view_video_paths = [real_path]  # 视角0：真实视频
        for gdir in gen_dirs:
            gen_path = os.path.join(gdir, fname)
            if not os.path.exists(gen_path):
                print(f"[WARN] generated video not found: {gen_path}, skip {fname}")
                view_video_paths = None
                break
            view_video_paths.append(gen_path)

        if view_video_paths is None:
            continue

        # 读入所有视角的视频帧
        view_frames_list = []
        for vp in view_video_paths:
            frames = _read_video_frames(vp, max_frames=max_frames)
            if len(frames) == 0:
                print(f"[WARN] no frames in {vp}, skip {fname}")
                view_frames_list = None
                break
            view_frames_list.append(frames)

        if view_frames_list is None:
            continue

        # 该 clip 可用的 timestep 数：受所有视角长度和 num_frames 共同限制
        total_timesteps = min(len(frames) for frames in view_frames_list)
        if total_timesteps == 0:
            print(f"[WARN] no usable timesteps for {fname}")
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
            f"[INFO] Building frame-dirs for {fname}: "
            f"{num_views} views, {num_timesteps} timesteps"
        )

        # 对每个 timestep t，构造一个“视角帧目录”
        for t_idx in chosen_indices:
            pseudo_dir = os.path.join(clip_out_dir, f"t{t_idx:04d}")
            os.makedirs(pseudo_dir, exist_ok=True)

            for v_idx in range(num_views):
                frame = view_frames_list[v_idx][t_idx]
                frame_name = os.path.join(pseudo_dir, f"frame_{v_idx:04d}.jpg")
                cv2.imwrite(frame_name, frame)

        print(f"[INFO] Done: {fname}, output to {clip_out_dir}")


if __name__ == "__main__":
    # ========== 示例：构造 FVD-V 视角伪视频（帧目录） ==========
    real_dir = "/data1/home/liu_kai/ReCamMaster/example_test_data/videos"  # 真实视频目录 传视频目录.mp4
    gen_dirs = [
        "/data1/home/liu_kai/ReCamMaster/example_test_data/videos",
        # 可以继续加更多视角
    ]
    fvdv_output_root = "/data1/home/liu_kai/RecamMaster_plucker/evaluate/fvd_v_view_frames_real"

    build_view_pseudo_videos_as_frame_dirs(
        real_dir=real_dir,
        gen_dirs=gen_dirs,
        output_root=fvdv_output_root,
        max_frames=81,
        num_timesteps=21,
    )
