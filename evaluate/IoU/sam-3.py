import os
import numpy as np
import cv2
from typing import Optional, Dict, Any
from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import torch
import cv2
import numpy as np

def read_video_opencv(video_path: str, max_frames: int | None = None):
    """
    用 OpenCV 读取整个视频 返回 List[np.ndarray]
    每帧为 HxWx3, RGB, dtype=uint8.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV 默认是 BGR，需要转成 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"视频 {video_path} 没有读到任何帧")
    return frames

def extract_sam3_masks_for_single_object(
    video_path: str,
    text_prompt: str,
    out_dir: str,
    max_frames: Optional[int] = None,
    center_prior: bool = True,
):
    """
    用 SAM3 对整个视频做概念分割 + 跟踪, 并导出“一个目标对象”的逐帧 mask

    video_path : 输入视频.mp4或 JPG 帧目录(Transformers 支持两种)
    text_prompt: 文本概念, 例如 "car", "white car in front", "truck" 等
    out_dir    : 输出 mask PNG 目录（自动创建）
    max_frames : 若不为 None, 则只处理前 max_frames 帧
    center_prior: 在第 0 帧选目标对象时，是否考虑“靠近中心”的先验
    """
    
    os.makedirs(out_dir, exist_ok=True)
    device = Accelerator().device
    
    local_model_dir = "/data1/home/liu_kai/IoU/facebook/sam3"
    model = Sam3VideoModel.from_pretrained(local_model_dir,local_files_only=True,).to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained(local_model_dir,local_files_only=True,)

    video_frames = read_video_opencv(video_path, max_frames=max_frames)
    num_frames = len(video_frames)
    print(f"[INFO] Loaded {num_frames} frames from {video_path}")
    
    # 初始化 video session
    session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # 加入文本提示（检测 & 跟踪所有满足该概念的实例）
    session = processor.add_text_prompt(
        inference_session=session,
        text=text_prompt,
    )

    outputs_per_frame: Dict[int, Dict[str, Any]] = {}
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=session,
        max_frame_num_to_track=len(video_frames) - 1,
    ):
        processed_outputs = processor.postprocess_outputs(session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed_outputs

    # ---------- 在第 0 帧上选择“目标 track” ----------
    frame0 = outputs_per_frame.get(0, None)
    if frame0 is None or len(frame0["object_ids"]) == 0:
        print("[WARN] 第 0 帧没有检测到任何对象，整个视频都输出空 mask")
        H, W = video_frames[0].shape[:2]
        empty = np.zeros((H, W), dtype=np.uint8)
        for i in range(len(video_frames)):
            cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), empty)
        return

    masks0 = frame0["masks"]      # [N, H, W]
    obj_ids0 = frame0["object_ids"]  # [N]
    scores0 = frame0["scores"]    # [N]

    # 转 numpy
    if isinstance(masks0, torch.Tensor):
        masks0_np = masks0.cpu().numpy()
    else:
        masks0_np = np.asarray(masks0)
    if isinstance(scores0, torch.Tensor):
        scores0_np = scores0.cpu().numpy()
    else:
        scores0_np = np.asarray(scores0)
    if isinstance(obj_ids0, torch.Tensor):
        obj_ids0_np = obj_ids0.cpu().numpy()
    else:
        obj_ids0_np = np.asarray(obj_ids0)

    H, W = masks0_np.shape[-2:]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    img_center = np.array([H / 2.0, W / 2.0])

    # 为每个候选实例计算一个“打分”：score * area / (1 + center_dist)
    areas = masks0_np.reshape(masks0_np.shape[0], -1).sum(axis=1)  # 每个实例像素数
    centers = []
    for k in range(masks0_np.shape[0]):
        m = masks0_np[k] > 0.5
        if m.sum() == 0:
            centers.append(np.array([1e9, 1e9]))
            continue
        cy = (yy[m].mean())
        cx = (xx[m].mean())
        centers.append(np.array([cy, cx]))
    centers = np.stack(centers, axis=0)
    dists = np.linalg.norm(centers - img_center[None, :], axis=1)

    if center_prior:
        # 越靠近中心，惩罚越小
        score_obj = scores0_np * (areas / (1.0 + dists))
    else:
        score_obj = scores0_np * areas

    best_idx = int(score_obj.argmax())
    target_id = obj_ids0_np[best_idx]
    print(f"[INFO] 选择 object_id={target_id} 作为评测目标对象")

    # ---------- 按 frame 导出该 target_id 的 mask ----------
    for frame_idx in range(len(video_frames)):
        out_path = os.path.join(out_dir, f"{frame_idx:04d}.png")

        if frame_idx not in outputs_per_frame:
            # 没有结果，就输出全黑 mask
            Hf, Wf = video_frames[frame_idx].shape[:2]
            empty = np.zeros((Hf, Wf), dtype=np.uint8)
            cv2.imwrite(out_path, empty)
            continue

        pf = outputs_per_frame[frame_idx]
        masks = pf["masks"]
        obj_ids = pf["object_ids"]

        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        if isinstance(obj_ids, torch.Tensor):
            obj_ids_np = obj_ids.cpu().numpy()
        else:
            obj_ids_np = np.asarray(obj_ids)

        if masks_np.shape[0] == 0:
            # 没有任何实例
            Hf, Wf = video_frames[frame_idx].shape[:2]
            empty = np.zeros((Hf, Wf), dtype=np.uint8)
            cv2.imwrite(out_path, empty)
            continue

        # 找到当前帧中 object_id == target_id 的实例
        idxs = np.where(obj_ids_np == target_id)[0]
        if len(idxs) == 0:
            # 目标暂时被遮挡 / 丢失：给空 mask
            Hf, Wf = video_frames[frame_idx].shape[:2]
            empty = np.zeros((Hf, Wf), dtype=np.uint8)
            cv2.imwrite(out_path, empty)
            continue

        k = int(idxs[0])
        m = masks_np[k] > 0.5
        # m = (masks_np > 0.5).any(axis=0)  # [H, W] 如果不止检测一个目标对象, 所有实例的 union
        mask_u8 = (m.astype(np.uint8) * 255)
        cv2.imwrite(out_path, mask_u8)

    print(f"[DONE] 已将 {len(video_frames)} 帧的目标前景 mask 保存到: {out_dir}")


if __name__ == "__main__":
    # 真值视频
    extract_sam3_masks_for_single_object(
        video_path="/data1/home/liu_kai/ReCamMaster/example_test_data/videos/1.mp4",
        text_prompt="woman",
        out_dir="masks_gt",
        max_frames=50,
    )

    # 生成视频
    extract_sam3_masks_for_single_object(
        video_path="/data1/home/liu_kai/ReCamMaster/recam_result/cam_type9/1.mp4",
        text_prompt="woman",
        out_dir="masks_gen",
        max_frames=50,
    )

    from mask_iou import compute_video_iou
    video_iou, frame_ious = compute_video_iou("masks_gen", "masks_gt")
    print("Video Mean IoU:", video_iou)
