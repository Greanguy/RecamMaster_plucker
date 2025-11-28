import numpy as np

# 计算两个二值mask的IoU
def mask_iou(mask_pred: np.ndarray, mask_gt: np.ndarray, eps: float = 1e-7) -> float:
    """
    mask_pred, mask_gt: HxW, dtype=bool 或 {0,1}/{0,255}
    """
    # 先转成 bool
    mask_pred = mask_pred.astype(bool)
    mask_gt   = mask_gt.astype(bool)

    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        # 两边都没有前景，按场景需求处理：
        # 1) 返回1.0（完全“对齐”）；2) 返回np.nan并在外面跳过；3) 返回0.0
        return 1.0
    return inter / (union + eps)

import os
import cv2

# 计算一段视频的平均mask IoU
def compute_video_iou(gen_mask_dir: str, gt_mask_dir: str):
    """
    gen_mask_dir: 生成视频的前景mask帧目录
    gt_mask_dir:  真值视频的前景mask帧目录
    """
    gen_files = sorted(os.listdir(gen_mask_dir))
    gt_files  = sorted(os.listdir(gt_mask_dir))

    assert len(gen_files) == len(gt_files), "两侧帧数不一致"

    ious = []
    for gf, gtf in zip(gen_files, gt_files):
        gen_mask_path = os.path.join(gen_mask_dir, gf)
        gt_mask_path  = os.path.join(gt_mask_dir,  gtf)

        gen_mask = cv2.imread(gen_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask  = cv2.imread(gt_mask_path,  cv2.IMREAD_GRAYSCALE)

        # 转为 0/1
        _, gen_bin = cv2.threshold(gen_mask, 127, 1, cv2.THRESH_BINARY)
        _, gt_bin  = cv2.threshold(gt_mask,  127, 1, cv2.THRESH_BINARY)

        iou = mask_iou(gen_bin, gt_bin)
        ious.append(iou)

    # 这里可以过滤掉为 NaN 的帧（如果上面 union=0 时返回的是 np.nan）
    ious = np.array(ious, dtype=np.float32)
    valid_ious = ious[~np.isnan(ious)]
    video_mean_iou = float(valid_ious.mean()) if len(valid_ious) > 0 else float("nan")
    return video_mean_iou, ious

# 示例用法
if __name__ == "__main__":
    gen_mask_dir = "path/to/generated/masks"
    gt_mask_dir  = "path/to/ground_truth/masks"

    video_iou, frame_ious = compute_video_iou(gen_mask_dir, gt_mask_dir)
    print(f"Video Mean IoU: {video_iou}")
    print(f"Frame-wise IoUs: {frame_ious}")