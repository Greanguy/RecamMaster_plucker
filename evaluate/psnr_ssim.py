import os
import argparse
from typing import Tuple, List
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img

def ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """保证两张图大小一致，不一致时将第二张 resize 到第一张大小。"""
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    return img1, img2

def calc_psnr_ssim(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
    """
    计算两张图的 PSNR 和 SSIM
    输入为 BGR uint8 内部转成 float32 并按 0-255 的 data_range
    """
    img1, img2 = ensure_same_size(img1, img2) # 保证 size 一致
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32) # 转为 float32
    psnr = peak_signal_noise_ratio(img1_f, img2_f, data_range=255)# skimage 默认当 data_range=差值 max-min (这里直接用 255)
    # SSIM：注意 channel_axis=-1 表示最后一维是通道 (H, W, C)
    ssim = structural_similarity(
        img1_f,
        img2_f,
        data_range=255,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return psnr, ssim

def calc_for_two_images(ref_path: str, dst_path: str):
    img_ref = read_image(ref_path)
    img_dst = read_image(dst_path)
    psnr, ssim = calc_psnr_ssim(img_ref, img_dst)
    print(f"Reference: {ref_path}")
    print(f"Distorted: {dst_path}")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.6f}")

def is_image_file(name: str) -> bool:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    return name.lower().endswith(exts)

def list_image_files(dir_path: str) -> List[str]:
    """列出目录下所有图片文件（文件名排序）"""
    files = [f for f in os.listdir(dir_path) if is_image_file(f)]
    files.sort()
    return files

def calc_for_dirs(ref_dir: str, dst_dir: str, csv_out: str = None):
    """
    对两个目录下“同名图片”逐一计算 PSNR/SSIM 并打印平均值
    规则：取两个目录交集文件名 按名称排序后逐个匹配
    """
    ref_files = set(list_image_files(ref_dir))
    dst_files = set(list_image_files(dst_dir))
    common_files = sorted(ref_files & dst_files)

    if not common_files:
        raise RuntimeError("两个目录下没有同名图片可匹配。")

    results = []
    print(f"在 {len(common_files)} 对图片上计算 PSNR/SSIM：\n")

    for name in common_files:
        ref_path = os.path.join(ref_dir, name)
        dst_path = os.path.join(dst_dir, name)
        img_ref = read_image(ref_path)
        img_dst = read_image(dst_path)
        psnr, ssim = calc_psnr_ssim(img_ref, img_dst)
        results.append((name, psnr, ssim))
        print(f"{name:40s}  PSNR: {psnr:8.4f} dB   SSIM: {ssim:8.6f}")

    # 统计平均
    psnr_mean = float(np.mean([r[1] for r in results]))
    ssim_mean = float(np.mean([r[2] for r in results]))
    print("\n==== 统计结果 ====")
    print(f"平均 PSNR: {psnr_mean:.4f} dB")
    print(f"平均 SSIM: {ssim_mean:.6f}")

    # 输出 CSV
    if csv_out is not None:
        import csv
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "psnr", "ssim"])
            for name, psnr, ssim in results:
                writer.writerow([name, f"{psnr:.6f}", f"{ssim:.6f}"])
        print(f"\n结果已保存到: {csv_out}")


def main():
    parser = argparse.ArgumentParser(description="计算两张图片或两个目录下成对图片的 PSNR/SSIM")
    parser.add_argument("ref", help="参考图/参考目录")
    parser.add_argument("dist", help="失真图/失真目录")
    parser.add_argument("--csv-out",type=str,default=None,help="（可选）目录模式下，将结果保存为 CSV 文件路径",)
    args = parser.parse_args()

    if os.path.isdir(args.ref) and os.path.isdir(args.dist):
        calc_for_dirs(args.ref, args.dist, args.csv_out)
    elif os.path.isfile(args.ref) and os.path.isfile(args.dist):
        calc_for_two_images(args.ref, args.dist)
    else:
        raise RuntimeError("ref 和 dist 要么都是文件，要么都是目录。")

if __name__ == "__main__":
    main()
