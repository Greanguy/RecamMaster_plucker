import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames_advanced(video_path, output_dir, frame_limit=80, quality=95):
    """
    高级版本：提取视频帧
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        frame_limit: 最大帧数
        quality: jpg质量 (0-100)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
    
    frame_count = 0
    for frame_count in tqdm(range(min(frame_limit, total_frames)), desc=f"处理 {video_path.name}"):
        ret, frame = cap.read()
        if not ret:
            break
            
        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    cap.release()
    return frame_count

def main():
    parser = argparse.ArgumentParser(description='将MP4视频转换为帧图片')
    parser.add_argument('--input', '-i', default='.', help='输入目录')
    parser.add_argument('--output', '-o', default='frames_output', help='输出目录')
    parser.add_argument('--frames', '-f', type=int, default=80, help='每视频提取帧数')
    parser.add_argument('--quality', '-q', type=int, default=95, help='JPEG质量 (0-100)')
    
    args = parser.parse_args()
    
    # 处理所有视频
    video_files = list(Path(args.input).glob("*.mp4")) + list(Path(args.input).glob("*.MP4"))
    
    for video_path in video_files:
        output_dir = Path(args.output) / video_path.stem
        frames_extracted = extract_frames_advanced(video_path, output_dir, args.frames, args.quality)
        print(f"从 {video_path.name} 提取了 {frames_extracted} 帧到 {output_dir}")

if __name__ == "__main__":
    main()