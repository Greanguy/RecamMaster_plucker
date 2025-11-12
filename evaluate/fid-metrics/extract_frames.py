import cv2
import os

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

# 提取原视频和生成视频的前 81 帧
extract_frames('1.mp4', 'real_video_frames', num_frames=81)
# extract_frames('video0.mp4', 'generated_video_frames', num_frames=81)
