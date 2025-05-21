!pip install scikit-image tabulate

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate
import os
import matplotlib.pyplot as plt

# Hàm tính SSIM giữa 2 frame
def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

# Tính SSIM trung bình giữa 2 video
def compute_ssim(original_video_path, processed_video_path, frame_step=10):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_proc = cv2.VideoCapture(processed_video_path)

    if not cap_orig.isOpened() or not cap_proc.isOpened():
        print(f"Lỗi mở video: {original_video_path} hoặc {processed_video_path}")
        return None

    ssim_values = []
    frame_count = 0

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_proc, frame_proc = cap_proc.read()

        if not ret_orig or not ret_proc:
            break

        frame_count += 1
        if frame_count % frame_step != 0:
            continue

        if frame_orig.shape != frame_proc.shape:
            frame_proc = cv2.resize(frame_proc, (frame_orig.shape[1], frame_orig.shape[0]))

        ssim_val = calculate_ssim(frame_orig, frame_proc)
        ssim_values.append(ssim_val)

    cap_orig.release()
    cap_proc.release()
    return np.mean(ssim_values) if ssim_values else float('nan')

# Cấu hình dữ liệu
base_path = '/content/'
original_video = base_path + 'final_output2(1).mp4'
bitrates = [100, 500, 1000, 2000, 5000]
discarding_rates = [5.75, 11.25]
prefixes = ['output2(run22)', 'output1(run12)']

# Lưu kết quả
results_ssim = []

for i, dr in enumerate(discarding_rates):
    for bitrate in bitrates:
        video_name = f"{prefixes[i]}_{bitrate}k.mkv"
        video_path = base_path + video_name
        if os.path.exists(video_path) and os.path.exists(original_video):
            avg_ssim = compute_ssim(original_video, video_path)
            results_ssim.append([video_name, bitrate, dr, avg_ssim])
        else:
            print(f"Không tìm thấy: {video_name}")

# In bảng SSIM
headers = ['Tên Video', 'Bitrate (kbps)', 'Discarding Rate (%)', 'SSIM trung bình']
print(tabulate(results_ssim, headers=headers, tablefmt='grid', floatfmt=".4f"))

# Vẽ biểu đồ SSIM
ssim_dr1 = [row[3] for row in results_ssim if row[2] == 5.75]
ssim_dr2 = [row[3] for row in results_ssim if row[2] == 11.25]

plt.figure(figsize=(8,5))
plt.plot(bitrates, ssim_dr1, label='Discard Rate 5.75%', marker='o')
plt.plot(bitrates, ssim_dr2, label='Discard Rate 11.25%', marker='o')
plt.xlabel('Bitrate (kbps)')
plt.ylabel('SSIM trung bình')
plt.title('SSIM theo Bitrate và Discarding Rate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
