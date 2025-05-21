# ⚙️ CÀI ĐẶT THƯ VIỆN (nếu cần)
!pip install tabulate

import cv2
import numpy as np
from tabulate import tabulate
import os
import matplotlib.pyplot as plt

# Hàm tính PSNR cho 1 cặp frame
def calculate_psnr(frame1, frame2):
    mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Tính PSNR trung bình cho 2 video
def compute_psnr(original_video_path, processed_video_path, frame_step=10):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_proc = cv2.VideoCapture(processed_video_path)
    
    if not cap_orig.isOpened() or not cap_proc.isOpened():
        print(f"Lỗi mở video: {original_video_path} hoặc {processed_video_path}")
        return None
    
    psnr_values = []
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

        psnr = calculate_psnr(frame_orig, frame_proc)
        if psnr != float('inf'):
            psnr_values.append(psnr)

    cap_orig.release()
    cap_proc.release()
    return np.mean(psnr_values) if psnr_values else float('nan')

# Cấu hình dữ liệu
base_path = '/content/'
original_video = base_path + 'final_output2(1).mp4'
bitrates = [100, 500, 1000, 2000, 5000]
discarding_rates = [5.75, 11.25]
prefixes = ['output2(run22)', 'output1(run12)']  # 5.75% và 11.25%

# Lưu kết quả
results_psnr = []

for i, dr in enumerate(discarding_rates):
    for bitrate in bitrates:
        video_name = f"{prefixes[i]}_{bitrate}k.mkv"
        video_path = base_path + video_name
        if os.path.exists(video_path) and os.path.exists(original_video):
            avg_psnr = compute_psnr(original_video, video_path)
            results_psnr.append([video_name, bitrate, dr, avg_psnr])
        else:
            print(f"Không tìm thấy: {video_name}")

# In bảng PSNR
headers = ['Tên Video', 'Bitrate (kbps)', 'Discarding Rate (%)', 'PSNR trung bình (dB)']
print(tabulate(results_psnr, headers=headers, tablefmt='grid', floatfmt=".4f"))

# Vẽ biểu đồ PSNR
psnr_dr1 = [row[3] for row in results_psnr if row[2] == 5.75]
psnr_dr2 = [row[3] for row in results_psnr if row[2] == 11.25]

plt.figure(figsize=(8,5))
plt.plot(bitrates, psnr_dr1, label='Discard Rate 5.75%', marker='o')
plt.plot(bitrates, psnr_dr2, label='Discard Rate 11.25%', marker='o')
plt.xlabel('Bitrate (kbps)')
plt.ylabel('PSNR trung bình (dB)')
plt.title('PSNR theo Bitrate và Discarding Rate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
