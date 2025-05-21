import cv2
import os
from pydub import AudioSegment
import random

# --- Thông số ---
INPUT_VIDEO = "data/extracted/video_only2.avi"
INPUT_AUDIO = "data/extracted/audio_only2.flac"
OUTPUT_FOLDER = "data/processed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Thông số Gilbert-Elliott model ---
p = 0.1  # Xác suất chuyển G -> B
r = 0.3  # Xác suất chuyển B -> G

# Danh sách các cặp (loss_prob_good, loss_prob_bad) để chạy 5 lần (đổi tên loss4)
loss_params_list = [
    (0.01, 0.2),
    (0.05, 0.3),
    (0.1, 0.4),
    (0.15, 0.5),
    (0.2, 0.6),
]

def process_one_run(run_id, loss_prob_good, loss_prob_bad):
    # Đọc video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Đường dẫn file đầu ra theo run_id
    out_video_path = os.path.join(OUTPUT_FOLDER, f"drop_video_loss4_run{run_id+1}2.avi")
    out_audio_path = os.path.join(OUTPUT_FOLDER, f"drop_audio_loss4_run{run_id+1}2.flac")
    final_output_path = os.path.join(OUTPUT_FOLDER, f"final_output_loss4_run{run_id+1}2.mkv")
    log_file_path = os.path.join(OUTPUT_FOLDER, f"frame_status_log_loss4_run{run_id+1}2.txt")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    state = "Good"
    frame_status = []

    # Log file
    with open(log_file_path, "w") as f:
        f.write("Frame\tState\tAction\n")
        for i in range(n_frames):
            # Chuyển trạng thái
            if state == "Good" and random.random() < p:
                state = "Bad"
            elif state == "Bad" and random.random() < r:
                state = "Good"

            loss_prob = loss_prob_good if state == "Good" else loss_prob_bad
            keep = random.random() > loss_prob
            frame_status.append((keep, state))

            action = "KEEP" if keep else "DROP"
            f.write(f"{i}\t{state}\t{action}\n")

    # Ghi video giữ frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for keep, _ in frame_status:
        ret, frame = cap.read()
        if not ret:
            break
        if keep:
            out.write(frame)

    cap.release()
    out.release()
    print(f"Run {run_id+1}: ✅ Video xử lý xong: giữ {sum(1 for k, _ in frame_status if k)} / {len(frame_status)} frame.")

    # Cắt audio tương ứng
    audio = AudioSegment.from_file(INPUT_AUDIO, format="flac")
    frame_duration = 1000 / fps  # ms

    new_audio = AudioSegment.empty()
    for i, (keep, _) in enumerate(frame_status):
        if keep:
            start = int(i * frame_duration)
            end = int((i + 1) * frame_duration)
            new_audio += audio[start:end]

    new_audio.export(out_audio_path, format="flac")
    print(f"Run {run_id+1}: ✅ Âm thanh đã cắt khớp với video.")

    # Ghép video + audio, mã hóa lại video với MJPEG
    cmd = f'ffmpeg -y -i "{out_video_path}" -i "{out_audio_path}" -c:v mjpeg -q:v 5 -c:a flac "{final_output_path}"'
    os.system(cmd)

    print(f"Run {run_id+1}: 🎉 Hoàn tất! Video đầu ra: {final_output_path}")
    print(f"Run {run_id+1}: 📄 Log trạng thái: {log_file_path}\n")

# --- Chạy 5 lần với các tỉ lệ mất khác nhau ---
for run_id, (loss_good, loss_bad) in enumerate(loss_params_list):
    print(f"=== Bắt đầu chạy lần {run_id+1} với loss_prob_good={loss_good}, loss_prob_bad={loss_bad} ===")
    process_one_run(run_id, loss_good, loss_bad)
