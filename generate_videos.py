import os
import cv2
import numpy as np

# 4 classes folders[span_3](start_span)[span_3](end_span)
folders = [
    "real_dataset_folder/FakeVideo-FakeAudio",
    "real_dataset_folder/FakeVideo-RealAudio",
    "real_dataset_folder/RealVideo-FakeAudio",
    "real_dataset_folder/RealVideo-RealAudio"
]

print("Generating sample mp4 videos...")

# OpenCV tho direct ga video create chestunnam, moviepy audio tho sambandham lekunda
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    video_path = os.path.join(folder, "sample_video.mp4")
    
    # 224x224 video writer setup[span_4](start_span)[span_4](end_span)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 16.0, (224, 224))
    
    for _ in range(16):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
        
    out.release()

print("Done! Videos created successfully.")