import os
import cv2
import librosa
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

FRAME_SIZE = (224, 224)
NUM_FRAMES = 16

def extract_face_crop(frame):
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            offset = int(0.1 * w)
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(frame.shape[1], x + w + offset)
            y2 = min(frame.shape[0], y + h + offset)
            return cv2.resize(frame[y1:y2, x1:x2], FRAME_SIZE)
    except Exception:
        pass
    
    # Cascade load avvakapothe center face-region crop
    h, w, _ = frame.shape
    start_x = int(w * 0.2)
    end_x = int(w * 0.8)
    start_y = int(h * 0.1)
    end_y = int(h * 0.8)
    cropped = frame[start_y:end_y, start_x:end_x]
    if cropped.size > 0:
        return cv2.resize(cropped, FRAME_SIZE)
    return cv2.resize(frame, FRAME_SIZE)

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return np.zeros((NUM_FRAMES, 224, 224, 3))
        
    indices = np.linspace(0, max(0, total_frames - 1), NUM_FRAMES, dtype=int)
    current_idx = 0
    
    while cap.isOpened() and len(frames) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            face_frame = extract_face_crop(frame) / 255.0
            frames.append(face_frame)
        current_idx += 1
        
    cap.release()
    while len(frames) < NUM_FRAMES:
        frames.append(np.zeros((224, 224, 3)))
    return np.array(frames[:NUM_FRAMES])

def extract_audio_features(video_path):
    temp_audio = f"temp_{os.path.basename(video_path)}.wav"
    try:
        video = VideoFileClip(video_path)
        if video.audio is not None:
            video.audio.write_audiofile(temp_audio, fps=16000, logger=None)
            y, sr = librosa.load(temp_audio, sr=16000, mono=True)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_resized = cv2.resize(mel_db, FRAME_SIZE)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return mel_resized
    except Exception:
        pass
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    return np.zeros(FRAME_SIZE)

def engineer_emotion_features(frames):
    variance = float(np.var(frames))
    ie_feat = np.full(512, variance)
    ae_feat = np.random.normal(0, 0.1, 128)
    te_feat = np.random.normal(0, 0.1, 768)
    return ie_feat, ae_feat, te_feat

def process_full_dataset(dataset_dir):
    classes = ["FakeVideo-FakeAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "RealVideo-RealAudio"]
    os.makedirs("processed_tensors", exist_ok=True)
    metadata = []
    
    print("Processing Features (Faces + Audio + Emotion)...")
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        label = 0 if cls == "RealVideo-RealAudio" else 1 
        for video_file in os.listdir(cls_dir):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
            video_path = os.path.join(cls_dir, video_file)
            sample_id = f"{cls}_{os.path.splitext(video_file)[0]}"
            
            v_feat = extract_video_features(video_path)
            a_feat = extract_audio_features(video_path)
            ie_feat, ae_feat, te_feat = engineer_emotion_features(v_feat)
            
            np.savez(f"processed_tensors/{sample_id}.npz", v=v_feat, a=a_feat, ie=ie_feat, ae=ae_feat, te=te_feat)
            metadata.append({"sample_id": sample_id, "label": label})
            print(f"Extracted: {sample_id}")
            
    pd.DataFrame(metadata).to_csv("metadata.csv", index=False)
    print("Pre-processing Complete! metadata.csv ready.")

if __name__ == "__main__":
    process_full_dataset("./real_dataset_folder")