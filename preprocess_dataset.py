import argparse, os, subprocess, time
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

NUM_FRAMES, IMG, SR, CLIP_S, N_MELS, MEL_SIZE = 16, 224, 16000, 10.0, 128, 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FACE_MODEL = "trpakov/vit-face-expression"
SER_MODEL = "superb/wav2vec2-base-superb-er"
TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
ASR_MODEL = "openai/whisper-tiny"
EMB = 768
LAST_INFO = {}  # what the last call found (shown in the app)


# ---------------- video ----------------
def _sample_frames_uint8(path, n=NUM_FRAMES, size=IMG, clip_s=CLIP_S):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 25.0
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = int(round(clip_s * fps))
    last = (min(cnt, window) if cnt > 0 else window) - 1
    idxs = np.linspace(0, max(last, 0), n).round().astype(int)
    wanted, got = set(idxs.tolist()), {}
    for i in range(last + 1):
        if not cap.grab():
            break
        if i in wanted:
            ok, fr = cap.retrieve()
            if ok:
                got[i] = fr
    cap.release()
    if not got:
        raise IOError(f"no frames decoded: {path}")
    keys = np.array(sorted(got))
    out = []
    for i in idxs:
        fr = cv2.cvtColor(got[keys[np.abs(keys - i).argmin()]], cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(out)


def extract_video_features(path):
    """-> float32 [16,224,224,3], RGB, 0..1"""
    return _sample_frames_uint8(path).astype(np.float32) / 255.0


# ---------------- audio ----------------
_wav = {"path": None, "y": None, "has": False}


def _load_wav(path):
    if _wav["path"] == path:
        return _wav["y"], _wav["has"]
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(CLIP_S), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    y = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.float32).copy()
    _wav.update(path=path, y=y, has=y.size > 0)
    return y, y.size > 0


def extract_audio_features(path):
    """-> float32 [224,224] standardized log-Mel (zeros if the video has no audio track)."""
    import librosa
    y, has = _load_wav(path)
    if not has:
        return np.zeros((MEL_SIZE, MEL_SIZE), np.float32)
    if len(y) < 1024:
        y = np.pad(y, (0, 1024 - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=320, n_mels=N_MELS, fmax=SR / 2)
    db = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0).astype(np.float32)
    db = cv2.resize(db, (MEL_SIZE, MEL_SIZE), interpolation=cv2.INTER_AREA)
    return ((db - db.mean()) / (db.std() + 1e-5)).astype(np.float32)


# ---------------- emotion / text (pretrained, frozen) ----------------
_M = {}


def _get(name):
    if name in _M:
        return _M[name]
    from transformers import (AutoFeatureExtractor, AutoImageProcessor, AutoModelForAudioClassification,
                              AutoModelForImageClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline)
    if name == "haar":
        _M[name] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    elif name == "face":
        _M[name] = (AutoImageProcessor.from_pretrained(FACE_MODEL),
                    AutoModelForImageClassification.from_pretrained(FACE_MODEL).to(DEVICE).eval())
    elif name == "speech":
        _M[name] = (AutoFeatureExtractor.from_pretrained(SER_MODEL),
                    AutoModelForAudioClassification.from_pretrained(SER_MODEL).to(DEVICE).eval())
    elif name == "text":
        _M[name] = (AutoTokenizer.from_pretrained(TEXT_MODEL),
                    AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL).to(DEVICE).eval())
    elif name == "asr":
        _M[name] = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=0 if DEVICE == "cuda" else -1)
    return _M[name]


@torch.no_grad()
def _face_emotion(frames):
    haar, (proc, m) = _get("haar"), _get("face")
    crops = []
    for fr in frames[::2]:
        faces = haar.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            crops.append(fr[y:y + h, x:x + w])
    if not crops:
        return np.zeros(EMB, np.float32), 0, None
    inp = proc(images=crops, return_tensors="pt").to(DEVICE)
    out = m(**inp, output_hidden_states=True)
    label = m.config.id2label[int(torch.softmax(out.logits, -1).mean(0).argmax())]
    return out.hidden_states[-1][:, 0].mean(0).float().cpu().numpy(), len(crops), label


@torch.no_grad()
def _speech_emotion(y):
    if len(y) < SR // 4:
        return np.zeros(EMB, np.float32), None
    fe, m = _get("speech")
    inp = {k: v.to(DEVICE) for k, v in fe(y, sampling_rate=SR, return_tensors="pt").items()}
    out = m(**inp, output_hidden_states=True)
    return out.hidden_states[-1].mean(1)[0].float().cpu().numpy(), m.config.id2label[int(out.logits.argmax())]


def _transcribe(y):
    if len(y) < SR // 2 or float(np.sqrt((y ** 2).mean())) < 1e-3:  # silence: Whisper tends to invent text
        return ""
    r = _get("asr")({"raw": y, "sampling_rate": SR}, chunk_length_s=30,
                    generate_kwargs={"language": "english", "task": "transcribe"})
    return r["text"].strip()


@torch.no_grad()
def _text_emotion(text):
    if not text:
        return np.zeros(EMB, np.float32), None
    tok, m = _get("text")
    inp = tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    out = m(**inp, output_hidden_states=True)
    return out.hidden_states[-1][0, 0].float().cpu().numpy(), m.config.id2label[int(out.logits.argmax())]


def engineer_emotion_features(path, frames=None):
    """-> (image_emotion[768], audio_emotion[768], text_emotion[768]) float32. Details in LAST_INFO."""
    if frames is None:
        frames = _sample_frames_uint8(path)
    elif frames.dtype != np.uint8:
        frames = np.uint8(np.clip(frames, 0, 1) * 255)
    y, has = _load_wav(path)
    ie, n_faces, face_label = _face_emotion(frames)
    ae, ser_label = _speech_emotion(y) if has else (np.zeros(EMB, np.float32), None)
    text = _transcribe(y) if has else ""
    te, text_label = _text_emotion(text)
    LAST_INFO.update(n_faces=n_faces, face_label=face_label, speech_label=ser_label, transcript=text,
                     text_label=text_label, has_audio=bool(has))
    return ie.astype(np.float32), ae.astype(np.float32), te.astype(np.float32)


# ---------------- batch CLI ----------------
def _save(path, **kw):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **kw)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="splits")
    ap.add_argument("--raw_root", required=True, help="folder containing the 4 category folders")
    ap.add_argument("--out", default="processed_real")
    ap.add_argument("--stage", choices=["va", "emo", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="process only N videos (use to time a trial run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = pd.concat([pd.read_csv(f"{a.splits_dir}/{s}.csv") for s in ("train", "val", "test")]).drop_duplicates("video_id")
    fails, n, t0 = [], 0, time.time()
    for r in tqdm(list(rows.itertuples()), desc="preprocessing"):
        fva, femo = f"{a.out}/{r.video_id}.va.npz", f"{a.out}/{r.video_id}.emo.npz"
        need_va = a.stage in ("va", "all") and not os.path.exists(fva)
        need_emo = a.stage in ("emo", "all") and not os.path.exists(femo)
        if not (need_va or need_emo):
            continue
        if a.limit and n >= a.limit:
            break
        n += 1
        path = os.path.join(a.raw_root, r.rel_path)
        try:
            frames = _sample_frames_uint8(path)
            if need_va:
                _, has = _load_wav(path)
                _save(fva, v=frames, a=extract_audio_features(path).astype(np.float16), has_audio=np.uint8(has))
            if need_emo:
                ie, ae, te = engineer_emotion_features(path, frames)
                _save(femo, ie=ie, ae=ae, te=te, n_faces=LAST_INFO["n_faces"], transcript=np.array(LAST_INFO["transcript"]))
        except Exception as e:
            fails.append((r.video_id, repr(e)[:200]))
    print(f"processed {n} videos in {time.time() - t0:.0f}s | failures: {len(fails)}")
    if fails:
        pd.DataFrame(fails, columns=["video_id", "error"]).to_csv(f"{a.out}/failures.csv", index=False)
        print("see", f"{a.out}/failures.csv")
    rep = [(r.label, int(np.load(f"{a.out}/{r.video_id}.va.npz")["has_audio"])) for r in rows.itertuples()
            if os.path.exists(f"{a.out}/{r.video_id}.va.npz")]
    if rep:
        print("\nlabel (0=Real,1=Fake) vs has_audio (shortcut check):")
        print(pd.crosstab(pd.Series([x[0] for x in rep], name="label"), pd.Series([x[1] for x in rep], name="has_audio")))


if __name__ == "__main__":
    main()
