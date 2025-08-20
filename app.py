from __future__ import annotations
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Video Emotion Percentage Analysis", page_icon="📊", layout="wide")


# ──────────────────────────────────────────────────────────────────────────────
# Constants & Session State
# ──────────────────────────────────────────────────────────────────────────────
EMOTION_OPTIONS: List[str] = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_TRANSLATIONS = {
    "angry": "Злость",
    "disgust": "Отвращение",
    "fear": "Страх",
    "happy": "Счастье",
    "sad": "Грусть",
    "surprise": "Удивление",
    "neutral": "Нейтрально",
}

# Simplified session state keys
STATE = {
    "video_path": "video_path",
    "results_df": "results_df",
}

# Initialize session state with defaults
for key, default in [
    (STATE["video_path"], None),
    (STATE["results_df"], None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    """A container for all the models and their configurations."""
    yolo: YOLO
    processor: AutoImageProcessor
    expr_model: AutoModelForImageClassification
    device: torch.device
    amp_dtype: Optional[torch.dtype]


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────
def get_device(use_gpu: bool = True) -> torch.device:
    """Gets the appropriate torch device based on GPU availability and user preference."""
    return torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")


@st.cache_resource(show_spinner="Загрузка моделей...")
def load_models(use_gpu: bool = True, use_fp16: bool = True) -> Optional[ModelBundle]:
    """
    Loads the YOLO face detector and ViT expression classifier.
    This function is cached to prevent reloading models on every script run.
    """
    device = get_device(use_gpu)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    try:
        yolo = YOLO("models/yolov8n-face.pt")
        yolo.to("cuda" if device.type == "cuda" else "cpu")
        yolo.fuse()

        # model_name = "trpakov/vit-face-expression"
        model_name = "mo-thecreator/vit-Facial-Expression-Recognition"
        processor = AutoImageProcessor.from_pretrained(model_name)
        expr_model = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()

        amp_dtype = torch.float16 if (device.type == "cuda" and use_fp16) else None
        return ModelBundle(yolo=yolo, processor=processor, expr_model=expr_model, device=device, amp_dtype=amp_dtype)

    except Exception as e:
        st.error(f"Не удалось загрузить модели: {e}. Убедитесь, что файл 'models/yolov8n-face.pt' существует.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Video & Frame Processing Utilities
# ──────────────────────────────────────────────────────────────────────────────
def get_video_info(video_path: str) -> Tuple[float, int, float]:
    """Returns (fps, frame_count, duration_sec) for a video file."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = (frame_count / fps) if fps > 0 else 0.0
        return fps, frame_count, duration
    finally:
        cap.release()


def iterate_frames(video_path: str, fps: float, duration: float, step_sec: float) -> Generator[Tuple[float, np.ndarray], None, None]:
    """Yields (timestamp_sec, frame_rgb) at specified intervals."""
    cap = cv2.VideoCapture(video_path)
    try:
        current_time = 0.0
        while current_time < duration:
            frame_idx = int(round(current_time * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield current_time, frame_rgb
            current_time += step_sec
    finally:
        cap.release()


def _crop_face_from_result(frame_rgb: np.ndarray, box_xyxy: np.ndarray, expand_ratio: float = 0.1) -> Optional[Image.Image]:
    """Crops a face from a frame given a bounding box, with a small expansion."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    box_width, box_height = x2 - x1, y2 - y1
    x1_exp = max(0, int(x1 - box_width * expand_ratio))
    y1_exp = max(0, int(y1 - box_height * expand_ratio))
    x2_exp = min(w, int(x2 + box_width * expand_ratio))
    y2_exp = min(h, int(y2 + box_height * expand_ratio))
    
    face_crop = frame_rgb[y1_exp:y2_exp, x1_exp:x2_exp]
    return Image.fromarray(face_crop) if face_crop.size > 0 else None


def detect_faces_batch(
    yolo_model: YOLO, frames_rgb: List[np.ndarray], device: torch.device, conf: float, imgsz: int, half: bool, batch_size: int
) -> List[Optional[Image.Image]]:
    """Detects the most prominent face in a batch of frames."""
    if not frames_rgb:
        return []

    ultra_device = 0 if device.type == "cuda" else "cpu"
    results = yolo_model.predict(
        source=frames_rgb, conf=conf, imgsz=imgsz, max_det=1, device=ultra_device, half=half, verbose=False, batch=batch_size
    )

    faces: List[Optional[Image.Image]] = []
    for frame, res in zip(frames_rgb, results):
        if res.boxes and res.boxes.xyxy.shape[0] > 0:
            box = res.boxes.xyxy[0].cpu().numpy()
            faces.append(_crop_face_from_result(frame, box))
        else:
            faces.append(None)
    return faces


def predict_expressions_batch(
    processor: AutoImageProcessor, expr_model: AutoModelForImageClassification, faces: List[Optional[Image.Image]], device: torch.device, amp_dtype: Optional[torch.dtype]
) -> List[Optional[Dict[str, float]]]:
    """Predicts emotion probabilities for a batch of face crops."""
    valid_faces = [img for img in faces if img is not None]
    if not valid_faces:
        return [None] * len(faces)

    inputs = processor(images=valid_faces, return_tensors="pt").to(device)

    with torch.no_grad():
        if device.type == "cuda" and amp_dtype:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = expr_model(**inputs).logits
        else:
            logits = expr_model(**inputs).logits

    # Use softmax to get probabilities for all emotions
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    labels = expr_model.config.id2label
    results_list = []
    for prob_tensor in probabilities:
        prob_dict = {labels[i]: p.item() * 100 for i, p in enumerate(prob_tensor)}
        results_list.append(prob_dict)

    # Re-insert None for frames where no face was detected
    result_iter = iter(results_list)
    return [next(result_iter) if face else None for face in faces]


# ──────────────────────────────────────────────────────────────────────────────
# Core Processing Logic
# ──────────────────────────────────────────────────────────────────────────────
def _run_batch(
    times_batch: List[float], frames_batch: List[np.ndarray], models: ModelBundle, params: dict
) -> List[Tuple[float, Optional[Dict[str, float]]]]:
    """Processes a single batch of frames for face detection and emotion classification."""
    faces = detect_faces_batch(
        models.yolo, frames_batch, models.device, conf=params["yolo_conf"], imgsz=params["yolo_imgsz"],
        half=(models.amp_dtype is torch.float16), batch_size=params["batch_size"]
    )
    emotions = predict_expressions_batch(
        models.processor, models.expr_model, faces, models.device, models.amp_dtype
    )
    return list(zip(times_batch, emotions))


def process_video(
    video_path: str, models: ModelBundle, params: dict, progress_cb: callable
) -> pd.DataFrame:
    """
    Analyzes the video to get emotion probabilities at each step.
    Returns a DataFrame with columns: ['Timestamp (s)', 'angry', 'disgust', ...].
    """
    fps, _, duration = get_video_info(video_path)
    if duration == 0:
        raise ValueError("Видео имеет нулевую длительность или не может быть прочитано.")

    step_sec = 1.0  # Analyze one frame per second
    total_steps = max(1, int(np.ceil(duration / step_sec)))
    
    all_results: List[Tuple[float, Optional[Dict[str, float]]]] = []
    times_buf, frames_buf = [], []

    frame_iterator = iterate_frames(video_path, fps, duration, step_sec)
    for i, (t, frame_rgb) in enumerate(frame_iterator):
        times_buf.append(t)
        frames_buf.append(frame_rgb)

        if len(frames_buf) >= params["batch_size"]:
            all_results.extend(_run_batch(times_buf, frames_buf, models, params))
            times_buf.clear()
            frames_buf.clear()
            progress_cb(min((i + 1) / total_steps, 1.0))

    if frames_buf:
        all_results.extend(_run_batch(times_buf, frames_buf, models, params))
        progress_cb(1.0)
    
    if not all_results:
        return pd.DataFrame()

    records = []
    for timestamp, emotions_dict in all_results:
        record = {"Timestamp (s)": round(timestamp, 2)}
        if emotions_dict:
            for emotion, percentage in emotions_dict.items():
                record[emotion] = round(percentage, 2)
        else: # Fill with 0 if no face is detected
            for emotion in EMOTION_OPTIONS:
                record[emotion] = 0.0
        records.append(record)
    
    df = pd.DataFrame(records)
    
    for emotion in EMOTION_OPTIONS: # Ensure all emotion columns exist
        if emotion not in df.columns:
            df[emotion] = 0.0
            
    return df[["Timestamp (s)"] + EMOTION_OPTIONS]


# ──────────────────────────────────────────────────────────────────────────────
# Main Application UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Main function to run the Streamlit application."""
    st.title("📊 Анализ процентного соотношения эмоций на видео")
    st.markdown("Загрузите видеофайл, чтобы увидеть динамику эмоций с течением времени.")

    with st.sidebar:
        st.title("⚙️ Настройки")
        st.subheader("Параметры производительности")
        
        use_gpu = torch.cuda.is_available()
        use_fp16 = use_gpu
        
        st.info(f"Доступен GPU: {'Да' if use_gpu else 'Нет'}. Обработка на **{'GPU' if use_gpu else 'CPU'}**.")

        batch_size = st.slider("Размер пакета (Batch Size)", 4, 128, 64, step=4)
        yolo_imgsz = st.select_slider("Размер изображения YOLO", options=[640, 960, 1280], value=960)
        yolo_conf = st.slider("Порог уверенности YOLO", 0.10, 0.50, 0.25, step=0.05)
        
        perf_params = {
            "use_gpu": use_gpu, "use_fp16": use_fp16, "batch_size": batch_size,
            "yolo_imgsz": yolo_imgsz, "yolo_conf": yolo_conf,
        }

        st.divider()
        st.subheader("Распознаваемые эмоции")
        emotion_list = [f"- **{EMOTION_TRANSLATIONS.get(e)}** ({e})" for e in EMOTION_OPTIONS]
        st.markdown("\n".join(emotion_list))

    uploaded_file = st.file_uploader(
        "Выберите видеофайл", type=["mp4", "mov", "avi", "mkv"],
        on_change=lambda: st.session_state.update({STATE["results_df"]: None, STATE["video_path"]: None})
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state[STATE["video_path"]] = tmp_file.name

        st.video(st.session_state[STATE["video_path"]])
        
        fps, _, duration = get_video_info(st.session_state[STATE["video_path"]])
        c1, c2 = st.columns(2)
        c1.metric("Длительность видео", f"{duration:.2f} сек")
        c2.metric("Кадров в секунду (FPS)", f"{fps:.2f}")

        if st.button("🚀 Начать анализ", type="primary", use_container_width=True):
            models = load_models(use_gpu=perf_params["use_gpu"], use_fp16=perf_params["use_fp16"])
            if models:
                progress_bar = st.progress(0, text="Инициализация...")
                try:
                    df = process_video(
                        video_path=st.session_state[STATE["video_path"]], models=models, params=perf_params,
                        progress_cb=lambda p: progress_bar.progress(p, text=f"Обработка... {int(p*100)}%")
                    )
                    st.session_state[STATE["results_df"]] = df
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"Произошла ошибка при обработке: {e}")
                    progress_bar.empty()

    if st.session_state[STATE["results_df"]] is not None:
        st.divider()
        st.subheader("📈 Результаты анализа")
        time_series_df = st.session_state[STATE["results_df"]]

        if time_series_df.empty or time_series_df.drop("Timestamp (s)", axis=1).sum().sum() == 0:
            st.warning("Не удалось обнаружить эмоции в видео. Возможно, на видео нет лиц.")
        else:
            # --- 1. Summary: Average percentages ---
            st.write("#### Сводка: Средний процент эмоций")
            st.info("Средние значения рассчитаны только для тех моментов, когда было обнаружено лицо.")
            
            emotion_cols = time_series_df[EMOTION_OPTIONS]
            valid_frames = emotion_cols.sum(axis=1) > 1 # Sum > 1 to be safe with float precision
            
            avg_emotions = emotion_cols[valid_frames].mean() if valid_frames.any() else pd.Series(0.0, index=EMOTION_OPTIONS)

            summary_df = pd.DataFrame({
                "Эмоция": [EMOTION_TRANSLATIONS.get(e, e) for e in avg_emotions.index],
                "Средний процент (%)": avg_emotions.values.round(2)
            }).sort_values(by="Средний процент (%)", ascending=False).reset_index(drop=True)

            chart_data = summary_df.rename(columns={"Средний процент (%)": "Процент"}).set_index("Эмоция")
            st.bar_chart(chart_data["Процент"])
            st.dataframe(summary_df, use_container_width=True)

            # --- 2. Time-series chart and data ---
            st.divider()
            st.write("#### Динамика эмоций во времени")
            
            chart_df = time_series_df.set_index("Timestamp (s)")
            chart_df.columns = [EMOTION_TRANSLATIONS.get(col, col) for col in chart_df.columns]
            st.line_chart(chart_df)
            
            st.write("#### Детальные данные по времени (в процентах)")
            display_df = time_series_df.copy()
            display_df.columns = ["Timestamp (s)"] + [EMOTION_TRANSLATIONS.get(e, e) for e in EMOTION_OPTIONS]
            st.dataframe(display_df, use_container_width=True)

            # --- 3. CSV Download ---
            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            
            video_name = Path(uploaded_file.name).stem
            st.download_button(
                label="📥 Скачать детальные результаты (CSV)", data=csv_buffer.getvalue(),
                file_name=f"{video_name}_emotion_time_series.csv", mime="text/csv", type="primary"
            )

if __name__ == "__main__":
    main()