from __future__ import annotations
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Video Emotion Analysis & Review", page_icon="🎭", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Constants & state
# ──────────────────────────────────────────────────────────────────────────────
EMOTION_OPTIONS: List[str] = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

STATE = {
    "page": "page",
    "csv": "processed_csv",
    "video": "video_path",
    "review_df": "review_data",
    "idx": "current_transition",
    "show_corr": "show_correction",
}

PAGES = {"processing": "processing", "review": "review"}

# Defaults
for key, default in [
    (STATE["page"], PAGES["processing"]),
    (STATE["csv"], None),
    (STATE["video"], None),
    (STATE["review_df"], None),
    (STATE["idx"], 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    yolo: YOLO
    processor: AutoImageProcessor
    expr_model: AutoModelForImageClassification
    device: torch.device
    amp_dtype: Optional[torch.dtype]


# ──────────────────────────────────────────────────────────────────────────────
# Device and models
# ──────────────────────────────────────────────────────────────────────────────
def get_device(use_gpu: bool = True) -> torch.device:
    return torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_models(use_gpu: bool = True, use_fp16: bool = True) -> Optional[ModelBundle]:
    """
    Load YOLO face detector and ViT expression classifier with GPU/AMP settings.
    Cached per (use_gpu, use_fp16) to let you switch in the sidebar.
    """
    device = get_device(use_gpu)

    # Perf knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")  # PyTorch 2.x
        except Exception:
            pass

    try:
        # YOLO (Ultralytics)
        yolo = YOLO("models/yolov12n-face.pt")  # make sure the file exists locally
        try:
            yolo.to("cuda" if device.type == "cuda" else "cpu")
        except Exception:
            pass
        try:
            yolo.fuse()  # small speedup if supported
        except Exception:
            pass

        # HF classifier
        processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
        expr_model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression").to(device).eval()

        amp_dtype = torch.float16 if (device.type == "cuda" and use_fp16) else None
        return ModelBundle(yolo=yolo, processor=processor, expr_model=expr_model, device=device, amp_dtype=amp_dtype)

    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Video utilities
# ──────────────────────────────────────────────────────────────────────────────
def get_video_info(video_path: str) -> Tuple[float, int, float, int, int]:
    """Return (fps, frame_count, duration_sec, width, height)."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        return fps, frame_count, duration, width, height
    finally:
        cap.release()


def iterate_frames(video_path: str, fps: float, duration: float, step_sec: float) -> Generator[Tuple[float, np.ndarray], None, None]:
    """Yield (cur_time_sec, frame_rgb_np) every step_sec."""
    cap = cv2.VideoCapture(video_path)
    try:
        cur_time = 0.0
        while cur_time < duration:
            frame_idx = int(round(cur_time * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield cur_time, frame_rgb
            cur_time += step_sec
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Batched inference
# ──────────────────────────────────────────────────────────────────────────────
def _crop_face_from_result(frame_rgb: np.ndarray, box_xyxy: np.ndarray, expand_ratio: float = 0.10) -> Optional[Image.Image]:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    x1e = max(0, int(x1 - bw * expand_ratio))
    y1e = max(0, int(y1 - bh * expand_ratio))
    x2e = min(w, int(x2 + bw * expand_ratio))
    y2e = min(h, int(y2 + bh * expand_ratio))
    face = frame_rgb[y1e:y2e, x1e:x2e]
    if face.size == 0:
        return None
    return Image.fromarray(face)


def detect_faces_batch(
    yolo_model: YOLO,
    frames_rgb: List[np.ndarray],
    device: torch.device,
    conf: float = 0.25,
    imgsz: int = 960,
    half: bool = True,
    batch: int = 16,
) -> List[Optional[Image.Image]]:
    """frames_rgb -> list of PIL face crops (or None)"""
    if len(frames_rgb) == 0:
        return []
    ultra_device = 0 if device.type == "cuda" else "cpu"
    results = yolo_model.predict(
        source=frames_rgb,
        conf=conf,
        imgsz=imgsz,
        max_det=1,
        device=ultra_device,
        half=(half and device.type == "cuda"),
        verbose=False,
        batch=batch,
    )
    faces: List[Optional[Image.Image]] = []
    for frame, res in zip(frames_rgb, results):
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
            faces.append(None)
        else:
            box = boxes.xyxy[0].detach().cpu().numpy()
            faces.append(_crop_face_from_result(frame, box))
    return faces


def predict_expressions_batch(
    processor: AutoImageProcessor,
    expr_model: AutoModelForImageClassification,
    faces: List[Optional[Image.Image]],
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = torch.float16,
) -> List[str]:
    """faces -> list of emotion labels; 'no_face' where crop is None."""
    valid_imgs = [img for img in faces if img is not None]
    if len(valid_imgs) == 0:
        return ["no_face"] * len(faces)

    inputs = processor(images=valid_imgs, return_tensors="pt").to(device)

    with torch.no_grad():
        if device.type == "cuda" and amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = expr_model(**inputs).logits
        else:
            logits = expr_model(**inputs).logits

    preds = logits.softmax(dim=1).argmax(dim=1).tolist()
    labels = [expr_model.config.id2label[i] for i in preds]

    # reinsert Nones
    out: List[str] = []
    j = 0
    for img in faces:
        if img is None:
            out.append("no_face")
        else:
            out.append(labels[j])
            j += 1
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────
def process_video(
    video_path: str,
    models: ModelBundle,
    step_sec: float = 1.0,
    batch_size: int = 16,
    imgsz: int = 960,
    conf: float = 0.25,
    progress_cb: Optional[callable] = None,
) -> pd.DataFrame:
    """
    Batched processing to maximize GPU utilization.
    Returns DataFrame: time_from, time_to, emotion_from, emotion_to
    """
    fps, frame_count, duration, *_ = get_video_info(video_path)
    if fps <= 0 or frame_count == 0 or duration == 0:
        raise RuntimeError("Невалидное видео: FPS/frames/duration = 0.")

    # collect per-step emotions
    per_step: List[Tuple[float, str]] = []

    total_steps = max(1, int(np.ceil(duration / step_sec)))
    done_steps = 0

    times_buf: List[float] = []
    frames_buf: List[np.ndarray] = []

    for t, frame_rgb in iterate_frames(video_path, fps, duration, step_sec):
        times_buf.append(t)
        frames_buf.append(frame_rgb)

        if len(frames_buf) >= batch_size:
            _run_batch(times_buf, frames_buf, models, imgsz, conf, batch_size, per_step)
            done_steps += len(frames_buf)
            if progress_cb:
                progress_cb(min(done_steps / total_steps, 1.0))
            times_buf.clear()
            frames_buf.clear()

    if frames_buf:
        _run_batch(times_buf, frames_buf, models, imgsz, conf, batch_size, per_step)
        done_steps += len(frames_buf)
        if progress_cb:
            progress_cb(min(done_steps / total_steps, 1.0))
        times_buf.clear()
        frames_buf.clear()

    # Build transitions
    per_step.sort(key=lambda x: x[0])
    final = []
    prev_emotion: Optional[str] = None
    segment_start = 0.0
    for t, emotion in per_step:
        if prev_emotion is None:
            prev_emotion = emotion
            segment_start = t
        elif emotion != prev_emotion:
            final.append({
                "time_from": round(segment_start, 2),
                "time_to": round(t, 2),
                "emotion_from": prev_emotion,
                "emotion_to": emotion,
            })
            segment_start = t
            prev_emotion = emotion

    if prev_emotion is not None:
        final.append({
            "time_from": round(segment_start, 2),
            "time_to": round(duration, 2),
            "emotion_from": prev_emotion,
            "emotion_to": prev_emotion,
        })

    return pd.DataFrame(final)


def _run_batch(
    times_batch: List[float],
    frames_batch: List[np.ndarray],
    models: ModelBundle,
    imgsz: int,
    conf: float,
    batch_size: int,
    collect: List[Tuple[float, str]],
) -> None:
    faces = detect_faces_batch(
        models.yolo,
        frames_batch,
        models.device,
        conf=conf,
        imgsz=imgsz,
        half=(models.amp_dtype is torch.float16),
        batch=batch_size,
    )
    emotions = predict_expressions_batch(models.processor, models.expr_model, faces, models.device, models.amp_dtype)
    for t, e in zip(times_batch, emotions):
        collect.append((float(t), str(e)))


# ──────────────────────────────────────────────────────────────────────────────
# Frame & clip helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_frame_at_time(video_path: str, time_sec: float) -> Optional[np.ndarray]:
    fps, *_ = get_video_info(video_path)
    if fps <= 0:
        return None
    frame_idx = int(round(time_sec * fps))
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def extract_video_clip(video_path: str, start_time: float, end_time: float, output_path: str) -> None:
    fps, _, _, w, h = get_video_info(video_path)
    if fps <= 0:
        raise RuntimeError("Invalid FPS; cannot write clip.")
    cap = cv2.VideoCapture(video_path)
    try:
        start_frame = int(round(start_time * fps))
        end_frame = int(round(end_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        try:
            for _ in range(max(0, end_frame - start_frame)):
                ok, frame = cap.read()
                if not ok:
                    break
                out.write(frame)
        finally:
            out.release()
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# UI: Processing page
# ──────────────────────────────────────────────────────────────────────────────
def page_processing(perf_opts: dict) -> None:
    st.title("🎭 Анализ эмоций на видео")
    st.markdown("Загрузите видеофайл для анализа эмоциональных переходов")

    uploaded_file = st.file_uploader(
        "Выберите видеофайл",
        type=["mp4", "avi", "mov", "mkv"],
        help="Поддерживаемые форматы: MP4, AVI, MOV, MKV",
    )

    if uploaded_file is not None:
        # Persist uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix or ".mp4") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state[STATE["video"]] = tmp.name

        fps, frame_count, duration, *_ = get_video_info(st.session_state[STATE["video"]])
        c1, c2, c3 = st.columns(3)
        c1.metric("Длительность", f"{duration:.2f} сек")
        c2.metric("Кадров в секунду", f"{fps:.2f}")
        c3.metric("Всего кадров", frame_count)

        st.subheader("Параметры обработки")
        # step_sec = st.slider("Интервал анализа (секунды)", 0.5, 5.0, 1.0, 0.5)
        step_sec = 1.0

        if st.button("🔄 Обработать видео", type="primary"):
            with st.spinner("Загрузка моделей…"):
                models = load_models(use_gpu=perf_opts["use_gpu"], use_fp16=perf_opts["use_fp16"])
            if models is None:
                st.error("Не удалось загрузить модели. Проверьте наличие файлов моделей.")
                return

            pbar = st.progress(0)
            status = st.empty()
            status.text("Обработка видео…")

            try:
                df = process_video(
                    video_path=st.session_state[STATE["video"]],
                    models=models,
                    step_sec=step_sec,
                    batch_size=perf_opts["batch_size"],
                    imgsz=perf_opts["yolo_imgsz"],
                    conf=perf_opts["yolo_conf"],
                    progress_cb=pbar.progress,
                )
                # Save CSV temp for review page
                video_path = st.session_state[STATE["video"]]
                video_name = Path(video_path).stem
                csv_name = f"{video_name}.csv"
                csv_path = str(Path(tempfile.gettempdir()) / csv_name)
                df.to_csv(csv_path, index=False)
                st.session_state[STATE["csv"]] = csv_path

                status.text("Обработка завершена!")
                st.subheader("📊 Результаты обработки")
                c1, c2 = st.columns(2)
                c1.metric("Всего переходов", len(df))
                unique_emotions = set(df["emotion_from"].tolist() + df["emotion_to"].tolist())
                c2.metric("Уникальных эмоций", len(unique_emotions))

                st.subheader("📋 Предпросмотр переходов эмоций")
                st.dataframe(df, use_container_width=True)

                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button(
                    label="📥 Скачать CSV",
                    data=buf.getvalue(),
                    file_name=csv_name,
                    mime="text/csv",
                )

                if st.button("➡️ Перейти к проверке", type="primary"):
                    st.session_state[STATE["page"]] = PAGES["review"]
                    st.rerun()

            except Exception as e:
                st.error(f"Ошибка при обработке видео: {e}")
                status.text("Обработка не удалась!")


# ──────────────────────────────────────────────────────────────────────────────
# UI: Review page
# ──────────────────────────────────────────────────────────────────────────────
def ensure_review_df() -> Optional[pd.DataFrame]:
    csv_path = st.session_state.get(STATE["csv"])
    if not csv_path:
        return None
    if st.session_state.get(STATE["review_df"]) is None:
        base = pd.read_csv(csv_path)
        base = base.copy()
        base["review"] = "pending"
        base["reviewed_emotion_from"] = ""
        base["reviewed_emotion_to"] = ""
        base["comment"] = ""
        st.session_state[STATE["review_df"]] = base
    return st.session_state[STATE["review_df"]]


def page_review() -> None:
    st.title("✅ Проверка переходов эмоций")

    if not st.session_state.get(STATE["csv"]):
        st.warning("Нет обработанных данных. Сначала обработайте видео.")
        if st.button("← Назад к обработке"):
            st.session_state[STATE["page"]] = PAGES["processing"]
            st.rerun()
        return

    df = ensure_review_df()
    if df is None or df.empty:
        st.warning("Нет переходов для проверки.")
        return

    idx: int = int(st.session_state.get(STATE["idx"], 0))
    idx = max(0, min(idx, len(df) - 1))

    st.progress((idx + 1) / len(df))
    st.markdown(f"**Переход {idx + 1} из {len(df)}**")

    row = df.iloc[idx]
    st.subheader(f"Переход: {row['emotion_from']} → {row['emotion_to']}")
    st.markdown(f"**Время:** с {row['time_from']}с по {row['time_to']}с")

    c1, c2 = st.columns([2, 2])
    with c1:
        st.markdown("**Начальный кадр**")
        start = extract_frame_at_time(st.session_state[STATE["video"]], float(row["time_from"]))
        if start is not None:
            st.image(start, caption=f"В {row['time_from']}с", use_container_width=True)
        else:
            st.info("Нет кадра для начального времени.")
    with c2:
        st.markdown("**Конечный кадр**")
        end = extract_frame_at_time(st.session_state[STATE["video"]], float(row["time_to"]))
        if end is not None:
            st.image(end, caption=f"В {row['time_to']}с", use_container_width=True)
        else:
            st.info("Нет кадра для конечного времени.")

    st.divider()

    st.subheader("Проверка перехода")
    st.markdown("Одобрите переход, если он корректен, или отклоните и внесите исправления.")

    if row["review"] != "pending":
        status_color = "🟢" if row["review"] == "approved" else "🔴"
        st.markdown(f"**Статус:** {status_color} {row['review'].title()}")
        if row["reviewed_emotion_from"] or row["reviewed_emotion_to"]:
            st.markdown(f"**Исправлено:** {row['reviewed_emotion_from']} → {row['reviewed_emotion_to']}")
        if row["comment"]:
            st.markdown(f"**Комментарий:** {row['comment']}")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("✅ Одобрить", type="primary", use_container_width=True):
            df.loc[idx, "review"] = "approved"
            st.session_state[STATE["review_df"]] = df
            if idx < len(df) - 1:
                st.session_state[STATE["idx"]] = idx + 1
            st.rerun()
    with b2:
        if st.button("❌ Отклонить", use_container_width=True):
            df.loc[idx, "review"] = "rejected"
            st.session_state[STATE["review_df"]] = df
            st.session_state[STATE["show_corr"]] = True
            st.rerun()

    if row["review"] == "rejected" or st.session_state.get(STATE["show_corr"], False):
        st.subheader("Интерфейс исправления")
        st.markdown("Выберите правильные эмоции и добавьте комментарий при необходимости.")

        c1, c2 = st.columns(2)
        with c1:
            corrected_from = st.selectbox(
                "Исправленная эмоция (от):",
                options=[""] + EMOTION_OPTIONS,
                index=0 if not row["reviewed_emotion_from"] else EMOTION_OPTIONS.index(row["reviewed_emotion_from"]) + 1,
            )
        with c2:
            corrected_to = st.selectbox(
                "Исправленная эмоция (к):",
                options=[""] + EMOTION_OPTIONS,
                index=0 if not row["reviewed_emotion_to"] else EMOTION_OPTIONS.index(row["reviewed_emotion_to"]) + 1,
            )

        comment = st.text_area("Комментарий:", value=row["comment"])

        if st.button("💾 Сохранить исправление"):
            df.loc[idx, "reviewed_emotion_from"] = corrected_from
            df.loc[idx, "reviewed_emotion_to"] = corrected_to
            df.loc[idx, "comment"] = comment
            st.session_state[STATE["review_df"]] = df
            st.session_state[STATE["show_corr"]] = False
            st.success("Исправление сохранено!")
            if idx < len(df) - 1:
                st.session_state[STATE["idx"]] = idx + 1
            st.rerun()

    st.divider()

    st.subheader("Навигация")
    n1, n2, n3 = st.columns([1, 2, 1])
    with n1:
        if st.button("⬅️ Предыдущий", disabled=(idx == 0)):
            st.session_state[STATE["idx"]] = max(0, idx - 1)
            st.rerun()
    with n2:
        jump_to = st.number_input("Перейти к переходу:", min_value=1, max_value=len(df), value=idx + 1)
        if int(jump_to) - 1 != idx:
            st.session_state[STATE["idx"]] = int(jump_to) - 1
            st.rerun()
    with n3:
        if st.button("➡️ Следующий", disabled=(idx == len(df) - 1)):
            st.session_state[STATE["idx"]] = min(len(df) - 1, idx + 1)
            st.rerun()

    st.divider()

    st.subheader("Экспорт результатов")
    approved = int((df["review"] == "approved").sum())
    rejected = int((df["review"] == "rejected").sum())
    pending = int((df["review"] == "pending").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Одобрено", approved)
    c2.metric("❌ Отклонено", rejected)
    c3.metric("⏳ В ожидании", pending)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label="📥 Экспортировать итоговый CSV",
        data=buf.getvalue(),
        file_name="reviewed_emotion_transitions.csv",
        mime="text/csv",
        type="primary",
    )

    if st.button("← Назад к обработке"):
        st.session_state[STATE]["page"] = PAGES["processing"]
        st.session_state[STATE]["csv"] = None
        st.session_state[STATE]["review_df"] = None
        st.session_state[STATE]["idx"] = 0
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main router + sidebar
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.sidebar.title("Навигация")
    if st.sidebar.button("📹 Обработка видео", use_container_width=True):
        st.session_state[STATE["page"]] = PAGES["processing"]
    if st.sidebar.button("✅ Проверка переходов", use_container_width=True):
        st.session_state[STATE["page"]] = PAGES["review"]

    # Perf controls
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Производительность")
    use_gpu = st.sidebar.checkbox("Использовать GPU (CUDA)", value=torch.cuda.is_available())
    use_fp16 = st.sidebar.checkbox("FP16/AMP", value=True)
    batch_size = st.sidebar.slider("Batch size", 4, 64, 16, step=4)
    # yolo_imgsz = st.sidebar.select_slider("YOLO img size", options=[640, 768, 896, 960, 1024, 1280], value=960)
    # yolo_conf = st.sidebar.slider("YOLO conf", 0.10, 0.60, 0.25, step=0.05)
    yolo_imgsz = 1280
    yolo_conf = 0.25
    perf_opts = {
        "use_gpu": use_gpu,
        "use_fp16": use_fp16,
        "batch_size": batch_size,
        "yolo_imgsz": yolo_imgsz,
        "yolo_conf": yolo_conf,
    }

    # Add emotion list info to sidebar
    # Add emotion list info to sidebar with Russian translations
    emotion_translations = {
        "Angry": "Злость",
        "Disgust": "Отвращение",
        "Fear": "Страх",
        "Happy": "Счастье",
        "Sad": "Грусть",
        "Surprise": "Удивление",
        "Neutral": "Нейтрально",
    }
    emotion_list = [f"{e} ({emotion_translations.get(e, '')})" for e in EMOTION_OPTIONS]
    st.sidebar.info("**Эмоции:**\n" + ", ".join(emotion_list))

    page = st.session_state.get(STATE["page"], PAGES["processing"])
    if page == PAGES["processing"]:
        page_processing(perf_opts)
    elif page == PAGES["review"]:
        page_review()
    else:
        st.session_state[STATE["page"]] = PAGES["processing"]
        page_processing(perf_opts)


if __name__ == "__main__":
    main()