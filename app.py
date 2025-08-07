import streamlit as st
import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
import tempfile
import base64
from pathlib import Path
import io

# Configure page
st.set_page_config(
    page_title="Video Emotion Analysis & Review",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'processing'
if 'processed_csv' not in st.session_state:
    st.session_state.processed_csv = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'review_data' not in st.session_state:
    st.session_state.review_data = None
if 'current_transition' not in st.session_state:
    st.session_state.current_transition = 0

# Emotion detection code
@st.cache_resource
def load_models():
    """Load YOLO face detection and emotion classification models"""
    try:
        yolo = YOLO('models/yolov12n-face.pt')
        processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
        expr_model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
        return yolo, processor, expr_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def detect_face(yolo_model, frame):
    """Detect face in frame and return cropped face image"""
    results = yolo_model.predict(
        source=frame,
        conf=0.25,
        imgsz=1280,
        line_width=1,
        max_det=1,
        verbose=False
    )
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        x1, y1, x2, y2 = boxes[0]
        w, h = x2 - x1, y2 - y1
        expand_ratio = 0.1
        x1e = max(0, int(x1 - w * expand_ratio))
        y1e = max(0, int(y1 - h * expand_ratio))
        x2e = min(frame.width, int(x2 + w * expand_ratio))
        y2e = min(frame.height, int(y2 + h * expand_ratio))
        face_img = frame.crop((x1e, y1e, x2e, y2e))
        return face_img
    return None

def predict_expression(processor, expr_model, face_img):
    """Predict emotion from face image"""
    inputs = processor(images=face_img, return_tensors="pt")
    with torch.no_grad():
        outputs = expr_model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    pred = probs.argmax(dim=1).item()
    label = expr_model.config.id2label[pred]
    return label

def process_video(video_path, yolo_model, processor, expr_model, step_sec=1.0, output_csv='output.csv', progress_bar=None):
    """Process video and generate emotion transitions CSV"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    results = []
    cur_time = 0.0
    
    prev_emotion = None
    segment_start = 0.0
    
    total_steps = int(duration / step_sec)
    step = 0
    
    while cur_time < duration:
        frame_idx = int(cur_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_img = detect_face(yolo_model, pil_img)
        
        if face_img is not None:
            emotion = predict_expression(processor, expr_model, face_img)
        else:
            emotion = "no_face"
        
        if prev_emotion is None:
            prev_emotion = emotion
            segment_start = cur_time
        elif emotion != prev_emotion:
            results.append({
                'time_from': round(segment_start, 2),
                'time_to': round(cur_time, 2),
                'emotion_from': prev_emotion,
                'emotion_to': emotion
            })
            segment_start = cur_time
            prev_emotion = emotion
        
        cur_time += step_sec
        step += 1
        
        if progress_bar:
            progress_bar.progress(min(step / total_steps, 1.0))
    
    if prev_emotion is not None and segment_start < duration:
        results.append({
            'time_from': round(segment_start, 2),
            'time_to': round(duration, 2),
            'emotion_from': prev_emotion,
            'emotion_to': prev_emotion
        })
    
    cap.release()
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df

def extract_frame_at_time(video_path, time_sec):
    """Extract frame from video at specific time"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def extract_video_clip(video_path, start_time, end_time, output_path):
    """Extract video clip between timestamps"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()

def video_processing_page():
    """Page 1: Video Processing"""
    st.title("🎭 Анализ эмоций на видео")
    st.markdown("Загрузите видеофайл для анализа эмоциональных переходов")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Выберите видеофайл",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Поддерживаемые форматы: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            st.session_state.video_path = video_path
        
        # Display video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Длительность", f"{duration:.2f} сек")
        with col2:
            st.metric("Кадров в секунду", f"{fps:.2f}")
        with col3:
            st.metric("Всего кадров", frame_count)
        
        # Processing parameters
        st.subheader("Параметры обработки")
        step_sec = st.slider("Интервал анализа (секунды)", 0.5, 5.0, 1.0, 0.5)
        
        # Process button
        if st.button("🔄 Обработать видео", type="primary"):
            with st.spinner("Загрузка моделей..."):
                yolo_model, processor, expr_model = load_models()
                
            if yolo_model is None:
                st.error("Не удалось загрузить модели. Проверьте наличие файлов моделей.")
                return
            
            # Process video with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Обработка видео...")
            
            try:
                # Create temporary CSV file
                csv_path = tempfile.mktemp(suffix='.csv')
                
                df = process_video(
                    video_path, yolo_model, processor, expr_model, 
                    step_sec=step_sec, output_csv=csv_path, progress_bar=progress_bar
                )
                
                st.session_state.processed_csv = csv_path
                status_text.text("Обработка завершена!")
                
                # Display results summary
                st.subheader("📊 Результаты обработки")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Всего переходов", len(df))
                
                with col2:
                    unique_emotions = set(df['emotion_from'].tolist() + df['emotion_to'].tolist())
                    st.metric("Уникальных эмоций", len(unique_emotions))
                
                # Display CSV preview
                st.subheader("📋 Предпросмотр переходов эмоций")
                st.dataframe(df, use_container_width=True)
                
                # Download CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Скачать CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"emotion_transitions_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
                
                # Button to go to review page
                if st.button("➡️ Перейти к проверке", type="primary"):
                    st.session_state.page = 'review'
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Ошибка при обработке видео: {str(e)}")
                status_text.text("Обработка не удалась!")

def review_interface_page():
    """Page 2: Review Interface"""
    st.title("✅ Проверка переходов эмоций")
    
    if st.session_state.processed_csv is None:
        st.warning("Нет обработанных данных. Сначала обработайте видео.")
        if st.button("← Назад к обработке"):
            st.session_state.page = 'processing'
            st.rerun()
        return
    
    # Load review data
    if st.session_state.review_data is None:
        df = pd.read_csv(st.session_state.processed_csv)
        # Add review columns
        df['review'] = 'pending'
        df['reviewed_emotion_from'] = ''
        df['reviewed_emotion_to'] = ''
        df['comment'] = ''
        st.session_state.review_data = df
    
    df = st.session_state.review_data
    current_idx = st.session_state.current_transition
    
    if len(df) == 0:
        st.warning("Нет переходов для проверки.")
        return
    
    # Progress indicator
    st.progress((current_idx + 1) / len(df))
    st.markdown(f"**Переход {current_idx + 1} из {len(df)}**")
    
    # Current transition data
    row = df.iloc[current_idx]
    
    # Display transition info
    st.subheader(f"Переход: {row['emotion_from']} → {row['emotion_to']}")
    st.markdown(f"**Время:** с {row['time_from']}с по {row['time_to']}с")
    
    # Create columns for frames (wider for better visibility)
    col1, col2 = st.columns([2, 2])

    with col1:
        st.markdown("**Начальный кадр**")
        start_frame = extract_frame_at_time(st.session_state.video_path, row['time_from'])
        if start_frame is not None:
            st.image(start_frame, caption=f"В {row['time_from']}с", use_container_width=True)
        else:
            st.info("Нет кадра для начального времени.")

    with col2:
        st.markdown("**Конечный кадр**")
        end_frame = extract_frame_at_time(st.session_state.video_path, row['time_to'])
        if end_frame is not None:
            st.image(end_frame, caption=f"В {row['time_to']}с", use_container_width=True)
        else:
            st.info("Нет кадра для конечного времени.")

    st.divider()

    # Review controls
    st.subheader("Проверка перехода")
    st.markdown("Одобрите переход, если он корректен, или отклоните и внесите исправления.")

    # Show current review status
    if row['review'] != 'pending':
        status_color = "🟢" if row['review'] == 'approved' else "🔴"
        st.markdown(f"**Статус:** {status_color} {row['review'].title()}")
        if row['reviewed_emotion_from'] or row['reviewed_emotion_to']:
            st.markdown(f"**Исправлено:** {row['reviewed_emotion_from']} → {row['reviewed_emotion_to']}")
        if row['comment']:
            st.markdown(f"**Комментарий:** {row['comment']}")

    # Review buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("✅ Одобрить", type="primary", use_container_width=True, help="Отметить переход как корректный"):
            df.loc[current_idx, 'review'] = 'approved'
            st.session_state.review_data = df
            if current_idx < len(df) - 1:
                st.session_state.current_transition += 1
            st.rerun()

    with col2:
        if st.button("❌ Отклонить", use_container_width=True, help="Отметить переход как некорректный и внести исправления"):
            df.loc[current_idx, 'review'] = 'rejected'
            st.session_state.review_data = df
            st.session_state.show_correction = True
            st.rerun()

    # Correction interface
    if row['review'] == 'rejected' or st.session_state.get('show_correction', False):
        st.subheader("Интерфейс исправления")
        st.markdown("Выберите правильные эмоции и добавьте комментарий при необходимости.")

        emotion_options = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        col1, col2 = st.columns(2)
        with col1:
            corrected_from = st.selectbox(
                "Исправленная эмоция (от):",
                options=[''] + emotion_options,
                index=0 if not row['reviewed_emotion_from'] else emotion_options.index(row['reviewed_emotion_from']) + 1
            )

        with col2:
            corrected_to = st.selectbox(
                "Исправленная эмоция (к):",
                options=[''] + emotion_options,
                index=0 if not row['reviewed_emotion_to'] else emotion_options.index(row['reviewed_emotion_to']) + 1
            )

        comment = st.text_area("Комментарий:", value=row['comment'], help="Необязательно: добавьте комментарий к исправлению.")

        if st.button("💾 Сохранить исправление", help="Сохранить ваши исправления для этого перехода"):
            df.loc[current_idx, 'reviewed_emotion_from'] = corrected_from
            df.loc[current_idx, 'reviewed_emotion_to'] = corrected_to
            df.loc[current_idx, 'comment'] = comment
            st.session_state.review_data = df
            st.session_state.show_correction = False
            st.success("Исправление сохранено!")
            # Go to next transition if not last
            if current_idx < len(df) - 1:
                st.session_state.current_transition += 1
            st.rerun()

    st.divider()

    # Navigation
    st.subheader("Навигация")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Предыдущий", disabled=(current_idx == 0), help="К предыдущему переходу"):
            st.session_state.current_transition = max(0, current_idx - 1)
            st.rerun()

    with col2:
        jump_to = st.number_input("Перейти к переходу:", min_value=1, max_value=len(df), value=current_idx + 1, help="Введите номер перехода") - 1
        if jump_to != current_idx:
            st.session_state.current_transition = jump_to
            st.rerun()

    with col3:
        if st.button("➡️ Следующий", disabled=(current_idx == len(df) - 1), help="К следующему переходу"):
            st.session_state.current_transition = min(len(df) - 1, current_idx + 1)
            st.rerun()

    st.divider()

    # Export final CSV
    st.subheader("Экспорт результатов")
    st.markdown("Скачайте проверенные переходы в формате CSV.")

    approved = len(df[df['review'] == 'approved'])
    rejected = len(df[df['review'] == 'rejected'])
    pending = len(df[df['review'] == 'pending'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Одобрено", approved)
    with col2:
        st.metric("❌ Отклонено", rejected)
    with col3:
        st.metric("⏳ В ожидании", pending)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Экспортировать итоговый CSV",
        data=csv_buffer.getvalue(),
        file_name="reviewed_emotion_transitions.csv",
        mime="text/csv",
        type="primary"
    )

    if st.button("← Назад к обработке", help="Вернуться к странице обработки видео"):
        st.session_state.page = 'processing'
        st.session_state.processed_csv = None
        st.session_state.review_data = None
        st.session_state.current_transition = 0
        st.rerun()

# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("Навигация")
    
    if st.sidebar.button("📹 Обработка видео", use_container_width=True):
        st.session_state.page = 'processing'
    
    if st.sidebar.button("✅ Проверка переходов", use_container_width=True):
        st.session_state.page = 'review'
    
    # Page routing
    if st.session_state.page == 'processing':
        video_processing_page()
    elif st.session_state.page == 'review':
        review_interface_page()

if __name__ == "__main__":
    main()