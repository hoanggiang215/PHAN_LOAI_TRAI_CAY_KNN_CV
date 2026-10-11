import streamlit as st
import cv2
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_SAVE_PATH = BASE_DIR / "knn_fruit_model.pkl"
METRICS_SAVE_PATH = BASE_DIR / "k_metrics.json"

st.set_page_config(
    page_title="Hệ Thống Phân Loại Trái Cây KNN",
    page_icon="🍌",
    layout="wide"
)


# Load feature extractor
@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))


# Load KNN model
@st.cache_resource
def load_knn_model():
    if not MODEL_SAVE_PATH.exists():
        return None
    return joblib.load(MODEL_SAVE_PATH)


def extract_features_single(img_pil, extractor):
    img_rgb = np.array(img_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. Color features
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
    s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
    v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
    color_features = [h_mean, h_std, s_mean, s_std, v_mean, v_std]

    # 2. Deep features
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    deep_features = extractor.predict(img_array, verbose=0).flatten()

    return np.hstack([color_features, deep_features])


# --- GIAO DIỆN STREAMLIT ---
st.title("🍊 🍌 🍚 NHẬN DIỆN TRÁI CÂY VỚI THUẬT TOÁN KNN")
st.markdown(
    "Chương trình nhận diện 3 loại trái cây: **Thanh Long**, **Cam**, và **Chuối** bằng phương pháp trích xuất đặc trưng đa khía cạnh (Màu sắc, Hình dáng, Kết cấu & Deep Feature Vector)[cite: 3].")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Tải ảnh lên hệ thống")
    uploaded_file = st.file_uploader("Chọn hình ảnh trái cây (JPG, PNG, WEBP)...", type=["jpg", "png", "jpeg", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hình ảnh đã tải lên", use_container_width=True)
        classify_btn = st.button("🔍 Phân loại Trái Cây", type="primary", use_container_width=True)
    else:
        classify_btn = False

with col2:
    st.subheader("2. Kết quả Phân loại")

    model_data = load_knn_model()
    extractor = load_feature_extractor()

    if classify_btn and uploaded_file is not None:
        if model_data is None:
            st.error("⚠️ Chưa tìm thấy file `knn_fruit_model.pkl`. Vui lòng chạy `python train_model.py` trước!")
        else:
            with st.spinner("Đang trích xuất đặc trưng và tính toán khoảng cách KNN..."):
                scaler = model_data["scaler"]
                knn_model = model_data["model"]
                class_mapping = model_data["class_mapping"]

                # Trích xuất và dự đoán
                feats = extract_features_single(image, extractor)
                feats_scaled = scaler.transform([feats])

                pred_label = knn_model.predict(feats_scaled)[0]
                pred_proba = knn_model.predict_proba(feats_scaled)[0]
                probs_percent = [round(p * 100, 2) for p in pred_proba]

                vietnamese_label = class_mapping.get(pred_label, pred_label)

                st.success(f"### 🎉 Kết quả nhận diện: **{vietnamese_label.upper()}**")

                # Hiển thị xác suất từng class
                df_prob = pd.DataFrame({
                    "Loại quả": [class_mapping[c] for c in knn_model.classes_],
                    "Độ tin cậy (%)": probs_percent
                })
                st.dataframe(df_prob, use_container_width=True)

st.markdown("---")
st.subheader("📊 Bảng đánh giá độ chính xác theo các giá trị K (K=3, 4, 5, 6, 7, 8, 9, 10)")

if METRICS_SAVE_PATH.exists():
    with open(METRICS_SAVE_PATH, "r", encoding="utf-8") as f:
        k_metrics = json.load(f)

    df_k = pd.DataFrame(list(k_metrics.items()), columns=["Giá trị K", "Độ chính xác Accuracy (%)"])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("**Bảng số liệu chi tiết:**")
        st.table(df_k)
    with c2:
        st.write("**Biểu đồ biến thiên Accuracy theo K:**")
        st.line_chart(df_k.set_index("Giá trị K"))
else:
    st.info("💡 Chạy `python train_model.py` để tạo dữ liệu bảng so sánh các giá trị K.")