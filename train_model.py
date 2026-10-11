import os
import json
import joblib
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Tắt bớt log dư thừa của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_SAVE_PATH = BASE_DIR / "knn_fruit_model.pkl"
METRICS_SAVE_PATH = BASE_DIR / "k_metrics.json"

TARGET_CLASSES = {
    "Banana": "Chuối",
    "Orange": "Cam",
    "Pitaya": "Thanh Long"
}
SAMPLES_PER_CLASS = 2500
BATCH_SIZE = 64  # Xử lý 64 ảnh/lần giúp GPU/CPU tính toán cực nhanh
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("🔄 Đang nạp mô hình MobileNetV2...")
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))


def process_single_image_color(img_path):
    """Đọc ảnh và trích xuất đặc trưng màu sắc HSV"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None

    # 1. Trích xuất màu sắc (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
    s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
    v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
    color_feats = [h_mean, h_std, s_mean, s_std, v_mean, v_std]

    # 2. Resize chuẩn bị cho MobileNetV2
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    return color_feats, img_resized


def process_class_in_batches(class_dir, samples_limit):
    """Xử lý song song và gom Batch trích xuất đặc trưng"""
    images = [f for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS][:samples_limit]

    # Đọc ảnh đa luồng bằng ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_image_color, images))

    color_list = []
    img_resized_list = []

    for c_feat, img_res in results:
        if c_feat is not None:
            color_list.append(c_feat)
            img_resized_list.append(img_res)

    color_arr = np.array(color_list)
    img_resized_arr = np.array(img_resized_list, dtype=np.float32)
    img_preprocessed = preprocess_input(img_resized_arr)

    # Trích xuất đặc trưng bằng MobileNetV2 theo BATCH
    print(f"   ⚡ Đang trích xuất đặc trưng Deep Feature cho {len(images)} ảnh (Batch size: {BATCH_SIZE})...")
    deep_feats = feature_extractor.predict(img_preprocessed, batch_size=BATCH_SIZE, verbose=1)

    # Gộp đặc trưng Màu sắc + Deep Feature
    combined_feats = np.hstack([color_arr, deep_feats])
    return combined_feats


def main():
    X_all = []
    y_all = []

    print("\n📂 ĐANG NẠP VÀ TRÍCH XUẤT ĐẶC TRƯNG DỮ LIỆU...")
    for class_name in TARGET_CLASSES.keys():
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"❌ Không tìm thấy thư mục: {class_dir}")
            continue

        print(f"\n📦 Đang xử lý class '{class_name}' ({TARGET_CLASSES[class_name]})...")
        feats = process_class_in_batches(class_dir, SAMPLES_PER_CLASS)

        X_all.append(feats)
        y_all.extend([class_name] * len(feats))

    X = np.vstack(X_all)
    y = np.array(y_all)

    print(f"\n✅ Hoàn tất! Tổng số mẫu: {X.shape[0]} | Kích thước Vector đặc trưng: {X.shape[1]}")

    # Chia tập train/test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện KNN đa tiến trình (n_jobs=-1 để dùng full CPU)
    k_values = list(range(3, 11))
    k_accuracy_results = {}
    best_k = 3
    best_acc = 0.0
    best_knn_model = None

    print("\n🔍 ĐANG HUẤN LUYỆN VÀ SO SÁNH CÁC GIÁ TRỊ K (K=3,4,5,6,7,8,9,10)...")
    for k in k_values:
        # n_jobs=-1 giúp KNN chạy song song trên tất cả lõi CPU
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean', n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        k_accuracy_results[str(k)] = round(float(acc * 100), 2)
        print(f"   ➡️ K = {k:2d} | Độ chính xác (Accuracy): {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_knn_model = knn

    print(f"\n🏆 Mô hình KNN tối ưu nhất tại K = {best_k} với Accuracy: {best_acc * 100:.2f}%")

    # Lưu Model & Scaler
    model_data = {
        "scaler": scaler,
        "model": best_knn_model,
        "best_k": best_k,
        "class_mapping": TARGET_CLASSES
    }
    joblib.dump(model_data, MODEL_SAVE_PATH)

    # Lưu file JSON kết quả K
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(k_accuracy_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 Đã hoàn tất và lưu tại:\n - {MODEL_SAVE_PATH}\n - {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()