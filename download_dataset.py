import os
import shutil
from pathlib import Path

# Store KaggleHub cache INSIDE the project, not in C:\Users\Admin\.cache.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_CACHE = BASE_DIR / ".kagglehub_cache"
DATASET_DIR = BASE_DIR / "dataset"

os.environ["KAGGLEHUB_CACHE"] = str(PROJECT_CACHE)

DATASET = "chrisfilo/fruit-recognition"
EXPECTED_CLASSES = {
    "Apple", "Banana", "Carambola", "Guava", "Kiwi", "Mango",
    "Orange", "Peach", "Pear", "Persimmon", "Pitaya", "Plum",
    "Pomegranate", "Tomatoes", "muskmelon"
}


def find_dataset_root(root: Path):
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    best = None
    best_score = 0
    for p in candidates:
        try:
            children = {x.name for x in p.iterdir() if x.is_dir()}
        except OSError:
            continue
        score = len(EXPECTED_CLASSES.intersection(children))
        if score > best_score:
            best_score = score
            best = p
        if score == 15:
            return p
    return best if best_score >= 10 else None


def main():
    print("=" * 65)
    print("🍎 DOWNLOAD FRUIT DATASET - CHỈ CHẠY KHI CẦN")
    print("=" * 65)

    existing = find_dataset_root(DATASET_DIR) if DATASET_DIR.exists() else None
    if existing:
        print(f"✅ Dataset đã có sẵn: {existing}")
        print("Không tải lại. Bạn có thể chạy train_model.py")
        return

    PROJECT_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"📁 Cache: {PROJECT_CACHE}")
    print("📦 Dataset: chrisfilo/fruit-recognition")
    print("⚠️ Dataset khoảng 8.5 GB. Chỉ tải một lần.")
    print("⚠️ Hãy đảm bảo ổ đĩa còn đủ dung lượng.")

    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "Chưa có kagglehub. Chạy:\n"
            "python -m pip install kagglehub"
        )

    print("\n⬇️ Bắt đầu tải. Không đóng cửa sổ này giữa chừng...\n")
    downloaded = Path(kagglehub.dataset_download(DATASET))
    print(f"\n✅ KaggleHub đã tải xong: {downloaded}")

    data_root = find_dataset_root(downloaded)
    if data_root is None:
        raise RuntimeError(
            "Tải xong nhưng không tìm thấy 15 class. "
            f"Hãy kiểm tra: {downloaded}"
        )

    print(f"📂 Dataset root: {data_root}")
    print(f"📁 Đang chuẩn bị thư mục local: {DATASET_DIR}")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Copy class folders to project/dataset. shutil.copytree keeps the original cache intact.
    copied = 0
    for class_dir in sorted(data_root.iterdir(), key=lambda p: p.name.lower()):
        if not class_dir.is_dir():
            continue
        destination = DATASET_DIR / class_dir.name
        if destination.exists():
            print(f"⏭️ Bỏ qua class đã tồn tại: {class_dir.name}")
            continue
        print(f"📂 Copy: {class_dir.name}")
        shutil.copytree(class_dir, destination)
        copied += 1

    final_root = find_dataset_root(DATASET_DIR)
    if final_root is None:
        raise RuntimeError("Copy dataset chưa hoàn tất hoặc cấu trúc không đúng.")

    print("\n" + "=" * 65)
    print("✅ DATASET ĐÃ SẴN SÀNG")
    print(f"📂 {final_root}")
    print("➡️ Tiếp theo chạy: python train_model.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
