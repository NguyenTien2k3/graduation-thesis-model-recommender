import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import time
import threading # <-- Import mới để chạy nền

# ==========================================================================
# 1. Cấu hình & Hằng số
# ==========================================================================
# Thiết lập cấu hình log cơ bản
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Các URL tải model và CSV
MODEL_HF_URL = "https://huggingface.co/Stas2k3/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced/resolve/main/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
CSV_HF_URL = "https://huggingface.co/datasets/Stas2k3/Cell_Phones_and_Accessories_Train/resolve/main/Cell_Phones_and_Accessories.train.csv"

# Đường dẫn cache tạm
CACHE_DIR = "/tmp"
MODEL_PATH = os.path.join(CACHE_DIR, "model.pkl")
CSV_PATH = os.path.join(CACHE_DIR, "data.csv")
# Đường dẫn cache mới cho danh sách Item ID đã xử lý (TỐC ĐỘ CAO)
ITEM_IDS_PATH = os.path.join(CACHE_DIR, "item_ids.pkl")

# Khởi tạo dữ liệu toàn cục (sẽ được cập nhật sau)
item_ids = []
topk_model = None

# ==========================================================================
# 2. Hàm tải file tối ưu RAM (stream)
# ==========================================================================

def download_file_stream(url, save_path, name, max_retries=5):
    """Tải file theo luồng (stream) để tránh tốn RAM và giảm tần suất ghi log."""
    if os.path.exists(save_path):
        logging.info(f"[{name}] File đã tồn tại trong cache: {save_path}")
        return True

    for attempt in range(max_retries):
        try:
            logging.info(f"[{name}] Tải từ {url} (attempt {attempt + 1})...")
            last_reported_percent = -1
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                percent = downloaded / total_size * 100
                                # Chỉ ghi log khi tiến độ vượt qua ngưỡng 20% mới
                                current_major_percent = int(percent // 20) * 20
                                if current_major_percent > last_reported_percent and current_major_percent < 100:
                                    logging.info(f"[{name}] {current_major_percent}% downloaded")
                                    last_reported_percent = current_major_percent

            logging.info(f"[{name}] ✅ Hoàn tất tải: {save_path}")
            return True
        except Exception as e:
            logging.warning(f"[{name}] Lỗi khi tải: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logging.info(f"[{name}] Đợi {wait}s trước khi retry...")
                time.sleep(wait)
    logging.error(f"[{name}] ❌ Tải thất bại sau {max_retries} lần.")
    return False

# ==========================================================================
# 3. Load dữ liệu
# ==========================================================================

def _load_items_blocking():
    """Tải và load danh sách item ID (Sẽ chạy nhanh nếu cache tồn tại)."""
    # 1. Thử load từ cache nhanh ITEM_IDS_PATH
    if os.path.exists(ITEM_IDS_PATH):
        try:
            logging.info("Đang load Item IDs từ cache nhanh...")
            with open(ITEM_IDS_PATH, 'rb') as f:
                unique_items = pickle.load(f)
            logging.info(f"✅ Item IDs đã load từ cache: {len(unique_items)} items duy nhất.")
            return unique_items
        except Exception as e:
            logging.warning(f"Lỗi đọc cache Item IDs: {e}. Sẽ load lại từ CSV.")

    # 2. Nếu cache nhanh không tồn tại hoặc lỗi, fallback về CSV (quá trình chậm)
    if not download_file_stream(CSV_HF_URL, CSV_PATH, "CSV Data"):
        return []
    
    try:
        logging.info("Đọc CSV TỪ ĐĨA (quá trình chậm)...")
        items_df = pd.read_csv(CSV_PATH)
        # Xử lý cột parent_asin để chỉ lấy ASIN đầu tiên nếu là danh sách
        items_df["parent_asin"] = (
            items_df["parent_asin"].astype(str).str.split(",").str[0]
        )
        unique_items = items_df["parent_asin"].dropna().unique().tolist()
        logging.info(f"✅ CSV đã load và xử lý: {len(unique_items)} items duy nhất.")

        # 3. Lưu kết quả xử lý vào cache nhanh ITEM_IDS_PATH
        try:
            with open(ITEM_IDS_PATH, 'wb') as f:
                pickle.dump(unique_items, f)
            logging.info("✅ Đã tạo cache nhanh Item IDs.")
        except Exception as e:
            logging.warning(f"Lỗi khi lưu cache Item IDs: {e}")

        return unique_items
    except Exception as e:
        logging.error(f"Lỗi đọc CSV: {e}")
        return []

def _load_model_blocking():
    """Tải model pickle bằng stream (blocking operation)."""
    if not download_file_stream(MODEL_HF_URL, MODEL_PATH, "Model"):
        return None
    try:
        logging.info("Đang load Model từ đĩa (quá trình chậm)...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logging.info("✅ Model đã load thành công.")
        return model
    except Exception as e:
        logging.error(f"Lỗi load model: {e}")
        return None

# ==========================================================================
# 4. Khởi động dữ liệu bất đồng bộ (ASYNC)
# ==========================================================================

def load_data_and_model_async():
    """Hàm mục tiêu cho luồng nền, chịu trách nhiệm load dữ liệu và model."""
    global item_ids, topk_model
    logging.info("🚀 Luồng nền: Bắt đầu tải Item IDs (nhanh) và Model (chậm)...")
    
    # 1. Tải Item IDs (Vẫn giữ synchronous vì nó đã nhanh)
    item_ids = _load_items_blocking()
    
    # 2. Tải Model (Phần chậm, nhưng đang chạy ở luồng nền)
    topk_model = _load_model_blocking()
    
    logging.info("✅ Luồng nền: Hoàn tất tất cả quá trình load dữ liệu.")


# ==========================================================================
# 5. Hàm gợi ý
# ==========================================================================

def get_top_k_recommendations(user_id, current_item_ids, model, k=10, blocked_items=None):
    # Kiểm tra model trước khi chạy
    if model is None:
        logging.error("Model chưa được load. Không thể gợi ý.")
        return [{"error": "Model not loaded"}]
    
    if not current_item_ids:
        logging.error("Không có danh sách items. Không thể gợi ý.")
        return [{"error": "No items available"}]

    blocked_set = set(blocked_items or [])
    valid_items = [iid for iid in current_item_ids if iid not in blocked_set]
    if not valid_items:
        logging.warning("Không còn items hợp lệ sau khi loại bỏ blocked_items.")
        return [{"error": "No valid items"}]

    predictions = []
    # Lưu ý: Việc lặp qua TẤT CẢ items (94k+) và gọi predict là rất tốn thời gian.
    # Đây vẫn là điểm cần tối ưu hóa hiệu suất sau khi giải quyết vấn đề khởi động.
    start_time = time.time()
    for iid in valid_items:
        try:
            # Model.predict() ước tính rating cho cặp (user, item)
            # Sử dụng global topk_model được cập nhật từ luồng nền
            pred = model.predict(uid=str(user_id), iid=str(iid)).est
            predictions.append((iid, pred))
        except Exception:
            # Bỏ qua nếu item hoặc user ID không có trong dữ liệu huấn luyện
            continue

    end_time = time.time()
    logging.info(f"Đã tạo {len(predictions)} dự đoán trong {end_time - start_time:.2f}s.")

    if not predictions:
        return [{"error": "No predictions generated"}]

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [
        {"item_id": iid, "predicted_rating": round(r, 2)}
        for iid, r in predictions[:k]
    ]

# ==========================================================================
# 6. API Endpoint
# ==========================================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    # Kiểm tra trạng thái model trước khi xử lý request
    if topk_model is None:
        logging.warning("Yêu cầu gợi ý thất bại: Model đang được tải.")
        # Trả về lỗi 503 (Service Unavailable) nếu model chưa load xong
        return jsonify({"error": "Model is still loading. Please try again in a few seconds."}), 503
    
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {e}"}), 400

    user_id = data.get("user_id")
    k = int(data.get("top_k", 10))
    blocked_items = data.get("blocked_items", [])

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    logging.info(f"Yêu cầu gợi ý cho user_id: {user_id}, top_k: {k}")

    # Truyền biến item_ids và topk_model đã được cập nhật bởi luồng nền
    recommendations = get_top_k_recommendations(user_id, item_ids, topk_model, k, blocked_items)

    # Thêm kiểm tra lỗi cuối cùng
    if recommendations and "error" in recommendations[0]:
        return jsonify({"error": recommendations[0]["error"]}), 500

    return jsonify(recommendations), 200

@app.route("/health", methods=["GET"])
def health():
    # Health check phản ánh trạng thái của model
    return jsonify({
        "status": "healthy",
        # Trả về false trong 8-9 giây đầu
        "model_loaded": topk_model is not None, 
        "items_count": len(item_ids)
    }), 200

if __name__ == "__main__":
    # Bắt đầu luồng nền để tải dữ liệu và model
    threading.Thread(target=load_data_and_model_async, daemon=True).start()
    
    # Server Flask bắt đầu chạy ngay lập tức
    logging.info("✅ Server Flask bắt đầu lắng nghe cổng (Startup Time < 1s)...")
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
