import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import time
import sys # Thêm sys để kiểm tra kích thước đối tượng

# ==========================================================================
# 1. Cấu hình & Hằng số
# ==========================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Các URL tải
MODEL_HF_URL = "https://huggingface.co/Stas2k3/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced/resolve/main/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
# 🚨 URL MỚI: Thay thế CSV bằng file pickle chứa list item IDs đã được tiền xử lý
ITEM_IDS_HF_URL = "https://huggingface.co/datasets/Stas2k3/Cell_Phones_and_Accessories_Train/resolve/main/item_ids.pkl" 
# Bạn cần thay URL này bằng URL của file item_ids.pkl mà bạn đã upload

# Đường dẫn cache tạm
CACHE_DIR = "/tmp"
MODEL_PATH = os.path.join(CACHE_DIR, "model.pkl")
ITEM_IDS_PATH = os.path.join(CACHE_DIR, "item_ids.pkl")

# ==========================================================================
# 2. Hàm tải file tối ưu RAM (stream)
# ==========================================================================


def download_file_stream(url, save_path, name, max_retries=5):
    """Tải file theo luồng (stream) để tránh tốn RAM."""
    if os.path.exists(save_path):
        logging.info(f"[{name}] File đã tồn tại trong cache: {save_path}")
        return True

    for attempt in range(max_retries):
        try:
            logging.info(f"[{name}] Tải từ {url} (attempt {attempt + 1})...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                # Thêm log kiểm tra kích thước file để theo dõi
                total_size_bytes = int(r.headers.get("content-length", 0))
                total_size_mb = total_size_bytes / (1024 * 1024)
                logging.info(f"[{name}] Kích thước file: {total_size_mb:.2f} MB")
                
                downloaded = 0
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Bỏ log % tải xuống để giảm I/O và tăng tốc
                            
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
# 3. Load dữ liệu (Đã tối ưu)
# ==========================================================================


def load_items():
    """Tải và đọc list Item ID từ file pickle siêu nhỏ gọn."""
    if not download_file_stream(ITEM_IDS_HF_URL, ITEM_IDS_PATH, "Item IDs"):
        return []
    try:
        logging.info("Đọc Item IDs từ đĩa...")
        with open(ITEM_IDS_PATH, "rb") as f:
            item_ids = pickle.load(f)
        
        # Thêm log kiểm tra RAM sử dụng
        size_mb = sys.getsizeof(item_ids) / (1024 * 1024)
        logging.info(f"✅ Item IDs đã load: {len(item_ids)} items duy nhất. RAM: {size_mb:.2f} MB")
        return item_ids
    except Exception as e:
        logging.error(f"Lỗi đọc Item IDs: {e}")
        return []


def load_model():
    """Tải model pickle bằng stream."""
    if not download_file_stream(MODEL_HF_URL, MODEL_PATH, "Model"):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        # Thêm log kiểm tra RAM sử dụng
        size_mb = sys.getsizeof(model) / (1024 * 1024)
        logging.info(f"✅ Model đã load thành công. RAM: {size_mb:.2f} MB")
        return model
    except Exception as e:
        logging.error(f"Lỗi load model: {e}")
        return None


# ==========================================================================
# 4. Khởi động dữ liệu toàn cục
# ==========================================================================

logging.info("🚀 Khởi động server — bắt đầu load dữ liệu...")
item_ids = load_items()
topk_model = load_model()

# Kiểm tra sau khi load
if topk_model is None or not item_ids:
    logging.error("🚨 Không thể load Model hoặc Item IDs. Dịch vụ sẽ bị lỗi.")

# ==========================================================================
# 5. Hàm gợi ý
# ==========================================================================


def get_top_k_recommendations(user_id, item_ids, model, k=10, blocked_items=None):
    if model is None:
        return [{"error": "Model not loaded"}]
    if not item_ids:
        return [{"error": "No items available"}]

    blocked_set = set(blocked_items or [])
    # Chỉ giữ lại các ASIN có ít nhất một ký tự và không nằm trong blocked_set
    valid_items = [iid for iid in item_ids if iid and iid not in blocked_set]
    
    if not valid_items:
        return [{"error": "No valid items"}]
    
    # Giới hạn số lượng dự đoán để tránh timeout/OOM nếu list item_ids quá lớn
    # Nếu list ID quá lớn (ví dụ > 50k), bạn nên lấy mẫu ngẫu nhiên (sampling)
    # hoặc dùng các hàm tối ưu hơn của Surprise (get_top_n)
    
    predictions = []
    
    # Sử dụng mô hình (Surprise SVD) để dự đoán
    for iid in valid_items:
        try:
            # model.predict yêu cầu uid và iid là chuỗi (str)
            pred = model.predict(uid=str(user_id), iid=str(iid)).est
            predictions.append((iid, pred))
        except Exception:
            # Bỏ qua nếu có lỗi (ví dụ: Item ID hoặc User ID không có trong mô hình)
            continue

    if not predictions:
        # Nếu mô hình không tạo ra dự đoán nào cho user này
        # Có thể dùng một chiến lược dự phòng (fallback) ở đây
        return [{"error": "No predictions generated or user unknown"}]

    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return [
        {"item_id": iid, "predicted_rating": round(r, 2)} for iid, r in predictions[:k]
    ]


# ==========================================================================
# 6. API Endpoint
# ==========================================================================


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {e}"}), 400

    user_id = data.get("user_id")
    k = int(data.get("top_k", 10))
    # Sử dụng giá trị mặc định ít gây tranh cãi hơn, hoặc để rỗng
    blocked_items = data.get("blocked_items", []) 
    
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    if topk_model is None or not item_ids:
        return jsonify({"error": "Service not ready: Model or Items not loaded"}), 503

    recommendations = get_top_k_recommendations(
        user_id, item_ids, topk_model, k, blocked_items
    )
    return jsonify(recommendations), 200


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify(
            {
                "status": "healthy",
                "model_loaded": topk_model is not None,
                "items_count": len(item_ids) if item_ids else 0,
            }
        ),
        200,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Web server starting on port {port}")
    app.run(host="0.0.0.0", port=port)