import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import time
import threading
import gc # Import Garbage Collector để quản lý RAM

# ==========================================================================
# 1. Cấu hình & Hằng số
# ==========================================================================
# Thiết lập cấu hình log cơ bản
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# CÁC URL VÀ PATH CỦA FILE DỮ LIỆU
MODEL_HF_URL = "https://huggingface.co/Stas2k3/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced/resolve/main/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
CSV_HF_URL = "https://huggingface.co/datasets/Stas2k3/Cell_Phones_and_Accessories_Train/resolve/main/Cell_Phones_and_Accessories.train.csv"

# URL DỮ LIỆU ĐÃ TÍNH TOÁN TRƯỚC (RẤT QUAN TRỌNG CHO TỐC ĐỘ API)
# BẠN CẦN THAY THẾ BẰNG URL DẪN ĐẾN FILE PICKLE CHỈ CHỨA TOP N GỢI Ý CHO MỖI USER
PRECOMPUTED_HF_URL = "YOUR_PRECOMPUTED_TOP_K_URL_HERE" 

# Đường dẫn cache tạm
CACHE_DIR = "/tmp"
MODEL_PATH = os.path.join(CACHE_DIR, "model.pkl")
CSV_PATH = os.path.join(CACHE_DIR, "data.csv")
ITEM_IDS_PATH = os.path.join(CACHE_DIR, "item_ids.pkl")
PRECOMPUTED_PATH = os.path.join(CACHE_DIR, "precomputed_recs.pkl")

# Khởi tạo dữ liệu toàn cục (sẽ được cập nhật bởi luồng nền)
item_ids = []
topk_model = None
precomputed_recommendations = {} # Dữ liệu gợi ý đã tính toán trước

# ==========================================================================
# 2. Hàm tải file tối ưu RAM (stream)
# ==========================================================================

def download_file_stream(url, save_path, name, max_retries=5):
    """Tải file theo luồng (stream) để tránh tốn RAM và giảm tần suất ghi log."""
    if os.path.exists(save_path):
        logging.info(f"[{name}] File đã tồn tại trong cache: {save_path}")
        return True
    
    # Bỏ qua nếu URL là placeholder
    if url == "YOUR_PRECOMPUTED_TOP_K_URL_HERE":
        logging.warning(f"[{name}] URL chưa được cấu hình. Bỏ qua tải.")
        return False

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
# 3. Load dữ liệu (Tối ưu RAM)
# ==========================================================================

def _load_items_blocking():
    """Tải và load danh sách item ID (Ưu tiên load từ cache nhanh ITEM_IDS_PATH)."""
    if os.path.exists(ITEM_IDS_PATH):
        try:
            logging.info("Đang load Item IDs từ cache nhanh...")
            with open(ITEM_IDS_PATH, 'rb') as f:
                unique_items = pickle.load(f)
            logging.info(f"✅ Item IDs đã load từ cache: {len(unique_items)} items duy nhất.")
            return unique_items
        except Exception as e:
            logging.warning(f"Lỗi đọc cache Item IDs: {e}. Sẽ load lại từ CSV.")

    if not download_file_stream(CSV_HF_URL, CSV_PATH, "CSV Data"):
        return []
    
    try:
        logging.info("Đọc CSV TỪ ĐĨA (quá trình chậm)...")
        # CHỈ TẢI CÁC CỘT CẦN THIẾT (parent_asin) ĐỂ GIẢM RAM
        items_df = pd.read_csv(CSV_PATH, usecols=["parent_asin"])
        
        # Xử lý cột parent_asin
        items_df["parent_asin"] = (
            items_df["parent_asin"].astype(str).str.split(",").str[0]
        )
        unique_items = items_df["parent_asin"].dropna().unique().tolist()
        logging.info(f"✅ CSV đã load và xử lý: {len(unique_items)} items duy nhất.")

        # DỌN DẸP RAM SAU KHI SỬ DỤNG DATAFRAME LỚN
        del items_df
        gc.collect() 
        logging.info("Bộ nhớ đã được dọn dẹp sau khi xử lý CSV.")

        # Lưu kết quả xử lý vào cache nhanh ITEM_IDS_PATH
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
    """Tải model pickle (SVD) bằng stream (blocking operation)."""
    if not download_file_stream(MODEL_HF_URL, MODEL_PATH, "Model"):
        return None
    try:
        logging.info("Đang load Model từ đĩa (quá trình chậm và tốn RAM)...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logging.info("✅ Model đã load thành công.")
        return model
    except Exception as e:
        logging.error(f"Lỗi load model: {e}")
        return None

def _load_precomputed_data():
    """Tải và load dữ liệu gợi ý đã tính toán trước (TỐC ĐỘ CAO)."""
    if not download_file_stream(PRECOMPUTED_HF_URL, PRECOMPUTED_PATH, "Precomputed Data"):
        return {}
    
    try:
        logging.info("Đang load Dữ liệu gợi ý đã tính toán trước...")
        with open(PRECOMPUTED_PATH, "rb") as f:
            data = pickle.load(f)
        logging.info(f"✅ Dữ liệu Pre-computed đã load thành công. (Số lượng users: {len(data)})")
        return data
    except Exception as e:
        logging.error(f"Lỗi load dữ liệu Pre-computed: {e}")
        return {}

# ==========================================================================
# 4. Khởi động dữ liệu bất đồng bộ (ASYNC)
# ==========================================================================

def load_data_and_model_async():
    """Hàm mục tiêu cho luồng nền, chịu trách nhiệm load dữ liệu và model nặng."""
    global item_ids, topk_model, precomputed_recommendations
    logging.info("🚀 Luồng nền: Bắt đầu tải Item IDs, Model và Dữ liệu Pre-computed...")
    
    # 1. Tải Item IDs (Sử dụng cache nhanh)
    item_ids = _load_items_blocking()
    
    # 2. Tải Model (Phần chậm, tốn RAM)
    topk_model = _load_model_blocking()
    
    # 3. Tải Dữ liệu Pre-computed (Quan trọng cho tốc độ gợi ý)
    precomputed_recommendations = _load_precomputed_data()
    
    logging.info("✅ Luồng nền: Hoàn tất tất cả quá trình load dữ liệu.")


# ==========================================================================
# 5. Hàm gợi ý (Sử dụng Cache)
# ==========================================================================

def get_top_k_recommendations(user_id, k=10, blocked_items=None):
    """
    Hàm gợi ý hiệu suất cao: Chỉ tra cứu trong dữ liệu đã tính toán trước.
    Loại bỏ vòng lặp 95,000 phép tính chậm chạp.
    """
    # 1. Kiểm tra Dữ liệu Pre-computed
    if not precomputed_recommendations:
        # Nếu chưa load được Pre-compute, fallback về Model SVD (CHẬM)
        if topk_model:
            logging.warning("Sử dụng fallback gợi ý SVD (RẤT CHẬM).")
            # Nếu người dùng đã cung cấp URL Pre-computed, nhưng nó lỗi, 
            # chúng ta không nên chạy 95k phép tính ở đây. 
            # Giả sử chúng ta CHỈ dựa vào Pre-computed.
            return [{"item_id": "fallback_error", "predicted_rating": 0.0, "note": "Precomputed data missing, cannot suggest."}]
        else:
            return [{"error": "Model and Precomputed data not loaded"}]

    user_str = str(user_id)
    
    # 2. Tra cứu trong Cache Pre-computed
    if user_str not in precomputed_recommendations:
        logging.warning(f"User {user_id} không có trong cache Pre-computed.")
        # Nếu user mới, có thể fallback về gợi ý Top Phổ Biến (chưa được cài đặt)
        return [{"error": "User not found in precomputed cache"}]

    # Lấy danh sách gợi ý đã sắp xếp cho user
    all_recs = precomputed_recommendations[user_str]
    
    # 3. Áp dụng Blocked Items và giới hạn Top K
    blocked_set = set(blocked_items or [])
    final_recs = []
    
    for rec in all_recs:
        if rec['item_id'] not in blocked_set:
            final_recs.append(rec)
            if len(final_recs) >= k:
                break
                
    return final_recs


# ==========================================================================
# 6. API Endpoint
# ==========================================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    # Kiểm tra trạng thái dữ liệu Pre-computed trước khi xử lý request
    if not precomputed_recommendations:
        logging.warning("Yêu cầu gợi ý thất bại: Dữ liệu Pre-computed đang được tải.")
        # Trả về lỗi 503 (Service Unavailable) nếu dữ liệu gợi ý tốc độ cao chưa load xong
        return jsonify({"error": "Precomputed recommendations are still loading (Model loading). Please wait a few seconds."}), 503
    
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {e}"}), 400

    user_id = data.get("user_id")
    k = int(data.get("top_k", 10))
    blocked_items = data.get("blocked_items", [])

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    start_time = time.time()
    recommendations = get_top_k_recommendations(user_id, k, blocked_items)
    end_time = time.time()

    logging.info(f"✅ Gợi ý cho user {user_id} hoàn tất trong {(end_time - start_time) * 1000:.2f}ms")

    # Thêm kiểm tra lỗi cuối cùng
    if recommendations and "error" in recommendations[0]:
        return jsonify({"error": recommendations[0]["error"]}), 500

    return jsonify(recommendations), 200

@app.route("/health", methods=["GET"])
def health():
    # Health check phản ánh trạng thái của model và dữ liệu gợi ý (tức là đã load xong chưa)
    return jsonify({
        "status": "healthy",
        # Trả về false trong quá trình tải model và dữ liệu pre-computed
        "model_loaded": topk_model is not None, 
        "precomputed_loaded": bool(precomputed_recommendations),
        "items_count": len(item_ids)
    }), 200

# ==========================================================================
# 7. Khởi động Gunicorn/WSGI (Thread Initialization)
# ==========================================================================
# Bắt đầu luồng nền để tải dữ liệu và model.
# Đoạn code này chạy khi module được import bởi Gunicorn/WSGI worker.
logging.info("🚀 Bắt đầu luồng nền tải dữ liệu...")
threading.Thread(target=load_data_and_model_async, daemon=True).start()

# Hàm main chỉ dành cho phát triển cục bộ (local development)
if __name__ == "__main__":
    logging.info("✅ Chạy chế độ phát triển cục bộ (local development)...")
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
