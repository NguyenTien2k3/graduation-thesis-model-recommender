import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import time

# ==========================================================================
# 1. Cấu hình & Hằng số
# ==========================================================================
# Đặt cấu hình log. Giữ nguyên mức INFO nhưng tối ưu hóa các log bên trong.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Các URL tải model và CSV
MODEL_HF_URL = "https://huggingface.co/Stas2k3/svd_model_nf32_lr0.001_reg0.05_reg0.05_ep40_p1.0_balanced/resolve/main/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
CSV_HF_URL = "https://huggingface.co/datasets/Stas2k3/Cell_Phones_and_Accessories_Train/resolve/main/Cell_Phones_and_Accessories.train.csv"

# Đường dẫn cache tạm
CACHE_DIR = "/tmp"
MODEL_PATH = os.path.join(CACHE_DIR, "model.pkl")
CSV_PATH = os.path.join(CACHE_DIR, "data.csv")

# ==========================================================================
# 2. Hàm tải file tối ưu RAM (stream) và tối ưu LOG (FIX)
# ==========================================================================


def download_file_stream(url, save_path, name, max_retries=5):
    """Tải file theo luồng (stream) để tránh tốn RAM và đã FIX lỗi log spam."""
    if os.path.exists(save_path):
        # Giảm log khi file đã tồn tại.
        logging.info(f"[{name}] File đã tồn tại trong cache: {save_path}")
        return True

    for attempt in range(max_retries):
        try:
            logging.info(f"[{name}] Tải từ {url} (attempt {attempt + 1})...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                downloaded = 0
                
                # Biến mới để kiểm soát log (Chỉ log khi vượt qua ngưỡng 20, 40, 60, 80)
                last_logged_step = 0 
                
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size:
                                percent = downloaded / total_size * 100
                                # Tính toán ngưỡng hiện tại (0, 20, 40, 60, 80)
                                current_step = int(percent // 20) * 20 

                                # Chỉ log khi vượt qua ngưỡng mới
                                if current_step > last_logged_step and current_step < 100:
                                    logging.info(f"[{name}] {current_step}% downloaded")
                                    last_logged_step = current_step
                                    
            # Log hoàn tất chỉ được gọi 1 lần khi thoát khỏi vòng lặp
            logging.info(f"[{name}] ✅ Hoàn tất tải: {save_path} (100%)")
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


def load_items():
    """Tải CSV bằng stream và đọc từ file."""
    if not download_file_stream(CSV_HF_URL, CSV_PATH, "CSV Data"):
        return []
    try:
        logging.info("Đọc CSV từ đĩa...")
        items_df = pd.read_csv(CSV_PATH)
        items_df["parent_asin"] = (
            items_df["parent_asin"].astype(str).str.split(",").str[0]
        )
        unique_items = items_df["parent_asin"].dropna().unique().tolist()
        logging.info(f"✅ CSV đã load: {len(unique_items)} items duy nhất.")
        return unique_items
    except Exception as e:
        logging.error(f"Lỗi đọc CSV: {e}")
        return []


def load_model():
    """Tải model pickle bằng stream."""
    if not download_file_stream(MODEL_HF_URL, MODEL_PATH, "Model"):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logging.info("✅ Model đã load thành công.")
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

# ==========================================================================
# 5. Hàm gợi ý
# ==========================================================================


def get_top_k_recommendations(user_id, item_ids, model, k=10, blocked_items=None):
    if model is None:
        return [{"error": "Model not loaded"}]
    if not item_ids:
        return [{"error": "No items available"}]

    blocked_set = set(blocked_items or [])
    valid_items = [iid for iid in item_ids if iid not in blocked_set]
    if not valid_items:
        return [{"error": "No valid items"}]

    predictions = []
    # Lưu ý: Không thêm log vào vòng lặp này vì nó chạy rất nhiều lần 
    # trong mỗi yêu cầu API, dễ gây ra log spam trở lại.
    for iid in valid_items:
        try:
            # Model.predict thường là hàm tính toán phức tạp
            pred = model.predict(uid=str(user_id), iid=str(iid)).est
            predictions.append((iid, pred))
        except Exception:
            continue

    if not predictions:
        return [{"error": "No predictions generated"}]

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [
        {"item_id": iid, "predicted_rating": round(r, 2)} for iid, r in predictions[:k]
    ]


# ==========================================================================
# 6. API Endpoint
# ==========================================================================


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    user_id = data.get("user_id")
    k = int(data.get("top_k", 10))
    blocked_items = data.get("blocked_items", ["B00K30H3O8"])
    
    # Có thể thêm log mức DEBUG ở đây nếu cần, nhưng tránh INFO/WARNING/ERROR
    # để không gây spam trong quá trình hoạt động bình thường.
    
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
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
                "items_count": len(item_ids),
            }
        ),
        200,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
