import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import io  # Để đọc CSV từ bytes

# ==============================================================================
# 1. Cấu hình & Hằng số
# ==============================================================================

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

# Các URL Hugging Face (Sử dụng link trực tiếp)
MODEL_HF_URL = "https://huggingface.co/Stas2k3/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced/resolve/main/svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
CSV_HF_URL = "https://huggingface.co/datasets/Stas2k3/Cell_Phones_and_Accessories_Train/resolve/main/Cell_Phones_and_Accessories.train.csv"


# ==============================================================================
# 2. Hàm load an toàn (chỉ tải 1 lần)
# ==============================================================================

def safe_load_pickle(url, name):
    """Tải mô hình pickle từ URL. Sẽ bị lỗi nếu thời gian tải quá lâu (timeout)."""
    logging.info(f"Attempting to load {name} model from URL: {url}")
    try:
        # Tăng timeout để phù hợp với việc tải file lớn
        response = requests.get(url, timeout=300) 
        response.raise_for_status()  # Báo lỗi nếu mã trạng thái là 4xx hoặc 5xx
        return pickle.loads(response.content)
    except requests.exceptions.Timeout:
        logging.error(f"Error loading '{url}' for {name}: Request timed out (300s).")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error loading '{url}' for {name}: HTTP Request failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading '{url}' for {name}: Deserialization failed: {e}")
        return None


def safe_load_csv(url):
    """Tải và đọc dữ liệu CSV từ URL."""
    logging.info(f"Attempting to load CSV data from URL: {url}")
    try:
        response = requests.get(url, timeout=300) 
        response.raise_for_status()
        # Sử dụng io.BytesIO để đọc dữ liệu CSV từ bytes trong bộ nhớ
        return pd.read_csv(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        logging.error(f"Error loading CSV from '{url}': HTTP Request failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing CSV data: {e}")
        return None

def load_items():
    """Tải CSV, xử lý và trích xuất danh sách item ID duy nhất."""
    items_df = safe_load_csv(CSV_HF_URL)
    if items_df is None:
        return []
    try:
        # Xử lý cột 'parent_asin': chuyển sang string và lấy ID đầu tiên
        items_df["parent_asin"] = items_df["parent_asin"].astype(str).str.split(",").str[0]
        unique_items = items_df["parent_asin"].dropna().unique().tolist()
        logging.info(f"Successfully loaded and processed {len(unique_items)} unique items")
        return unique_items
    except Exception as e:
        logging.error(f"Error processing CSV data columns: {e}.")
        return []


# ==============================================================================
# 3. Load Model và Item Data VÀO BỘ NHỚ (Chỉ chạy 1 lần khi app khởi động)
# ==============================================================================

topk_model = safe_load_pickle(MODEL_HF_URL, "Top-K SVD model")
item_ids = load_items()


# ==============================================================================
# 4. Logic Gợi ý
# ==============================================================================

def get_top_k_recommendations(user_id, item_ids, model, k=10, blocked_items=None):
    """
    Tạo gợi ý Top-K cho một người dùng bằng cách dự đoán rating cho các item chưa thấy.
    """
    if model is None:
        return [{"error": "Model not loaded"}]
    if not item_ids:
        return [{"error": "No items available"}]

    blocked_set = set(blocked_items or [])
    # Lọc bỏ các item đã bị chặn
    valid_items = [iid for iid in item_ids if iid not in blocked_set]

    if not valid_items:
        return [{"error": "No valid items after filtering blocked items"}]

    predictions = []
    seen_items = set()
    
    for iid in valid_items:
        if iid in seen_items:
            continue
        try:
            # Dự đoán rating cho cặp (user, item)
            pred = model.predict(uid=str(user_id), iid=str(iid)).est
            predictions.append((iid, pred))
            seen_items.add(iid)
        except Exception as e:
            # Bỏ qua nếu mô hình không thể dự đoán cho item này (ví dụ: item/user mới)
            logging.warning(f"Skipping prediction for item {iid} for user {user_id}: {e}")
            continue
    
    if not predictions:
        return [{"error": f"Could not generate any predictions for user {user_id}. Model may not recognize user/items."}]

    # Sắp xếp theo rating giảm dần và giới hạn K
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return [
        {"item_id": iid, "predicted_rating": round(r, 2)}
        for iid, r in predictions[: min(k, len(predictions))]
    ]


# ==============================================================================
# 5. Các API Endpoint
# ==============================================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    """Endpoint để nhận ID người dùng và trả về gợi ý Top-K."""
    try:
        # force=True cho phép xử lý ngay cả khi Content-Type không chính xác
        data = request.get_json(force=True) 
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_id = data.get("user_id")
    k = data.get("top_k", 10)
    blocked_items = data.get("blocked_items", []) # Mặc định là list rỗng

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if topk_model is None:
        # Trả về lỗi 500 nếu mô hình chưa được tải thành công lúc khởi động
        return jsonify({"error": "Model not loaded on server. Check logs for startup errors."}), 500
    if not item_ids:
        return jsonify({"error": "No items available (check data file load status)"}), 500

    try:
        k = max(1, int(k))
    except (ValueError, TypeError):
        return jsonify({"error": "top_k must be a positive integer"}), 400

    if not isinstance(blocked_items, list) or not all(isinstance(iid, str) for iid in blocked_items):
        return jsonify({"error": "blocked_items must be a list of string IDs"}), 400

    # Sử dụng các biến toàn cục đã được tải sẵn
    recommendations = get_top_k_recommendations(
        user_id, item_ids, topk_model, k, blocked_items
    )

    if recommendations and "error" in recommendations[0]:
        return jsonify(recommendations[0]), 500

    results = [
        {
            "user_id": user_id,
            "parent_asin": rec["item_id"],
            "predicted_rating": rec["predicted_rating"],
        }
        for rec in recommendations
    ]
    return jsonify(results), 200


@app.route("/health", methods=["GET"])
def health():
    """Endpoint kiểm tra trạng thái của ứng dụng và mô hình."""
    return (
        jsonify(
            {
                "status": "healthy",
                "model_loaded": topk_model is not None,
                "items_count": len(item_ids),
                "model_source": MODEL_HF_URL,
            }
        ),
        200,
    )


if __name__ == "__main__":
    # Flask lắng nghe trên cổng được cung cấp bởi môi trường (Railway)
    port = int(os.environ.get("PORT", 8000))
    # Sử dụng host 0.0.0.0 để ứng dụng có thể được truy cập bên ngoài container
    app.run(host="0.0.0.0", port=port)
