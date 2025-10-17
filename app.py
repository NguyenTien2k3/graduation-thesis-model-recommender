import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import io

# --- Cấu hình logging ---
# Thiết lập định dạng log để bao gồm cả thời gian, cấp độ và thông điệp
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Khởi tạo ứng dụng Flask ---
app = Flask(__name__)

# --- Lấy đường dẫn file từ biến môi trường, nếu không có thì dùng giá trị mặc định ---
# Điều này giúp code linh hoạt hơn khi chạy trên các môi trường khác nhau (local, server)
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "./svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
)
ITEMS_CSV_PATH = os.environ.get(
    "ITEMS_CSV_PATH", "./Cell_Phones_and_Accessories.train.csv"
)


def safe_load_pickle(path, name):
    """
    Tải file pickle một cách an toàn từ đường dẫn cục bộ.
    Bắt các lỗi phổ biến như file không tồn tại, file hỏng hoặc lỗi phiên bản.

    Args:
        path (str): Đường dẫn đến file .pkl.
        name (str): Tên của model để ghi log.

    Returns:
        object: Model đã được tải hoặc None nếu có lỗi.
    """
    try:
        # >>> FIX 1: Ghi log dung lượng file để chẩn đoán lỗi Git LFS
        # Nếu dung lượng file quá nhỏ, rất có thể nó chỉ là file "con trỏ" của LFS
        file_size = os.path.getsize(path)
        logging.info(f"Attempting to load '{path}'. File size: {file_size} bytes.")
        if file_size < 1024: # Nếu file nhỏ hơn 1KB
             logging.warning(
                f"File size is very small. This might be a Git LFS pointer file. "
                "Ensure 'git lfs pull' was run in the deployment environment."
            )
        
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error(f"File not found at '{path}' for {name}. Skipping load.")
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        # Lỗi 'invalid load key, v' thường rơi vào đây
        logging.error(
            f"Error unpickling '{path}' for {name}: {e}. "
            "File might be corrupted, from an incompatible library version, or it might be a Git LFS pointer. Skipping load."
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred loading '{path}' for {name}: {e}. Skipping load.")
    return None


def safe_load_csv(path):
    """
    Tải file CSV một cách an toàn từ đường dẫn cục bộ.

    Args:
        path (str): Đường dẫn đến file .csv.

    Returns:
        pd.DataFrame: DataFrame đã được tải hoặc None nếu có lỗi.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logging.error(f"CSV file not found at '{path}'.")
    except pd.errors.EmptyDataError:
        logging.error(f"CSV file at '{path}' is empty.")
    except Exception as e:
        logging.error(f"Error loading CSV from '{path}': {e}.")
    return None


def load_and_process_items(csv_path):
    """
    Tải và xử lý dữ liệu các item từ file CSV.
    Kiểm tra sự tồn tại của cột 'parent_asin' trước khi xử lý.

    Args:
        csv_path (str): Đường dẫn tới file CSV chứa thông tin item.

    Returns:
        list: Danh sách các item ID duy nhất hoặc danh sách rỗng nếu có lỗi.
    """
    items_df = safe_load_csv(csv_path)
    if items_df is None:
        return []

    # >>> FIX 2: Kiểm tra xem cột 'parent_asin' có tồn tại không TRƯỚC KHI sử dụng
    if "parent_asin" not in items_df.columns:
        logging.error(
            f"Required column 'parent_asin' not found in '{csv_path}'. "
            f"Available columns are: {list(items_df.columns)}"
        )
        return []

    try:
        items_df["parent_asin"] = (
            items_df["parent_asin"].astype(str).str.split(",").str[0]
        )
        unique_items = items_df["parent_asin"].dropna().unique().tolist()
        logging.info(f"Successfully loaded and processed {len(unique_items)} unique items.")
        return unique_items
    except Exception as e:
        logging.error(f"Error processing item data from CSV: {e}.")
        return []


# --- Tải các tài nguyên khi ứng dụng khởi động ---
logging.info(f"Loading SVD model from: {MODEL_PATH}")
topk_model = safe_load_pickle(MODEL_PATH, "Top-K SVD model")

logging.info(f"Loading items data from: {ITEMS_CSV_PATH}")
item_ids = load_and_process_items(ITEMS_CSV_PATH)


def get_top_k_recommendations(user_id, all_item_ids, model, k=10, blocked_items=None):
    """
    Tạo ra danh sách gợi ý top-K cho một người dùng.
    """
    if model is None:
        return [{"error": "Model is not loaded on the server."}]
    if not all_item_ids:
        return [{"error": "Item list is empty. Check data source."}]

    blocked_set = set(blocked_items or [])
    valid_items = [iid for iid in all_item_ids if iid not in blocked_set]

    if not valid_items:
        return [{"error": "No valid items remain after filtering blocked items."}]

    predictions = []
    for iid in valid_items:
        try:
            pred = model.predict(uid=user_id, iid=iid)
            predictions.append((pred.iid, pred.est))
        except Exception as e:
            logging.warning(f"Could not predict for user '{user_id}', item '{iid}': {e}")
            continue

    if not predictions:
        return [{"error": "No predictions could be made for the given user."}]

    predictions.sort(key=lambda x: x[1], reverse=True)

    return [
        {"item_id": iid, "predicted_rating": round(rating, 4)}
        for iid, rating in predictions[:k]
    ]


@app.route("/recommend", methods=["POST"])
def recommend():
    """Endpoint API để nhận yêu cầu và trả về danh sách gợi ý."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_id = data.get("user_id")
    k = data.get("top_k", 10)
    blocked_items = data.get("blocked_items", ["B00K30H3O8"])

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if topk_model is None:
        return jsonify({"error": "Model not available on server"}), 503
    if not item_ids:
        return jsonify({"error": "Items not available on server"}), 503

    try:
        k = int(k)
        if k <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "top_k must be a positive integer"}), 400

    if not isinstance(blocked_items, list) or not all(isinstance(i, str) for i in blocked_items):
        return jsonify({"error": "blocked_items must be a list of strings"}), 400

    recommendations = get_top_k_recommendations(
        user_id, item_ids, topk_model, k, blocked_items
    )

    if recommendations and "error" in recommendations[0]:
        return jsonify(recommendations[0]), 500

    return jsonify(recommendations), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint để kiểm tra trạng thái của ứng dụng."""
    return jsonify({
        "status": "ok",
        "model_loaded": topk_model is not None,
        "items_loaded": len(item_ids),
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)

