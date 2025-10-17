import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import io  # Đọc CSV từ bytes

# ========== Cấu hình logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ====== Hàm load model an toàn ======
def safe_load_pickle(url_or_path, name):
    logger.info(f"🔹 Bắt đầu load model '{name}' từ: {url_or_path}")
    try:
        if url_or_path.startswith("http"):
            response = requests.get(url_or_path)
            response.raise_for_status()
            model = pickle.loads(response.content)
        else:
            with open(url_or_path, "rb") as f:
                model = pickle.load(f)
        logger.info(f"✅ Load model '{name}' thành công.")
        return model
    except (FileNotFoundError, requests.exceptions.RequestException, Exception) as e:
        logger.error(f"❌ Lỗi khi load '{url_or_path}' cho {name}: {e}")
        return None


# ====== Load model SVD ======
model_url = "./svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
topk_model = safe_load_pickle(model_url, "Top-K SVD model")


# ====== Hàm load CSV an toàn ======
def safe_load_csv(url_or_path):
    logger.info(f"🔹 Đang load file CSV: {url_or_path}")
    try:
        if url_or_path.startswith("http"):
            response = requests.get(url_or_path)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content))
        else:
            df = pd.read_csv(url_or_path)
        logger.info(f"✅ Đọc CSV thành công, có {len(df)} dòng.")
        return df
    except Exception as e:
        logger.error(f"❌ Lỗi khi load CSV từ '{url_or_path}': {e}")
        return None


# ====== Load danh sách item ======
def load_items():
    csv_url = "./Cell_Phones_and_Accessories.train.csv"
    items_df = safe_load_csv(csv_url)
    if items_df is None:
        logger.warning("⚠️ Không thể load danh sách item (CSV rỗng hoặc lỗi).")
        return []
    try:
        # Xử lý parent_asin có thể chứa nhiều giá trị (chỉ lấy giá trị đầu)
        items_df["parent_asin"] = (
            items_df["parent_asin"].astype(str).str.split(",").str[0]
        )
        unique_items = items_df["parent_asin"].dropna().unique().tolist()
        logger.info(f"✅ Đã load {len(unique_items)} item duy nhất.")
        return unique_items
    except Exception as e:
        logger.error(f"❌ Lỗi khi xử lý dữ liệu CSV: {e}")
        return []


item_ids = load_items()


# ====== Logic gợi ý top-K ======
def get_top_k_recommendations(user_id, item_ids, model, k=10, blocked_items=None):
    logger.info(
        f"🔹 Tính toán gợi ý cho user={user_id}, top_k={k}, blocked={len(blocked_items or [])}"
    )

    if model is None:
        return [{"error": "Model not loaded"}]
    if not item_ids:
        return [{"error": "No items available"}]

    blocked_set = set(blocked_items or [])
    valid_items = [iid for iid in item_ids if iid not in blocked_set]
    if not valid_items:
        return [{"error": "No valid items after filtering blocked items"}]

    predictions = []
    # Chỉ tính toán rating cho các item chưa tương tác (hoặc chưa bị chặn)
    for iid in valid_items:
        try:
            # model.predict sẽ ước tính rating (est)
            pred = model.predict(uid=user_id, iid=iid).est
            predictions.append((iid, pred))
        except Exception as e:
            # Bỏ qua item nếu có lỗi trong quá trình dự đoán (ít xảy ra với surprise)
            logger.warning(f"⚠️ Bỏ qua item {iid} cho user {user_id}: {e}")
            continue

    if not predictions:
        logger.warning(f"⚠️ Không tạo được gợi ý nào cho user {user_id}.")
        return [{"error": "No predictions could be made"}]

    # Sắp xếp và lấy top K
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[: min(k, len(predictions))]
    logger.info(f"✅ Trả về {len(top_predictions)} gợi ý cho user {user_id}.")
    return [
        {"item_id": iid, "predicted_rating": round(r, 2)}
        for iid, r in top_predictions
    ]


# ====== API /recommend (POST) ======
@app.route("/recommend", methods=["POST"])
def recommend():
    logger.info("📩 Nhận yêu cầu POST /recommend")
    try:
        # force=True cho phép đọc data ngay cả khi Content-Type không phải application/json
        data = request.get_json(force=True) 
        logger.info(f"📦 Payload nhận được: {data}")
    except Exception:
        logger.error("❌ Payload không hợp lệ.")
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_id = data.get("user_id")
    k = data.get("top_k", 10)
    # Default blocked_items để test
    blocked_items = data.get("blocked_items", ["B00K30H3O8"]) 

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if topk_model is None:
        logger.error("❌ Model chưa được load.")
        return jsonify({"error": "Model not loaded on server"}), 500
    if not item_ids:
        logger.error("❌ Không có dữ liệu item khả dụng.")
        return jsonify({"error": "No items available (check data file)"}), 500

    try:
        k = max(1, int(k))
    except (ValueError, TypeError):
        return jsonify({"error": "top_k must be a positive integer"}), 400

    if not all(isinstance(iid, str) for iid in blocked_items):
        return jsonify({"error": "blocked_items must contain valid string IDs"}), 400

    recommendations = get_top_k_recommendations(
        user_id, item_ids, topk_model, k, blocked_items
    )

    if "error" in recommendations[0]:
        logger.warning(f"⚠️ Lỗi khi tạo gợi ý cho user {user_id}: {recommendations[0]}")
        return jsonify(recommendations[0]), 500

    # Định dạng kết quả đầu ra
    results = [
        {
            "user_id": user_id,
            "parent_asin": rec["item_id"],
            "predicted_rating": rec["predicted_rating"],
        }
        for rec in recommendations
    ]

    logger.info(f"✅ Hoàn tất trả kết quả cho user {user_id}.")
    return jsonify(results), 200


# ====== API /health (GET) ======
@app.route("/health", methods=["GET"])
def health():
    logger.info("🔍 Kiểm tra tình trạng hệ thống (/health)")
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
    # Dùng cổng từ biến môi trường PORT (Railway sẽ cung cấp) hoặc mặc định 8000
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Server đang chạy tại http://0.0.0.0:{port}")
    # Khi chạy cục bộ, dùng app.run. Trong Docker, Gunicorn sẽ chạy app.
    app.run(host="0.0.0.0", port=port)