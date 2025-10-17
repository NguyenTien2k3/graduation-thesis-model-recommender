import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import logging
import requests
import io  # Để đọc CSV từ bytes

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)


# ====== Hàm load model an toàn ======
def safe_load_pickle(url_or_path, name):
    try:
        if url_or_path.startswith("http"):
            response = requests.get(url_or_path)
            response.raise_for_status()
            return pickle.loads(response.content)
        else:
            with open(url_or_path, "rb") as f:
                return pickle.load(f)
    except (FileNotFoundError, requests.exceptions.RequestException, Exception) as e:
        logging.error(f"Error loading '{url_or_path}' for {name}: {e}. Skipping load.")
        return None


# Load model SVD Top-K từ Hugging Face
model_url = "./svd_model_nf32_lr0.001_reg0.05_ep40_p1.0_balanced.pkl"
topk_model = safe_load_pickle(model_url, "Top-K SVD model")


# ====== Hàm load CSV an toàn ======
def safe_load_csv(url_or_path):
    try:
        if url_or_path.startswith("http"):
            response = requests.get(url_or_path)
            response.raise_for_status()
            return pd.read_csv(io.BytesIO(response.content))
        else:
            return pd.read_csv(url_or_path)
    except Exception as e:
        logging.error(f"Error loading CSV from '{url_or_path}': {e}.")
        return None


def load_items():
    csv_url = "./Cell_Phones_and_Accessories.train.csv"
    items_df = safe_load_csv(csv_url)
    if items_df is None:
        return []
    try:
        items_df["parent_asin"] = (
            items_df["parent_asin"].astype(str).str.split(",").str[0]
        )
        unique_items = items_df["parent_asin"].dropna().unique().tolist()
        logging.info(f"Loaded {len(unique_items)} unique items")
        return unique_items
    except Exception as e:
        logging.error(f"Error processing CSV data: {e}.")
        return []


item_ids = load_items()


# ====== Logic gợi ý top-K ======
def get_top_k_recommendations(user_id, item_ids, model, k=10, blocked_items=None):
    if model is None:
        return [{"error": "Model not loaded"}]
    if not item_ids:
        return [{"error": "No items available"}]
    blocked_set = set(blocked_items or [])
    valid_items = [iid for iid in item_ids if iid not in blocked_set]
    if not valid_items:
        return [{"error": "No valid items after filtering blocked items"}]

    predictions = []
    seen_items = set()
    for iid in valid_items:
        if iid in seen_items:
            continue
        try:
            pred = model.predict(uid=user_id, iid=iid).est
            predictions.append((iid, pred))
            seen_items.add(iid)
        except Exception as e:
            logging.warning(f"Skipping item {iid} for user {user_id}: {e}")
            continue
    if not predictions:
        return [{"error": "No predictions could be made"}]

    # Sắp xếp theo rating giảm dần, giới hạn k
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [
        {"item_id": iid, "predicted_rating": round(r, 2)}
        for iid, r in predictions[: min(k, len(predictions))]
    ]


# ====== API /recommend ======
@app.route("/recommend", methods=["POST"])
def recommend():
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
        return jsonify({"error": "Model not loaded on server"}), 500
    if not item_ids:
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


# ====== API /health ======
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