from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = APP_DIR / "cyberbullying_tweets.csv"

app = Flask(__name__)
CORS(app)

# In-memory state (simple for a student project)
model: Pipeline | None = None
df: pd.DataFrame | None = None
trained: bool = False
loaded_csv_path: Path | None = None


def _validate_dataset_columns(dataframe: pd.DataFrame) -> tuple[bool, str | None]:
    required = {"tweet_text", "cyberbullying_type"}
    missing = sorted(required - set(map(str, dataframe.columns)))
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    return True, None


def load_data(csv_path: Path) -> tuple[bool, str | None]:
    global df, trained, model, loaded_csv_path

    if not csv_path.exists():
        return False, f"CSV not found at: {csv_path}"

    try:
        new_df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    except Exception as e:
        return False, f"Failed to read CSV: {e}"

    ok, err = _validate_dataset_columns(new_df)
    if not ok:
        return False, err

    df = new_df
    loaded_csv_path = csv_path

    # Reset model state because data changed
    model = None
    trained = False

    return True, None


def train_model() -> tuple[bool, str | dict]:
    global model, df, trained

    if df is None or df.empty:
        return False, "No data loaded"

    X = df["tweet_text"].astype(str)
    y = df["cyberbullying_type"].astype(str)

    if y.nunique() < 2:
        return False, "Need at least 2 different classes in 'cyberbullying_type'"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    model = pipeline
    trained = True

    return True, {
        "accuracy": float(accuracy),
        "report": report,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "classes": [str(c) for c in pipeline.classes_],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(
        {
            "success": True,
            "data_loaded": df is not None,
            "trained": trained,
            "rows": int(len(df)) if df is not None else 0,
            "csv_path": str(loaded_csv_path) if loaded_csv_path else None,
        }
    )


@app.route("/api/load-default", methods=["POST"])
def api_load_default():
    success, err = load_data(DEFAULT_CSV_PATH)
    if not success:
        return jsonify({"success": False, "error": err}), 400

    return jsonify(
        {
            "success": True,
            "rows": int(len(df)),  # type: ignore[arg-type]
            "columns": list(df.columns),  # type: ignore[union-attr]
            "sample": df.head(10).to_dict("records"),  # type: ignore[union-attr]
        }
    )


@app.route("/api/upload-csv", methods=["POST"])
def api_upload_csv():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file field named 'file'"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400

    uploads_dir = APP_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    save_path = uploads_dir / "dataset.csv"
    f.save(save_path)

    success, err = load_data(save_path)
    if not success:
        return jsonify({"success": False, "error": err}), 400

    return jsonify(
        {
            "success": True,
            "csv_path": str(save_path),
            "rows": int(len(df)),  # type: ignore[arg-type]
            "columns": list(df.columns),  # type: ignore[union-attr]
        }
    )


@app.route("/api/stats", methods=["GET"])
def api_stats():
    if df is None:
        return jsonify({"success": False, "error": "No data loaded"}), 400

    stats = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
        "label_distribution": df["cyberbullying_type"].value_counts().to_dict(),
    }

    return jsonify({"success": True, "stats": stats})


@app.route("/api/labels", methods=["GET"])
def api_labels():
    if df is None:
        return jsonify({"success": False, "error": "No data loaded"}), 400

    labels = sorted(df["cyberbullying_type"].dropna().astype(str).unique().tolist())
    return jsonify({"success": True, "labels": labels})


@app.route("/api/search", methods=["POST"])
def api_search():
    if df is None:
        return jsonify({"success": False, "error": "No data loaded"}), 400

    data = request.get_json(silent=True) or {}
    search_text = str(data.get("search_text", "")).strip()
    selected_labels = data.get("labels", []) or []
    limit = int(data.get("limit", 100) or 100)
    limit = max(1, min(limit, 1000))

    filtered_df = df

    if selected_labels:
        selected_labels = [str(x) for x in selected_labels]
        filtered_df = filtered_df[filtered_df["cyberbullying_type"].astype(str).isin(selected_labels)]

    if search_text:
        mask = filtered_df["tweet_text"].astype(str).str.contains(search_text, case=False, na=False)
        filtered_df = filtered_df[mask]

    results = filtered_df.head(limit).to_dict("records")
    return jsonify({"success": True, "count": int(len(filtered_df)), "results": results})


@app.route("/api/train", methods=["POST"])
def api_train():
    success, result = train_model()
    if success:
        return jsonify({"success": True, "data": result})
    return jsonify({"success": False, "error": result}), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not trained or model is None:
        return jsonify({"success": False, "error": "Model not trained yet"}), 400

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    try:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_
        prob_dict = {str(classes[i]): float(probabilities[i]) for i in range(len(classes))}

        return jsonify({"success": True, "prediction": str(prediction), "probabilities": prob_dict})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Attempt to load default CSV on startup (optional)
    if DEFAULT_CSV_PATH.exists():
        load_data(DEFAULT_CSV_PATH)

    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=True, host="0.0.0.0", port=port)

