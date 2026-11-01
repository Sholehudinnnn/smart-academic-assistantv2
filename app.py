
import json
import math
import sqlite3

import pandas as pd
from flask import Flask, jsonify, render_template, request
from config import PROJECT_DIR
from feedback_store import save_feedback
from retriever import search_faq

app = Flask(__name__)

EVALUATION_RESULTS_PATH = (
    PROJECT_DIR / "03_model_pipeline" / "artifacts" / "evaluation_results.csv"
)
COLAB_COMPARISON_PATH = (
    PROJECT_DIR
    / "03_model_pipeline"
    / "artifacts"
    / "colab_evaluation_comparison.csv"
)
EVALUATION_TEST_PREDICTIONS_PATH = (
    PROJECT_DIR
    / "03_model_pipeline"
    / "artifacts"
    / "evaluation_predictions_test_tfidf.csv"
)
EDA_SUMMARY_PATH = (
    PROJECT_DIR / "02_exploration" / "reports" / "eda_summary.json"
)
REQUIRED_EVALUATION_COLUMNS = {
    "Model",
    "Split",
    "Samples",
    "Top-1 Accuracy",
    "Recall@3",
    "MRR",
    "Coverage@0.20",
    "Score Source",
}
COLAB_COMPARISON_COLUMNS = {
    "Model",
    "Top-1 Accuracy",
    "Recall@3",
    "MRR",
}


def load_colab_comparison():
    if not COLAB_COMPARISON_PATH.is_file():
        raise FileNotFoundError(
            "Data grafik notebook tidak tersedia: "
            f"{COLAB_COMPARISON_PATH.name}."
        )

    comparison = pd.read_csv(COLAB_COMPARISON_PATH)
    missing_columns = COLAB_COMPARISON_COLUMNS - set(comparison.columns)
    if missing_columns:
        raise ValueError(
            "Kolom data grafik notebook tidak lengkap: "
            + ", ".join(sorted(missing_columns))
        )
    if comparison.empty or comparison["Model"].duplicated().any():
        raise ValueError("Data grafik notebook kosong atau memiliki model duplikat.")

    for column in COLAB_COMPARISON_COLUMNS - {"Model"}:
        comparison[column] = pd.to_numeric(comparison[column], errors="raise")
        if (
            not comparison[column].map(math.isfinite).all()
            or not comparison[column].between(0, 1).all()
        ):
            raise ValueError(
                f"Nilai {column} pada data grafik notebook tidak valid."
            )

    return {
        "models": comparison.to_dict(orient="records"),
        "metrics": ["Top-1 Accuracy", "Recall@3", "MRR"],
    }


def load_evaluation_report():
    if not EVALUATION_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            "Hasil evaluasi belum tersedia. Jalankan "
            "python .\\03_model_pipeline\\evaluate_retrieval.py."
        )

    metrics = pd.read_csv(EVALUATION_RESULTS_PATH)
    missing_columns = REQUIRED_EVALUATION_COLUMNS - set(metrics.columns)
    if missing_columns:
        raise ValueError(
            "Kolom hasil evaluasi tidak lengkap: "
            + ", ".join(sorted(missing_columns))
        )
    if metrics.empty:
        raise ValueError("File hasil evaluasi tidak memiliki baris metrik.")

    expected_metrics = (
        "Top-1 Accuracy",
        "Recall@3",
        "MRR",
        "Coverage@0.20",
    )
    for column in expected_metrics:
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")
        if not metrics[column].map(math.isfinite).all():
            raise ValueError(
                f"Metrik {column} berisi angka yang tidak valid."
            )
        if not metrics[column].between(0, 1).all():
            raise ValueError(
                f"Metrik {column} harus berada antara 0 dan 1."
            )

    metrics["Samples"] = pd.to_numeric(
        metrics["Samples"],
        errors="raise",
    ).astype(int)
    if (metrics["Samples"] <= 0).any():
        raise ValueError("Jumlah sampel evaluasi harus lebih dari nol.")

    rows = metrics.to_dict(orient="records")
    for row in rows:
        row["Samples"] = int(row["Samples"])
        for column in expected_metrics:
            row[column] = float(row[column])
        row["Model"] = str(row["Model"])
        row["Split"] = str(row["Split"])
        row["Score Source"] = str(row["Score Source"])

    provenance = {
        "data_provenance": (
            "Evaluasi yang tersedia memakai query sintetis turunan FAQ; "
            "belum memvalidasi pertanyaan nyata dari mahasiswa."
        ),
        "faq_count": None,
        "category_count": None,
        "source_count": None,
    }
    if EDA_SUMMARY_PATH.is_file():
        with EDA_SUMMARY_PATH.open(encoding="utf-8") as summary_file:
            eda_summary = json.load(summary_file)
        provenance.update(
            {
                "data_provenance": str(
                    eda_summary.get(
                        "data_provenance",
                        provenance["data_provenance"],
                    )
                ),
                "faq_count": eda_summary.get("faq_count"),
                "category_count": eda_summary.get("category_count"),
                "source_count": eda_summary.get("source_count"),
            }
        )

    predictions = []
    test_misranking_count = None
    if EVALUATION_TEST_PREDICTIONS_PATH.is_file():
        prediction_data = pd.read_csv(EVALUATION_TEST_PREDICTIONS_PATH)
        required_prediction_columns = {
            "query_id",
            "question",
            "expected_faq_id",
            "predicted_faq_id",
            "predicted_category",
            "predicted_source",
            "top_score",
            "expected_rank",
            "hit_at_3",
        }
        missing_prediction_columns = (
            required_prediction_columns - set(prediction_data.columns)
        )
        if missing_prediction_columns:
            raise ValueError(
                "Kolom prediksi pengujian tidak lengkap: "
                + ", ".join(sorted(missing_prediction_columns))
            )

        prediction_data["expected_rank"] = pd.to_numeric(
            prediction_data["expected_rank"],
            errors="raise",
        )
        prediction_data["top_score"] = pd.to_numeric(
            prediction_data["top_score"],
            errors="raise",
        )
        if not prediction_data["top_score"].map(math.isfinite).all():
            raise ValueError("Skor prediksi pengujian berisi angka tidak valid.")

        errors = prediction_data.loc[
            prediction_data["expected_rank"] != 1
        ]
        test_misranking_count = len(errors)
        errors = errors.head(10)
        predictions = errors[
            [
                "query_id",
                "question",
                "expected_faq_id",
                "predicted_faq_id",
                "predicted_category",
                "predicted_source",
                "top_score",
                "expected_rank",
                "hit_at_3",
            ]
        ].to_dict(orient="records")
        predictions = [
            {
                "query_id": str(row["query_id"]),
                "question": str(row["question"]),
                "expected_faq_id": str(row["expected_faq_id"]),
                "predicted_faq_id": str(row["predicted_faq_id"]),
                "predicted_category": str(row["predicted_category"]),
                "predicted_source": str(row["predicted_source"]),
                "top_score": float(row["top_score"]),
                "expected_rank": int(row["expected_rank"]),
                "hit_at_3": bool(row["hit_at_3"]),
            }
            for row in predictions
        ]

    return {
        "metrics": rows,
        "colab_comparison": load_colab_comparison(),
        "provenance": provenance,
        "test_misrankings": predictions,
        "test_misranking_count": test_misranking_count,
    }


# ==============================
# HALAMAN UTAMA
# ==============================
@app.route("/")
def index():
    page = request.args.get("page", "chat")

    if page not in ("chat", "history", "evaluation"):
        page = "chat"

    evaluation_report = None
    evaluation_error = None
    if page == "evaluation":
        try:
            evaluation_report = load_evaluation_report()
        except FileNotFoundError as error:
            evaluation_error = str(error)
        except (OSError, ValueError, json.JSONDecodeError):
            app.logger.exception("Gagal memuat laporan evaluasi")
            evaluation_error = (
                "Laporan hasil pengujian tidak dapat dibaca. "
                "Periksa log server untuk detailnya."
            )

    return render_template(
        "index.html",
        page=page,
        evaluation_report=evaluation_report,
        evaluation_error=evaluation_error,
    )


@app.route("/welcome")
def welcome():
    return render_template("welcome.html")


@app.route("/api/evaluation", methods=["GET"])
def evaluation():
    try:
        return jsonify(load_evaluation_report())
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404
    except (OSError, ValueError, json.JSONDecodeError):
        app.logger.exception("Gagal memuat laporan evaluasi")
        return jsonify({
            "error": (
                "Laporan hasil pengujian tidak dapat dibaca. "
                "Periksa artifacts evaluasi."
            )
        }), 500


# ==============================
# CEK STATUS SERVER
# ==============================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ==============================
# API CHATBOT
# ==============================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    message = data.get("message")
    if not isinstance(message, str):
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    message = message.strip()

    if not message:
        return jsonify({
            "error": "Pertanyaan tidak boleh kosong."
        }), 400

    try:
        result = search_faq(message)
        best_match = result.get("results", [])
        best_match = best_match[0] if best_match else {}

        return jsonify({
            "question": message,
            "answer": result["answer"],
            "results": result.get("results", []),
            "category": best_match.get("category"),
            "source": best_match.get("source"),
        })

    except Exception:
        app.logger.exception("Terjadi kesalahan pada chatbot")

        return jsonify({
            "error": "Maaf, terjadi kesalahan saat mencari jawaban. Silakan coba lagi."
        }), 500


# ==============================
# API FEEDBACK JAWABAN
# ==============================
@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Data feedback tidak valid."}), 400

    question = data.get("question")
    answer = data.get("answer")
    rating = data.get("rating")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    if not isinstance(answer, str) or not answer.strip():
        return jsonify({"error": "Jawaban tidak boleh kosong."}), 400

    if len(question) > 20000 or len(answer) > 20000:
        return jsonify({"error": "Teks feedback terlalu panjang."}), 400

    if rating not in ("helpful", "not_helpful"):
        return jsonify({"error": "Pilihan feedback tidak valid."}), 400

    try:
        save_feedback(question.strip(), answer.strip(), rating)
    except (OSError, sqlite3.Error):
        app.logger.exception("Gagal menyimpan feedback chatbot")
        return jsonify({
            "error": "Feedback gagal disimpan. Silakan coba lagi."
        }), 500

    return jsonify({"status": "ok"})


# ==============================
# JALANKAN SERVER
# ==============================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", "5000")),
        debug=False,
    )
