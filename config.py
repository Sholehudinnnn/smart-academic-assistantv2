
from pathlib import Path

# Lokasi folder aplikasi
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = (
    APP_DIR.parent
    if (APP_DIR.parent / "01_data").is_dir()
    else APP_DIR
)

# Dataset
DATA_DIR = PROJECT_DIR / "01_data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model dan artifacts berada di dalam 04_app
MODEL_DIR = APP_DIR / "models"
ARTIFACT_DIR = APP_DIR / "artifacts"
FEEDBACK_DB_PATH = ARTIFACT_DIR / "feedback.sqlite3"

FAQ_DATA_PATH = RAW_DATA_DIR / "faq_academic.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_queries.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_queries.csv"

TFIDF_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
TFIDF_METADATA_PATH = ARTIFACT_DIR / "tfidf_metadata.joblib"

DEFAULT_TOP_K = 3
MIN_SIMILARITY_SCORE = 0.20
DEBUG = True
