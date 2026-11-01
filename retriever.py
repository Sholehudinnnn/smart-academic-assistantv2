
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text
from config import (
    FAQ_DATA_PATH,
    TRAIN_DATA_PATH,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    TFIDF_METADATA_PATH,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE,
)


# ============================================================
# 1. VALIDASI FILE
# ============================================================

REQUIRED_FILES = {
    "TF-IDF vectorizer": TFIDF_VECTORIZER_PATH,
    "TF-IDF matrix": TFIDF_MATRIX_PATH,
    "TF-IDF metadata": TFIDF_METADATA_PATH,
    "Data training": TRAIN_DATA_PATH,
    "Dataset FAQ": FAQ_DATA_PATH,
}

missing_files = [
    f"{name}: {path}"
    for name, path in REQUIRED_FILES.items()
    if not Path(path).is_file()
]

if missing_files:
    raise FileNotFoundError(
        "File yang dibutuhkan retriever tidak ditemukan:\n- "
        + "\n- ".join(missing_files)
    )


# ============================================================
# 2. MEMUAT MODEL DAN DATA
# ============================================================

vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
metadata = joblib.load(TFIDF_METADATA_PATH)

train_data = pd.read_csv(TRAIN_DATA_PATH)
faq_data = pd.read_csv(FAQ_DATA_PATH)


# ============================================================
# 3. VALIDASI STRUKTUR DATA
# ============================================================

required_train_columns = {"faq_id"}
required_faq_columns = {
    "faq_id",
    "question",
    "answer",
    "category",
    "source",
}

missing_train_columns = required_train_columns - set(train_data.columns)
missing_faq_columns = required_faq_columns - set(faq_data.columns)

if missing_train_columns:
    raise ValueError(
        "Kolom data training tidak ditemukan: "
        + ", ".join(sorted(missing_train_columns))
    )

if missing_faq_columns:
    raise ValueError(
        "Kolom dataset FAQ tidak ditemukan: "
        + ", ".join(sorted(missing_faq_columns))
    )

if not hasattr(metadata, "get") or "faq_ids" not in metadata:
    raise ValueError(
        "Metadata TF-IDF harus berupa dictionary "
        "yang memiliki key 'faq_ids'."
    )

if len(train_data) != tfidf_matrix.shape[0]:
    raise ValueError(
        "Jumlah data training tidak cocok dengan jumlah baris "
        "matriks TF-IDF."
    )

if tfidf_matrix.shape[1] != len(vectorizer.get_feature_names_out()):
    raise ValueError(
        "Jumlah fitur vectorizer tidak cocok dengan matriks TF-IDF."
    )

train_data["faq_id"] = train_data["faq_id"].astype(str)
faq_data["faq_id"] = faq_data["faq_id"].astype(str)

train_faq_ids = train_data["faq_id"].tolist()
metadata_faq_ids = [str(item) for item in metadata["faq_ids"]]

if train_faq_ids != metadata_faq_ids:
    raise ValueError(
        "Urutan FAQ dalam data training tidak cocok "
        "dengan metadata model."
    )

if faq_data["faq_id"].duplicated().any():
    raise ValueError(
        "Dataset FAQ memiliki faq_id duplikat. "
        "Periksa dataset sebelum menjalankan chatbot."
    )

if faq_data[["question", "answer", "category", "source"]].isna().any().any():
    raise ValueError(
        "Dataset FAQ memiliki pertanyaan, jawaban, kategori, "
        "atau sumber kosong."
    )

# Pastikan semua ID dalam training memiliki FAQ pasangan.
faq_id_set = set(faq_data["faq_id"])
unmatched_ids = [
    faq_id for faq_id in train_faq_ids
    if faq_id not in faq_id_set
]

if unmatched_ids:
    raise ValueError(
        "Sebagian faq_id data training tidak ditemukan dalam dataset FAQ: "
        + ", ".join(unmatched_ids[:10])
    )


# ============================================================
# 5. PENYARINGAN PERTANYAAN DI LUAR TOPIK
# ============================================================

# Frasa nonakademik yang eksplisit.
# Daftar ini merupakan penyaring sederhana, bukan klasifikasi universal.
OFF_TOPIC_PHRASES = {
    "nasi goreng",
    "tanaman cabai",
    "merawat tanaman",
    "resep makanan",
    "resep masakan",
    "hewan peliharaan",
}

# Kata yang dapat menunjukkan konteks akademik.
# Daftar ini membantu menghindari penolakan terhadap pertanyaan
# akademik yang tidak memakai istilah yang sama persis dengan FAQ.
ACADEMIC_KEYWORDS = {
    "jadwal",
    "kuliah",
    "krs",
    "nilai",
    "akademik",
    "administrasi",
    "kehadiran",
    "absensi",
    "keuangan",
    "biaya",
    "ukt",
    "dosen",
    "pembimbing",
    "skripsi",
    "tugas",
    "akhir",
    "ujian",
    "kelas",
    "beasiswa",
    "wisuda",
    "perpustakaan",
    "registrasi",
    "kalender",
    "praktikum",
    "magang",
    "mbkm",
    "organisasi",
    "kampus",
    "parkir",
    "kartu",
    "mahasiswa",
    "email",
    "kelulusan",
    "sks",
    "surat",
    "rekomendasi",
    "pengaduan",
    "keamanan",
    "akun",
    "semester",
    "fakultas",
    "program",
    "studi",
    "prodi",
    "transkrip",
    "ipk",
    "irs",
    "daring",
    "online",
    "siakad",
    "portal",
    "cuti",
    "aktif",
    "perkuliahan",
    "konversi",
    "fasilitas",
    "layanan",
}


def contains_off_topic_phrase(question):
    """Mendeteksi frasa nonakademik eksplisit yang dikenal."""

    normalized = preprocess_text(question)

    return any(
        phrase in normalized
        for phrase in OFF_TOPIC_PHRASES
    )


def has_academic_keyword(question):
    """Memeriksa apakah pertanyaan mengandung kata akademik."""

    tokens = set(preprocess_text(question).split())

    return bool(tokens.intersection(ACADEMIC_KEYWORDS))


# ============================================================
# 6. PENCARIAN FAQ
# ============================================================

def search_faq(question, top_k=DEFAULT_TOP_K):
    """
    Mencari FAQ berdasarkan cosine similarity TF-IDF.

    Return:
        {
            "answer": jawaban terbaik atau pesan fallback,
            "results": daftar FAQ yang relevan
        }

    Catatan:
        Penyaringan topik menggunakan aturan sederhana.
        Pertanyaan yang belum dikenali tetap perlu dievaluasi
        dengan contoh pengujian yang lebih beragam.
    """

    question = preprocess_text(question)

    # ----------------------------------------
    # Validasi input kosong
    # ----------------------------------------

    if not question:
        return {
            "answer": None,
            "results": [],
        }

    try:
        top_k = max(1, int(top_k))
    except (TypeError, ValueError, OverflowError):
        top_k = DEFAULT_TOP_K

    # ----------------------------------------
    # Tolak frasa nonakademik yang eksplisit
    # ----------------------------------------

    if contains_off_topic_phrase(question):
        return {
            "answer": (
                "Maaf, saya hanya dapat membantu pertanyaan "
                "seputar layanan dan informasi akademik kampus. "
                "Silakan tanyakan tentang jadwal kuliah, KRS, nilai, "
                "administrasi, atau layanan kampus lainnya."
            ),
            "results": [],
        }

    # ----------------------------------------
    # Ubah pertanyaan menjadi vektor TF-IDF
    # ----------------------------------------

    query_vector = vectorizer.transform([question])

    if query_vector.nnz == 0:
        return {
            "answer": (
                "Maaf, pertanyaan belum dapat dipahami. "
                "Coba gunakan kata-kata yang berkaitan dengan "
                "layanan akademik kampus."
            ),
            "results": [],
        }

    # ----------------------------------------
    # Hitung cosine similarity
    # ----------------------------------------

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix,
    ).ravel()

    if scores.size == 0:
        return {
            "answer": (
                "Maaf, saya belum menemukan jawaban yang relevan."
            ),
            "results": [],
        }

    max_score = float(scores.max())

    if max_score < MIN_SIMILARITY_SCORE:
        return {
            "answer": (
                "Maaf, saya belum menemukan jawaban yang cukup relevan. "
                "Coba tuliskan pertanyaan akademik dengan lebih jelas."
            ),
            "results": [],
        }

    # ----------------------------------------
    # Urutkan hasil dari skor tertinggi
    # ----------------------------------------

    ranked_indices = scores.argsort()[::-1]

    results = []
    seen_faq_ids = set()

    for idx in ranked_indices:
        score = float(scores[idx])

        if score < MIN_SIMILARITY_SCORE:
            break

        faq_id = str(train_data.iloc[idx]["faq_id"])

        if faq_id in seen_faq_ids:
            continue

        faq_match = faq_data.loc[
            faq_data["faq_id"] == faq_id
        ]

        if faq_match.empty:
            continue

        faq = faq_match.iloc[0]

        results.append({
            "faq_id": faq_id,
            "question": str(faq["question"]),
            "answer": str(faq["answer"]),
            "category": str(faq["category"]),
            "source": str(faq["source"]),
            "score": round(score, 4),
        })

        seen_faq_ids.add(faq_id)

        if len(results) >= top_k:
            break

    # ----------------------------------------
    # Tangani hasil kosong
    # ----------------------------------------

    if not results:
        return {
            "answer": (
                "Maaf, saya belum menemukan jawaban yang relevan."
            ),
            "results": [],
        }

    # ----------------------------------------
    # Pemeriksaan relevansi tambahan
    # ----------------------------------------

    # Jika tidak ada indikasi akademik, gunakan fallback
    # hanya ketika skor terbaik juga relatif rendah.
    #
    # Ini mengurangi risiko pertanyaan umum yang kebetulan
    # berbagi kata dengan FAQ langsung dianggap pertanyaan akademik.
    if not has_academic_keyword(question) and max_score < 0.40:
        return {
            "answer": (
                "Maaf, saya hanya dapat membantu pertanyaan "
                "seputar layanan dan informasi akademik kampus. "
                "Silakan tanyakan tentang jadwal kuliah, KRS, nilai, "
                "administrasi, atau layanan kampus lainnya."
            ),
            "results": [],
        }

    # ----------------------------------------
    # Kembalikan jawaban terbaik
    # ----------------------------------------

    return {
        "answer": results[0]["answer"],
        "results": results,
    }


# ============================================================
# 7. INFORMASI STATUS RETRIEVER
# ============================================================

def get_retriever_info():
    """Mengembalikan informasi ringkas model yang dimuat."""

    return {
        "model": "TF-IDF + Cosine Similarity",
        "faq_training_count": len(train_data),
        "faq_dataset_count": len(faq_data),
        "feature_count": int(tfidf_matrix.shape[1]),
        "default_top_k": DEFAULT_TOP_K,
        "minimum_similarity_score": MIN_SIMILARITY_SCORE,
    }
