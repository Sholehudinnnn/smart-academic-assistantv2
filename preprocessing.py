
import re
import pandas as pd


def preprocess_text(text):
    """
    Membersihkan teks untuk kebutuhan pencarian FAQ.
    Harus konsisten dengan preprocessing saat training TF-IDF.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


def preprocess_series(series):
    """Menerapkan preprocessing ke seluruh nilai dalam Series."""
    return series.fillna("").astype(str).apply(preprocess_text)


def preprocess_dataframe(
    dataframe,
    text_columns=("question",)
):
    """
    Membersihkan kolom teks yang ditentukan.
    DataFrame asli tidak diubah.
    """
    result = dataframe.copy()

    for column in text_columns:
        if column not in result.columns:
            raise ValueError(
                f"Kolom '{column}' tidak ditemukan dalam dataset."
            )

        result[column] = preprocess_series(result[column])

    return result
