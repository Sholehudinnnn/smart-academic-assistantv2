import sqlite3

from config import FEEDBACK_DB_PATH


def save_feedback(question: str, answer: str, rating: str) -> None:
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(FEEDBACK_DB_PATH, timeout=5) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating TEXT NOT NULL
                    CHECK (rating IN ('helpful', 'not_helpful')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO feedback (question, answer, rating)
            VALUES (?, ?, ?)
            """,
            (question, answer, rating),
        )
