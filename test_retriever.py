
"""
Pengujian Retriever - Smart Academic Assistant
Jalankan dari folder 04_app:
    python test_retriever.py
"""

from retriever import search_faq, get_retriever_info


# ============================================================
# 1. INFORMASI MODEL
# ============================================================

def run_model_info_check():
    print("\n" + "=" * 70)
    print("1. INFORMASI MODEL RETRIEVER")
    print("=" * 70)

    try:
        info = get_retriever_info()

        for key, value in info.items():
            print(f"{key}: {value}")

        return True

    except Exception as error:
        print("GAGAL membaca informasi model:", error)
        return False


# ============================================================
# 2. PENGUJIAN PERTANYAAN
# ============================================================

def run_question_checks():
    questions = [
        {
            "category": "Akademik - Jadwal kuliah",
            "question": "Bagaimana cara melihat jadwal kuliah?",
            "expected_keyword": "jadwal",
        },
        {
            "category": "Akademik - Kartu mahasiswa",
            "question": "Bagaimana cara mengurus kartu mahasiswa yang hilang?",
            "expected_keyword": "kartu mahasiswa",
        },
        {
            "category": "Akademik - Surat aktif kuliah",
            "question": "Bagaimana cara mengurus surat aktif kuliah?",
            "expected_keyword": "surat aktif kuliah",
        },
        {
            "category": "Di luar topik - Resep makanan",
            "question": "Apa resep membuat nasi goreng?",
            "expected_keyword": None,
        },
        {
            "category": "Di luar topik - Tanaman",
            "question": "Bagaimana cara merawat tanaman cabai?",
            "expected_keyword": None,
        },
        {
            "category": "Pertanyaan tidak jelas",
            "question": "xyz abc qwerty",
            "expected_keyword": None,
        },
        {
            "category": "Input kosong",
            "question": "",
            "expected_keyword": None,
        },
    ]

    passed = 0
    failed = 0
    records = []

    print("\n" + "=" * 70)
    print("2. PENGUJIAN HASIL RETRIEVAL")
    print("=" * 70)

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"\n[{index}] {item['category']}")
        print("Pertanyaan:", repr(question))

        try:
            result = search_faq(question)
            answer = result.get("answer")
            results = result.get("results", [])

            print("Jawaban:", answer)
            print("Jumlah hasil:", len(results))

            if results:
                print("\nHasil teratas:")

                for rank, faq in enumerate(results, start=1):
                    print(
                        f"  {rank}. [{faq.get('faq_id', '-')}] "
                        f"Skor={faq.get('score', 0):.4f}"
                    )
                    print("     Pertanyaan:", faq.get("question", "-"))

            # Pemeriksaan dasar sesuai jenis input
            if not question.strip():
                test_passed = answer is None and len(results) == 0
                reason = "Input kosong ditangani dengan benar"

            elif item["expected_keyword"] is not None:
                keyword = item["expected_keyword"].lower()
                top_question = (
                    str(results[0].get("question", "")).lower()
                    if results else ""
                )
                test_passed = keyword in top_question
                reason = "FAQ teratas sesuai kata kunci yang diharapkan"

            else:
                # Untuk pertanyaan di luar topik, catat apakah
                # retriever masih mengembalikan FAQ.
                # Ini diagnostik, bukan otomatis dianggap lulus.
                test_passed = len(results) == 0
                reason = (
                    "Tidak ada hasil yang dikembalikan"
                    if test_passed
                    else "PERLU DITINJAU: FAQ mungkin tidak relevan"
                )

            status = "LULUS" if test_passed else "PERIKSA"
            print(f"Status: {status} — {reason}")

            if test_passed:
                passed += 1
            else:
                failed += 1

            records.append({
                "category": item["category"],
                "question": question,
                "answer": answer,
                "top_score": results[0].get("score") if results else None,
                "top_faq": results[0].get("question") if results else None,
                "result_count": len(results),
                "status": status,
            })

        except Exception as error:
            print("ERROR:", repr(error))
            failed += 1

            records.append({
                "category": item["category"],
                "question": question,
                "answer": None,
                "top_score": None,
                "top_faq": None,
                "result_count": 0,
                "status": "ERROR",
            })

    return passed, failed, records


# ============================================================
# 3. RINGKASAN PENGUJIAN
# ============================================================

def print_summary(passed, failed, records):
    total = passed + failed

    print("\n" + "=" * 70)
    print("3. RINGKASAN PENGUJIAN")
    print("=" * 70)

    print("Total pengujian:", total)
    print("Lulus:", passed)
    print("Perlu diperiksa:", failed)

    if total:
        print(f"Persentase pemeriksaan lulus: {passed / total * 100:.2f}%")

    print("\nCatatan:")
    print("- Pengujian ini bukan pengganti evaluasi model pada dataset uji.")
    print("- Pertanyaan di luar topik yang mengembalikan FAQ ditandai PERIKSA.")
    print("- Skor cosine similarity bukan probabilitas jawaban benar.")
    print("- Jangan mengubah threshold hanya berdasarkan beberapa pertanyaan.")

    print("\n" + "=" * 70)
    print("SELESAI")
    print("=" * 70)


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    info_ok = run_model_info_check()

    if not info_ok:
        print("\nPengujian dihentikan karena informasi model gagal dibaca.")
    else:
        passed, failed, records = run_question_checks()
        print_summary(passed, failed, records)
