ALGORITMA Summarize_With_BART
MASUKAN: teks_input (teks yang akan diringkas), target_sentence_count (jumlah kalimat target)
KELUARAN: ringkasan_teks, waktu_proses, jumlah_token, jumlah_kata, jumlah_kalimat

// 1. Inisialisasi
Inisialisasi waktu_mulai = waktu_sekarang()
Inisialisasi model_BART = muat_model("gaduhhartawan/indobart-base-v2")
Inisialisasi tokenizer = muat_tokenizer("gaduhhartawan/indobart-base-v2")

// 2. Tokenisasi dan Pemangkasan Teks
tokenized_input = tokenizer.encode(teks_input)
JIKA panjang(tokenized_input) > 1024 MAKA
    teks_input = tokenizer.decode(tokenized_input[1:1024])
AKHIR JIKA

// 3. Pra-pemrosesan Teks
teks_input = preprocess_text(teks_input, hapus_stopwords = Salah)

// 4. Tentukan Parameter Ringkasan
max_length = target_sentence_count * 300
min_length = target_sentence_count * 150

// 5. Generate Ringkasan Pertama
ringkasan = model_BART.generate(
    teks_input,
    do_sample = Salah,
    max_length = max_length,
    min_length = min_length,
    num_beams = 4,
    length_penalty = 1.0
)
ringkasan_teks = ringkasan[0]['summary_text']
kalimat = tokenize_sentences(ringkasan_teks)

// 6. Periksa Jumlah Kalimat
JIKA panjang(kalimat) < target_sentence_count MAKA
    max_length = max_length + 180
    min_length = min_length + 90
    ringkasan = model_BART.generate(
        teks_input,
        do_sample = Salah,
        max_length = max_length,
        min_length = min_length,
        num_beams = 4,
        length_penalty = 1.0
    )
    ringkasan_teks = ringkasan[0]['summary_text']
    kalimat = tokenize_sentences(ringkasan_teks)
AKHIR JIKA

// 7. Ambil Kalimat Sesuai Target
ringkasan_teks = gabung_kalimat(kalimat[1:target_sentence_count])

// 8. Hitung Metrik
jumlah_token = hitung_token(ringkasan_teks)
jumlah_kata = hitung_kata(ringkasan_teks)
jumlah_kalimat = hitung_kalimat(ringkasan_teks)
waktu_proses = waktu_sekarang() - waktu_mulai

// 9. Tangani Kesalahan
JIKA terjadi_kesalahan MAKA
    KEMBALIKAN "Error: pesan_kesalahan", 0, 0, 0, 0
AKHIR JIKA

// 10. Kembalikan Hasil
KEMBALIKAN ringkasan_teks, waktu_proses, jumlah_token, jumlah_kata, jumlah_kalimat
SELESAI