from transformers import pipeline, AutoTokenizer
from summa import summarizer as textrank_summarizer
import nltk
import re
import time
from flask import Flask, render_template, request, jsonify
from newspaper import Article
import requests
from bs4 import BeautifulSoup


# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Initialize BART summarizer and tokenizer
summarizer = pipeline("summarization", model="gaduhhartawan/indobart-base-v2", device=-1)
tokenizer = AutoTokenizer.from_pretrained("gaduhhartawan/indobart-base-v2")

# Daftar stopwords khusus untuk menghapus metadata berita, iklan, dan tanda --
custom_stopwords = [
    'ADVERTISEMENT', 'Liputan6.com', 'Jakarta-', ', Jakarta',
    'GambasVideo', 'CNN', 'Detik.com', 'Tribunnews.com', 
    'Baca Juga', 'Baca juga', 'Berita Terkait', 'Simak Juga', 
    '--', '---', '–', '—', 'SCROLL TO CONTINUE WITH CONTENT'
]

def preprocess_text(text, remove_stopwords=True):
    """Pemrosesan teks untuk menghapus karakter tidak perlu dan stopwords khusus"""
    text = re.sub(r'[^A-Za-z0-9\s.,!?\'"-\/]', '', text)
    if remove_stopwords:
        for stopword in custom_stopwords:
            pattern = re.escape(stopword)
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.,!?]{2,}', '', text)
        text = re.sub(r'\s*-\s*', ' ', text)
        text = text.strip()
    print(f"Text after preprocessing: {text[:200]}...")
    return text

# def get_article_content(url):
#     """Ekstrak konten artikel dari URL berita"""
#     try:
#         article = Article(url)
#         article.download()
#         article.parse()
#         return article.title, article.text
#     except Exception as e:
#         return None, f"Error: {str(e)}"



def get_article_content(url):
    """Scrape artikel berdasarkan URL. Gunakan BeautifulSoup jika situs tidak kompatibel dengan newspaper3k."""
    try:
        if "liputan6.com" in url:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Cari judul
            title_tag = soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else "Judul tidak ditemukan"

            # Ambil semua paragraf dari kontainer utama
            paragraphs = soup.select('div.article-content-body p')
            if not paragraphs:
                # fallback jika struktur berbeda
                paragraphs = soup.find_all('p')

            # Gabungkan isi teks
            content = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            if len(content.strip()) < 100:
                return None, "Konten artikel Liputan6 terlalu pendek atau gagal diproses."

            return title, content

        else:
            # === Gunakan newspaper untuk situs umum ===
            from newspaper import Article
            article = Article(url)
            article.download()
            article.parse()

            if not article.text.strip():
                return None, "Teks artikel kosong setelah parsing."

            return article.title, article.text

    except Exception as e:
        return None, f"Error saat scraping: {str(e)}"



def count_words(text):
    """Menghitung jumlah kata dalam teks"""
    return len(text.split())

def count_tokens(text):
    """Menghitung jumlah token dalam teks menggunakan tokenizer BART"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def count_sentences(text):
    """Menghitung jumlah kalimat dalam teks"""
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

def summarize_with_bart(text, target_sentence_count):
    """Ringkas teks menggunakan BART dengan panjang sesuai jumlah kalimat"""
    start_time = time.time()
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) > 1024:
            text = tokenizer.decode(tokens[:1024])
        text = preprocess_text(text, remove_stopwords=False)
        max_length = target_sentence_count * 150
        min_length = target_sentence_count * 50
        summary = summarizer(
            text, 
            do_sample=False, 
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=1.0
        )
        summary_text = summary[0]['summary_text']
        sentences = nltk.sent_tokenize(summary_text)
        if len(sentences) < target_sentence_count:
            summary = summarizer(
                text,
                do_sample=False,
                max_length=max_length + 180,
                min_length=min_length + 90,
                num_beams=4
            )
            summary_text = summary[0]['summary_text']
            sentences = nltk.sent_tokenize(summary_text)
        summary_text = " ".join(sentences[:target_sentence_count])
        token_count = count_tokens(summary_text)
        word_count = count_words(summary_text)
        sentence_count = count_sentences(summary_text)
        processing_time = time.time() - start_time
        return summary_text, round(processing_time, 2), token_count, word_count, sentence_count
    except Exception as e:
        return f"Error: {str(e)}", 0, 0, 0, 0

def summarize_with_textrank(text, target_sentence_count):
    """Ringkas teks menggunakan TextRank dengan jumlah kalimat sesuai input"""
    start_time = time.time()
    try:
        text = preprocess_text(text, remove_stopwords=True)
        print(f"Preprocessed text for TextRank: {text[:200]}...")
        
        # Gunakan rasio awal untuk mendapatkan ringkasan awal
        summary = textrank_summarizer.summarize(text, ratio=0.5, scores=True)
        if not summary:
            summary = textrank_summarizer.summarize(text, ratio=0.7, scores=True)
        
        # Urutkan kalimat berdasarkan skor TextRank
        sentences = [(sentence, score) for sentence, score in summary]
        sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
        print(f"Initial TextRank sentences: {[s[0] for s in sentences]}, Words: {count_words(' '.join(s[0] for s in sentences))}")
        
        # Pilih jumlah kalimat sesuai target_sentence_count
        selected_sentences = [sentence for sentence, _ in sentences[:target_sentence_count]]
        
        # Jika jumlah kalimat kurang dari target, tambahkan kalimat dari teks asli
        if len(selected_sentences) < target_sentence_count:
            original_sentences = nltk.sent_tokenize(text)
            original_sentences = sorted(original_sentences, key=lambda x: count_words(x))
            for sentence in original_sentences:
                if sentence not in selected_sentences:
                    selected_sentences.append(sentence)
                    if len(selected_sentences) >= target_sentence_count:
                        break
        
        # Gabungkan kalimat yang dipilih
        adjusted_summary = " ".join(selected_sentences)
        
        # Pastikan ringkasan berakhir pada kalimat lengkap
        if adjusted_summary:
            sentences = nltk.sent_tokenize(adjusted_summary)
            if sentences:
                adjusted_summary = " ".join(sentences[:target_sentence_count])
        
        print(f"Adjusted TextRank summary: {adjusted_summary}, Sentences: {count_sentences(adjusted_summary)}, Words: {count_words(adjusted_summary)}, Tokens: {count_tokens(adjusted_summary)}")
        processing_time = time.time() - start_time
        final_word_count = count_words(adjusted_summary)
        final_token_count = count_tokens(adjusted_summary)
        final_sentence_count = count_sentences(adjusted_summary)
        return adjusted_summary, round(processing_time, 2), final_token_count, final_word_count, final_sentence_count
    except Exception as e:
        return f"Error: {str(e)}", 0, 0, 0, 0

@app.route('/', methods=['GET', 'POST'])
def index():
    title = ""
    original_text = ""
    bart_summary = ""
    textrank_summary = ""
    bart_time = 0
    textrank_time = 0
    url = ""
    word_count = 0
    sentence_count = 0
    warning_message = ""
    bart_token_count = 0
    textrank_token_count = 0
    bart_word_count = 0
    textrank_word_count = 0
    bart_sentence_count = 0
    textrank_sentence_count = 0
    selected_sentence_count = 3  # Default value
    if request.method == 'POST':
        url = request.form.get('url', '')
        selected_sentence_count = int(request.form.get('sentence_count', 3))
        if url:
            title, original_text = get_article_content(url)
            if original_text and not original_text.startswith("Error"):
                word_count = count_words(original_text)
                sentence_count = count_sentences(original_text)
                if word_count < 10:
                    warning_message = f"Teks terlalu pendek (kurang dari 10 kata, hanya {word_count} kata)."
                elif word_count > 400:
                    warning_message = f"Teks terlalu panjang (lebih dari 400 kata, yaitu {word_count} kata)."
                elif sentence_count < selected_sentence_count:
                    warning_message = f"Teks memiliki terlalu sedikit kalimat (kurang dari {selected_sentence_count} kalimat)."
                else:
                    bart_summary, bart_time, bart_token_count, bart_word_count, bart_sentence_count = summarize_with_bart(original_text, selected_sentence_count)
                    textrank_summary, textrank_time, textrank_token_count, textrank_word_count, textrank_sentence_count = summarize_with_textrank(
                        original_text, target_sentence_count=selected_sentence_count
                    )
    return render_template('index.html', 
                          title=title,
                          original_text=original_text,
                          bart_summary=bart_summary,
                          textrank_summary=textrank_summary,
                          word_count=word_count,
                          sentence_count=sentence_count,
                          warning_message=warning_message,
                          bart_time=bart_time,
                          textrank_time=textrank_time,
                          url=url,
                          bart_token_count=bart_token_count,
                          textrank_token_count=textrank_token_count,
                          bart_word_count=bart_word_count,
                          textrank_word_count=textrank_word_count,
                          bart_sentence_count=bart_sentence_count,
                          textrank_sentence_count=textrank_sentence_count,
                          selected_sentence_count=selected_sentence_count)

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.json
    url = data.get('url', '')
    selected_sentence_count = int(data.get('sentence_count', 3))
    if not url:
        return jsonify({"error": "URL is required"}), 400
    title, original_text = get_article_content(url)
    if not original_text or original_text.startswith("Error"):
        return jsonify({"error": original_text}), 400
    word_count = count_words(original_text)
    sentence_count = count_sentences(original_text)
    if word_count < 10:
        return jsonify({"error": f"Teks terlalu pendek (kurang dari 10 kata, hanya {word_count} kata)."}), 400
    elif word_count > 400:
        return jsonify({"error": f"Teks terlalu panjang (lebih dari 400 kata, yaitu {word_count} kata)."}), 400
    elif sentence_count < selected_sentence_count:
        return jsonify({"error": f"Teks memiliki terlalu sedikit kalimat (kurang dari {selected_sentence_count} kalimat)."}), 400
    bart_summary, _, bart_token_count, bart_word_count, bart_sentence_count = summarize_with_bart(original_text, selected_sentence_count)
    textrank_summary, _, textrank_token_count, textrank_word_count, textrank_sentence_count = summarize_with_textrank(
        original_text, target_sentence_count=selected_sentence_count
    )
    return jsonify({
        "title": title,
        "original_text": original_text,
        "bart_summary": bart_summary,
        "textrank_summary": textrank_summary,
        "bart_token_count": bart_token_count,
        "textrank_token_count": textrank_token_count,
        "bart_word_count": bart_word_count,
        "textrank_word_count": textrank_word_count,
        "bart_sentence_count": bart_sentence_count,
        "textrank_sentence_count": textrank_sentence_count,
        "word_count": word_count,
        "sentence_count": sentence_count
    })

if __name__ == '__main__':
    app.run(debug=True)