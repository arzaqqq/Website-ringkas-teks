from transformers import pipeline, AutoTokenizer
from summa import summarizer as textrank_summarizer
import nltk
import re
import time
from flask import Flask, render_template, request, jsonify
from newspaper import Article

# Download nltk resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

# Initialize BART summarizer and tokenizer
summarizer = pipeline("summarization", model="gaduhhartawan/indobart-base-v2")
tokenizer = AutoTokenizer.from_pretrained("gaduhhartawan/indobart-base-v2")

# Daftar stopwords khusus untuk menghapus metadata berita, iklan, dan tanda --
custom_stopwords = [
    'ADVERTISEMENT', 'Liputan6.com', 'KOMPAS.com', 'Jakarta-', 
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

def get_article_content(url):
    """Ekstrak konten artikel dari URL berita"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        return None, f"Error: {str(e)}"

def count_words(text):
    """Menghitung jumlah kata dalam teks"""
    return len(text.split())

def count_tokens(text):
    """Menghitung jumlah token dalam teks menggunakan tokenizer BART"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def summarize_with_bart(text):
    """Ringkas teks menggunakan BART"""
    start_time = time.time()
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) > 1024:
            text = tokenizer.decode(tokens[:1024])
        text = preprocess_text(text, remove_stopwords=False)
        summary = summarizer(
            text, 
            do_sample=False, 
            max_length=200,
            min_length=80
        )
        summary_text = summary[0]['summary_text']
        token_count = count_tokens(summary_text)
        word_count = count_words(summary_text)
        processing_time = time.time() - start_time
        return summary_text, round(processing_time, 2), token_count, word_count
    except Exception as e:
        return f"Error: {str(e)}", 0, 0, 0

def summarize_with_textrank(text, target_word_count):
    """Ringkas teks menggunakan TextRank dengan jumlah kata sesuai BART"""
    start_time = time.time()
    try:
        text = preprocess_text(text, remove_stopwords=True)
        print(f"Preprocessed text for TextRank: {text[:200]}...")
        
        total_words = count_words(text)
        ratio = min(target_word_count / total_words * 1.5, 1.0) if total_words > 0 else 0.5
        
        summary = textrank_summarizer.summarize(text, ratio=ratio, scores=True)
        if not summary:
            summary = textrank_summarizer.summarize(text, ratio=0.5, scores=True)
        
        # Urutkan kalimat berdasarkan skor TextRank
        sentences = [(sentence, score) for sentence, score in summary]
        sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
        print(f"Initial TextRank sentences: {[s[0] for s in sentences]}, Words: {count_words(' '.join(s[0] for s in sentences))}")
        
        # Pilih kalimat hingga mendekati jumlah kata target
        adjusted_summary = ""
        current_words = 0
        selected_sentences = []
        for sentence, _ in sentences:
            sentence_words = count_words(sentence)
            if current_words + sentence_words <= target_word_count * 1.2:  # Toleransi 20%
                selected_sentences.append(sentence)
                current_words += sentence_words
            else:
                break
        
        # Jika jumlah kata kurang, tambahkan kalimat dari teks asli
        if current_words < target_word_count:
            original_sentences = nltk.sent_tokenize(text)
            original_sentences = sorted(original_sentences, key=lambda x: count_words(x))
            for sentence in original_sentences:
                if sentence not in selected_sentences:
                    sentence_words = count_words(sentence)
                    if current_words + sentence_words <= target_word_count * 1.2:
                        selected_sentences.append(sentence)
                        current_words += sentence_words
                    if current_words >= target_word_count * 0.9:  # Minimal 90% dari target
                        break
        
        # Gabungkan kalimat yang dipilih
        adjusted_summary = " ".join(selected_sentences)
        
        # Potong jika melebihi target kata
        if current_words > target_word_count:
            sentences = nltk.sent_tokenize(adjusted_summary)
            adjusted_summary = ""
            current_words = 0
            for sentence in sentences:
                sentence_words = count_words(sentence)
                if current_words + sentence_words <= target_word_count:
                    adjusted_summary += sentence + " "
                    current_words += sentence_words
                else:
                    break
            adjusted_summary = adjusted_summary.strip()
        
        # Pastikan ringkasan berakhir pada kalimat lengkap
        if adjusted_summary:
            sentences = nltk.sent_tokenize(adjusted_summary)
            if sentences:
                adjusted_summary = " ".join(sentences)
        
        print(f"Adjusted TextRank summary: {adjusted_summary}, Words: {count_words(adjusted_summary)}, Tokens: {count_tokens(adjusted_summary)}")
        processing_time = time.time() - start_time
        final_word_count = count_words(adjusted_summary)
        final_token_count = count_tokens(adjusted_summary)
        return adjusted_summary, round(processing_time, 2), final_token_count, final_word_count
    except Exception as e:
        return f"Error: {str(e)}", 0, 0, 0

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
    warning_message = ""
    bart_token_count = 0
    textrank_token_count = 0
    bart_word_count = 0
    textrank_word_count = 0
    if request.method == 'POST':
        url = request.form.get('url', '')
        if url:
            title, original_text = get_article_content(url)
            if original_text and not original_text.startswith("Error"):
                word_count = count_words(original_text)
                if word_count < 150:
                    warning_message = "Teks terlalu pendek (kurang dari 150 kata)."
                elif word_count > 400:
                    warning_message = "Teks terlalu panjang (lebih dari 400 kata)."
                else:
                    bart_summary, bart_time, bart_token_count, bart_word_count = summarize_with_bart(original_text)
                    textrank_summary, textrank_time, textrank_token_count, textrank_word_count = summarize_with_textrank(
                        original_text, target_word_count=bart_word_count
                    )
    return render_template('index.html', 
                          title=title,
                          original_text=original_text,
                          bart_summary=bart_summary,
                          textrank_summary=textrank_summary,
                          word_count=word_count,
                          warning_message=warning_message,
                          bart_time=bart_time,
                          textrank_time=textrank_time,
                          url=url,
                          bart_token_count=bart_token_count,
                          textrank_token_count=textrank_token_count,
                          bart_word_count=bart_word_count,
                          textrank_word_count=textrank_word_count)

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({"error": "URL is required"}), 400
    title, original_text = get_article_content(url)
    if not original_text or original_text.startswith("Error"):
        return jsonify({"error": original_text}), 400
    word_count = count_words(original_text)
    if word_count < 150:
        return jsonify({"error": "Teks terlalu pendek (kurang dari 150 kata)."}), 400
    elif word_count > 400:
        return jsonify({"error": "Teks terlalu panjang (lebih dari 400 kata)."}), 400
    bart_summary, _, bart_token_count, bart_word_count = summarize_with_bart(original_text)
    textrank_summary, _, textrank_token_count, textrank_word_count = summarize_with_textrank(
        original_text, target_word_count=bart_word_count
    )
    return jsonify({
        "title": title,
        "original_text": original_text,
        "bart_summary": bart_summary,
        "textrank_summary": textrank_summary,
        "bart_token_count": bart_token_count,
        "textrank_token_count": textrank_token_count,
        "bart_word_count": bart_word_count,
        "textrank_word_count": textrank_word_count
    })

if __name__ == '__main__':
    app.run(debug=True)