from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer
from newspaper import Article
import nltk
from summa import summarizer as textrank_summarizer
import re
import time

# Download nltk resources (jika diperlukan)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Initialize BART summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def preprocess_text(text):
    """Pemrosesan teks dasar untuk menghapus karakter yang tidak perlu"""
    # Hapus karakter yang bukan huruf, angka, spasi, tanda baca (kecuali - dan /)
    text = re.sub(r'[^A-Za-z0-9\s.,!?\'"-\/]', '', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_article_content(url):
    """Ekstrak konten artikel dari URL berita"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        return None, f"Error extracting content: {str(e)}"

def count_words(text):
    """Menghitung jumlah kata dalam teks"""
    return len(text.split())

def summarize_with_bart(text):
    """Ringkas teks menggunakan BART"""
    start_time = time.time()
    try:
        # Tokenize dan preprocessing
        tokens = tokenizer.encode(text)
        if len(tokens) > 1024:
            text = tokenizer.decode(tokens[:1024])
        text = preprocess_text(text)
        
        # Proses summarization
        summary = summarizer(text, do_sample=False)
        
        # Hitung waktu SETELAH proses
        processing_time = time.time() - start_time
        return summary[0]['summary_text'], round(processing_time, 2)
    except Exception as e:
        return f"Error: {str(e)}", 0

def summarize_with_textrank(text):
    """Ringkas teks menggunakan TextRank"""
    start_time = time.time()
    try:
        text = preprocess_text(text)
        
        # Proses summarization
        summary = textrank_summarizer.summarize(text)
        
        # Hitung waktu SETELAH proses
        processing_time = time.time() - start_time
        return summary, round(processing_time, 2)
    except Exception as e:
        return f"Error: {str(e)}", 0
def count_words(text):
    """Menghitung jumlah kata dalam teks"""
    return len(text.split())

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
    
    if request.method == 'POST':
        url = request.form.get('url', '')
        if url:
            title, original_text = get_article_content(url)
            if original_text and not original_text.startswith("Error"):
                word_count = count_words(original_text)
                if word_count < 150:
                    warning_message = "Teks terlalu pendek (kurang dari 150 kata)."
                elif word_count > 600:
                    warning_message = "Teks terlalu panjang (lebih dari 600 kata)."
                else:
                    bart_summary, bart_time = summarize_with_bart(original_text)
                    textrank_summary, textrank_time = summarize_with_textrank(original_text)
            
    return render_template('index.html', 
                          title=title,
                          original_text=original_text,
                          bart_summary=bart_summary,
                          textrank_summary=textrank_summary,
                          word_count=word_count,
                          warning_message=warning_message,
                          bart_time=bart_time,
                          textrank_time=textrank_time,
                          url=url)


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
    elif word_count > 600:
        return jsonify({"error": "Teks terlalu panjang (lebih dari 600 kata)."}), 400
    
    bart_summary = summarize_with_bart(original_text)
    textrank_summary = summarize_with_textrank(original_text)
    
    return jsonify({
        "title": title,
        "original_text": original_text,
        "bart_summary": bart_summary,
        "textrank_summary": textrank_summary
    })

if __name__ == '__main__':
    app.run(debug=True)


