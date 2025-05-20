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
    # Tokenize the text and limit to 1024 tokens for BART
    start_time = time.time()
    tokens = tokenizer.encode(text)
    max_input_length = 1024
    if len(tokens) > max_input_length:
        text = tokenizer.decode(tokens[:max_input_length])  # Trim the text to fit within BART's token limit
    
    # Proses teks sebelum melakukan ringkasan
    text = preprocess_text(text)
    
    try:
        # Ringkasan BART dengan pengaturan default
        processing_time = time.time() - start_time
        summary = summarizer(text, do_sample=False)
        return summary[0]['summary_text'], round(processing_time, 2)
    except Exception as e:
        return f"Error summarizing with BART: {str(e)}"

def summarize_with_textrank(text):
    """Ringkas teks menggunakan TextRank"""
    start_time = time.time()
    try:
        # Proses teks sebelum melakukan ringkasan
        text = preprocess_text(text)
        processing_time = time.time() - start_time
        
        # Ringkasan TextRank dengan pengaturan default (ratio 0.2)
        summary = textrank_summarizer.summarize(text)
        return summary, round(processing_time, 2)
        if not summary:
            return "Teks terlalu pendek untuk diringkas dengan TextRank"
        return summary
    except Exception as e:
        return f"Error summarizing with TextRank: {str(e)}"
def count_words(text):
    """Menghitung jumlah kata dalam teks"""
    return len(text.split())

@app.route('/', methods=['GET', 'POST'])
def index():
    title = ""
    original_text = ""
    bart_summary = ""
    textrank_summary = ""
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
                    bart_summary = summarize_with_bart(original_text)
                    textrank_summary = summarize_with_textrank(original_text)
            
    return render_template('index.html', 
                          title=title,
                          original_text=original_text,
                          bart_summary=bart_summary,
                          textrank_summary=textrank_summary,
                          word_count=word_count,
                          warning_message=warning_message,
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


