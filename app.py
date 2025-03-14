# app.py
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from newspaper import Article
import nltk
from summa import summarizer as textrank_summarizer
import os

# Download nltk resources (jika diperlukan)
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Initialize BART summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_article_content(url):
    """Ekstrak konten artikel dari URL berita"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        return None, f"Error extracting content: {str(e)}"

def summarize_with_bart(text, max_length=150, min_length=50):
    """Ringkas teks menggunakan BART"""
    if len(text.split()) < min_length:
        return text
    
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Error summarizing with BART: {str(e)}"

def summarize_with_textrank(text, ratio=0.2):
    """Ringkas teks menggunakan TextRank"""
    try:
        summary = textrank_summarizer.summarize(text, ratio=ratio)
        if not summary:
            return "Teks terlalu pendek untuk diringkas dengan TextRank"
        return summary
    except Exception as e:
        return f"Error summarizing with TextRank: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    title = ""
    original_text = ""
    bart_summary = ""
    textrank_summary = ""
    url = ""
    
    if request.method == 'POST':
        url = request.form.get('url', '')
        if url:
            title, original_text = get_article_content(url)
            if original_text and not original_text.startswith("Error"):
                bart_summary = summarize_with_bart(original_text)
                textrank_summary = summarize_with_textrank(original_text)
            
    return render_template('index.html', 
                          title=title,
                          original_text=original_text,
                          bart_summary=bart_summary,
                          textrank_summary=textrank_summary,
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