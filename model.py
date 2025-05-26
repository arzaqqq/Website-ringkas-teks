from transformers import pipeline, AutoTokenizer

# Inisialisasi BART summarizer dan tokenizer
try:
    summarizer = pipeline("summarization", model="gaduhhartawan/bart-indo-small")
    tokenizer = AutoTokenizer.from_pretrained("gaduhhartawan/bart-indo-small")
except Exception as e:
    print(f"Error loading model: {e}")
    summarizer = None
    tokenizer = None