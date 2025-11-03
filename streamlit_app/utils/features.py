"""
Feature extraction utilities for Streamlit app
"""

import re
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Load model once
_model = None

def get_embedding_model():
    """Get or load sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def clean_text(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_sentences(text):
    """Count number of sentences in text."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

def get_readability_score(text):
    """Calculate Flesch Reading Ease score."""
    try:
        score = textstat.flesch_reading_ease(text)
        return score
    except:
        return 0

def extract_top_keywords(text, n=5):
    """Extract top n keywords using TF-IDF."""
    try:
        vectorizer = TfidfVectorizer(max_features=n, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        keywords = [kw for kw, score in keyword_scores[:n]]
        return '|'.join(keywords)
    except:
        return ''

def extract_features(body_text):
    """
    Extract all features from body text.
    
    Args:
        body_text (str): Clean body text
    
    Returns:
        dict: Dictionary of features
    """
    clean_text_content = clean_text(body_text)
    
    # Get embedding model
    model = get_embedding_model()
    
    features = {
        'clean_text': clean_text_content,
        'sentence_count': count_sentences(body_text),
        'readability': get_readability_score(body_text),
        'keywords': extract_top_keywords(body_text),
        'embedding': model.encode([clean_text_content])[0]
    }
    
    return features
