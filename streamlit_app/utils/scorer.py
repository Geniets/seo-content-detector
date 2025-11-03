"""
Model loading and scoring utilities for Streamlit app
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    """
    Load trained model and configuration.
    
    Returns:
        tuple: (model, feature_columns)
    """
    model_path = Path(__file__).parent.parent.parent / 'models' / 'quality_model.pkl'
    model = joblib.load(model_path)
    feature_columns = ['word_count', 'sentence_count', 'flesch_reading_ease']
    return model, feature_columns

def predict_quality(model, features, feature_columns):
    """
    Predict content quality.
    
    Args:
        model: Trained classifier
        features (dict): Feature dictionary
        feature_columns (list): List of feature names
    
    Returns:
        str: Quality label (Low/Medium/High)
    """
    # Create feature DataFrame
    feature_values = {
        'word_count': features.get('word_count', 0),
        'sentence_count': features.get('sentence_count', 0),
        'flesch_reading_ease': features.get('readability', 0)
    }
    
    X = pd.DataFrame([feature_values])[feature_columns]
    prediction = model.predict(X)[0]
    
    return prediction

def find_duplicates(embedding, threshold=0.80, existing_embeddings=None, existing_urls=None):
    """
    Find similar content based on embeddings.
    
    Args:
        embedding: Query embedding
        threshold (float): Similarity threshold
        existing_embeddings: Database of embeddings
        existing_urls: Corresponding URLs
    
    Returns:
        list: List of similar content dictionaries
    """
    if existing_embeddings is None or existing_urls is None:
        return []
    
    # Compute similarities
    similarities = cosine_similarity([embedding], existing_embeddings)[0]
    
    # Find matches above threshold
    similar_content = []
    for i, sim in enumerate(similarities):
        if sim > threshold:
            similar_content.append({
                'url': existing_urls[i],
                'similarity': round(float(sim), 4)
            })
    
    # Sort by similarity
    similar_content.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_content[:5]  # Return top 5
