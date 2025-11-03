# SEO Content Quality & Duplicate Detector

A machine learning pipeline for analyzing web content quality and detecting duplicate content using NLP techniques.

## Project Overview

This project implements a data science pipeline that:
- Parses HTML content and extracts clean text
- Engineers NLP features (readability, keywords, embeddings)
- Detects duplicate content using cosine similarity
- Classifies content quality (Low/Medium/High) with Random Forest

## Setup Instructions

```bash
git clone https://github.com/yourusername/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb
```

## Quick Start

1. Ensure dataset is in `data/data.csv`
2. Open and run `notebooks/seo_pipeline.ipynb`
3. View results in `data/` and `models/` folders

## Key Decisions

- **BeautifulSoup4** for robust HTML parsing
- **Sentence Transformers (all-MiniLM-L6-v2)** for semantic embeddings
- **Random Forest** for quality classification - handles non-linear relationships
- **Similarity threshold 0.80** - balances precision and recall for duplicate detection
- **70/30 train-test split** with stratification for reliable evaluation

## Results Summary

**Model Performance:**
- Accuracy: 0.85 (vs 0.64 baseline)
- F1-Score: 0.83
- Improvement: +33% over rule-based baseline

**Top Features:**
1. word_count (0.45)
2. flesch_reading_ease (0.32)
3. sentence_count (0.23)

**Content Analysis:**
- Duplicate pairs detected above 0.80 similarity
- Thin content flagged at <500 words
- Quality labels: High/Medium/Low

## Limitations

- JavaScript-rendered pages may not parse correctly
- Quality labels are rule-based, not human-annotated
- Optimized for English content only
- Small dataset limits generalization

## Technologies

Python 3.9+ • Pandas • Scikit-learn • BeautifulSoup4 • Sentence Transformers • NLTK • Textstat
