# SEO Content Quality & Duplicate Detector

A machine learning pipeline for analyzing web content quality and detecting duplicate content using NLP techniques.

**🚀 Live Demo:** [https://seo-content-detector-ymmew82uesk2nsyr6dcrgs.streamlit.app/](https://seo-content-detector-ymmew82uesk2nsyr6dcrgs.streamlit.app/)

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
- Accuracy: 1.0000 (vs 0.4762 baseline)
- F1-Score: 1.0000
- Improvement: +52% over rule-based baseline

**Top Features:**
1. flesch_reading_ease (0.48)
2. sentence_count (0.27)
3. word_count (0.26)

**Content Analysis:**
- Duplicate pairs detected above 0.80 similarity
- Thin content flagged at <500 words
- Quality labels: High/Medium/Low

## Web Application

Try the live Streamlit app to analyze any URL in real-time:
- Quality assessment (Low/Medium/High)
- Readability scores
- Duplicate detection
- Thin content identification

Access at: [https://seo-content-detector-ymmew82uesk2nsyr6dcrgs.streamlit.app/](https://seo-content-detector-ymmew82uesk2nsyr6dcrgs.streamlit.app/)

## Limitations

- JavaScript-rendered pages may not parse correctly
- Quality labels are rule-based, not human-annotated
- Optimized for English content only
- Small dataset limits generalization

## Technologies

Python 3.9+ • Pandas • Scikit-learn • BeautifulSoup4 • Sentence Transformers • NLTK • Textstat
