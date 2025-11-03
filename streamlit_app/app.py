"""
SEO Content Quality Analyzer - Streamlit App

This is a bonus feature for the assignment.
Deploy this to Streamlit Cloud for +15 points.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.parser import parse_html_content
from utils.features import extract_features
from utils.scorer import load_model, predict_quality, find_duplicates

# Page configuration
st.set_page_config(
    page_title="SEO Content Quality Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 SEO Content Quality & Duplicate Detector")
st.markdown("""
Analyze web content for:
- **Quality Assessment** (Low/Medium/High)
- **Duplicate Detection**
- **Thin Content Identification**
- **Readability Scores**
""")

# Sidebar
st.sidebar.header("Configuration")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.5,
    max_value=1.0,
    value=0.80,
    step=0.05,
    help="Threshold for duplicate detection (higher = stricter)"
)

thin_content_threshold = st.sidebar.number_input(
    "Thin Content Threshold (words)",
    min_value=100,
    max_value=1000,
    value=500,
    step=50,
    help="Minimum word count for quality content"
)

# Load model
@st.cache_resource
def load_models():
    """Load ML models and existing content database"""
    model, feature_columns = load_model()
    return model, feature_columns

try:
    model, feature_columns = load_models()
    st.sidebar.success("✅ Models loaded")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.stop()

# Main content
tab1, tab2, tab3 = st.tabs(["Single URL Analysis", "Batch Analysis", "About"])

# Tab 1: Single URL Analysis
with tab1:
    st.header("Analyze Single URL")
    
    url = st.text_input(
        "Enter URL to analyze:",
        placeholder="https://example.com/article",
        help="Enter a valid URL to analyze its content quality"
    )
    
    analyze_button = st.button("🔍 Analyze", type="primary")
    
    if analyze_button and url:
        with st.spinner("Analyzing content..."):
            try:
                # Scrape and parse
                from utils.parser import scrape_url
                html_content = scrape_url(url)
                parsed = parse_html_content(html_content)
                
                if parsed['word_count'] == 0:
                    st.error("❌ Failed to extract content from URL")
                else:
                    # Extract features
                    features = extract_features(parsed['body_text'])
                    
                    # Predict quality
                    quality = predict_quality(model, features, feature_columns)
                    
                    # Find duplicates
                    similar_content = find_duplicates(
                        features['embedding'],
                        threshold=similarity_threshold
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Word Count", parsed['word_count'])
                        
                    with col2:
                        st.metric("Readability Score", f"{features['readability']:.1f}")
                        
                    with col3:
                        quality_color = {
                            'High': '🟢',
                            'Medium': '🟡',
                            'Low': '🔴'
                        }
                        st.metric("Quality", f"{quality_color.get(quality, '⚪')} {quality}")
                    
                    # Content details
                    st.subheader("Content Details")
                    st.write(f"**Title:** {parsed['title']}")
                    st.write(f"**Sentence Count:** {features['sentence_count']}")
                    st.write(f"**Top Keywords:** {features['keywords']}")
                    
                    # Thin content warning
                    if parsed['word_count'] < thin_content_threshold:
                        st.warning(f"⚠️ Thin content detected! ({parsed['word_count']} < {thin_content_threshold} words)")
                    
                    # Similar content
                    if similar_content:
                        st.subheader("🔍 Similar Content Found")
                        df_similar = pd.DataFrame(similar_content)
                        st.dataframe(df_similar, use_container_width=True)
                    else:
                        st.info("✅ No duplicate content detected")
                    
                    # JSON output
                    with st.expander("View Raw JSON"):
                        result = {
                            'url': url,
                            'title': parsed['title'],
                            'word_count': parsed['word_count'],
                            'readability': features['readability'],
                            'quality': quality,
                            'is_thin': parsed['word_count'] < thin_content_threshold,
                            'similar_content': similar_content
                        }
                        st.json(result)
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif analyze_button:
        st.warning("⚠️ Please enter a URL")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch URL Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with URLs",
        type=['csv'],
        help="CSV should have a column named 'url'"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'url' not in df.columns:
            st.error("❌ CSV must have a 'url' column")
        else:
            st.write(f"Found {len(df)} URLs")
            
            if st.button("🚀 Analyze All", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    try:
                        # Process each URL
                        url = row['url']
                        html_content = scrape_url(url)
                        parsed = parse_html_content(html_content)
                        features = extract_features(parsed['body_text'])
                        quality = predict_quality(model, features, feature_columns)
                        
                        results.append({
                            'url': url,
                            'word_count': parsed['word_count'],
                            'readability': features['readability'],
                            'quality': quality,
                            'is_thin': parsed['word_count'] < thin_content_threshold
                        })
                    except:
                        results.append({
                            'url': url,
                            'error': 'Failed to process'
                        })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "seo_analysis_results.csv",
                    "text/csv"
                )

# Tab 3: About
with tab3:
    st.header("About This Tool")
    
    st.markdown("""
    ### SEO Content Quality & Duplicate Detector
    
    This tool analyzes web content using machine learning and NLP techniques to:
    
    **Features:**
    - ✅ Parse HTML content and extract clean text
    - ✅ Calculate readability metrics (Flesch Reading Ease)
    - ✅ Extract top keywords using TF-IDF
    - ✅ Generate semantic embeddings for similarity detection
    - ✅ Classify content quality (Low/Medium/High)
    - ✅ Detect duplicate or similar content
    - ✅ Identify thin content
    
    **Technologies:**
    - Python 3.9+
    - Streamlit
    - Scikit-learn
    - Sentence Transformers
    - BeautifulSoup4
    
    **Model Details:**
    - **Classifier:** Random Forest
    - **Features:** word_count, sentence_count, flesch_reading_ease
    - **Embedding Model:** all-MiniLM-L6-v2
    
    ---
    
    **Created for:** Data Science Assignment
    
    **Repository:** [GitHub Link](https://github.com/yourusername/seo-content-detector)
    """)
    
    st.info("""
    **💡 Tip:** For best results, ensure URLs are publicly accessible and contain 
    meaningful text content (not just images or videos).
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit | "
    "<a href='https://github.com/yourusername/seo-content-detector'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)
