"""
HTML Parser utilities for Streamlit app
"""

import requests
from bs4 import BeautifulSoup
import re

def scrape_url(url, timeout=10):
    """
    Scrape HTML content from URL.
    
    Args:
        url (str): URL to scrape
        timeout (int): Request timeout in seconds
    
    Returns:
        str: HTML content
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text

def parse_html_content(html_content):
    """
    Parse HTML content and extract title and body text.
    
    Args:
        html_content (str): Raw HTML content
    
    Returns:
        dict: Dictionary with title, body_text, and word_count
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract title
        title = soup.find('title')
        title = title.get_text().strip() if title else 'No Title'
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link", "noscript"]):
            script.decompose()
        
        # Extract text from main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            body_text = ' '.join([elem.get_text().strip() for elem in text_elements])
        else:
            body_text = soup.get_text()
        
        # Clean the text
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        body_text = re.sub(r'[^\w\s.,!?;:\'-]', '', body_text)
        
        # Calculate word count
        word_count = len(body_text.split())
        
        return {
            'title': title,
            'body_text': body_text,
            'word_count': word_count
        }
    
    except Exception as e:
        return {
            'title': 'Error',
            'body_text': '',
            'word_count': 0
        }
