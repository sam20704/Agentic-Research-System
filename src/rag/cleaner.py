import re

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    # remove page numbers
    text = re.sub(r"Page \d+ of \d+", "", text)
    
    # remove citations like "219." or "116"
    text = re.sub(r"\b\d{1,3}\.\s", "", text)
    
    # remove URLs
    text = re.sub(r"http\S+", "", text)
    
    return text.strip()