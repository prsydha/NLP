import streamlit as st
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import requests

# Set page configuration
st.set_page_config(page_title="Nepali NLP Text Analysis", layout="wide")

st.title("Nepali NLP Text Analysis")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')

download_nltk_data()

# Nepali Stopwords
@st.cache_data
def get_nepali_stopwords():
    # Hardcoded fallback list
    fallback_stopwords = {
        "छ", "र", "पनि", "छन्", "लागि", "भएको", "गरेको", "भने", "गर्न", "गर्ने",
        "हो", "तथा", "यो", "रहेको", "उनले", "थियो", "हुने", "गरेका", "नै", "यस",
        "अझै", "अधिक", "अन्य", "आदि", "कति", "कहिले", "कसरी", "किन", "जब", "जसको",
        "जसले", "जहाँ", "त", "तिनी", "तिनीहरू", "तपाईं", "तपाईंको", "धेरै", "न",
        "नि", "नेपाल", "पछी", "फेरी", "बारे", "बाट", "मात्र", "माथि", "मेरो",
        "राम्रो", "रूप", "लाई", "ले", "वरिपरि", "संग", "सबै", "सधैं", "सम्म", "साथ",
        "हामी", "हाम्रो", "हुन्", "हुन्छ", "हुन्थे", "होइन", "हैन"
    }
    
    url = "https://raw.githubusercontent.com/prtx/Nepali-Stopwords/master/stopwords.txt"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Split by newline and remove empty strings
            external_stopwords = set(response.text.splitlines())
            # Combine with fallback to ensure we have at least the basics if the external list is weird
            # But usually external list is better. Let's just return external if successful.
            # Actually, merging might be safer to guarantee coverage of common words we know.
            combined_stopwords = external_stopwords.union(fallback_stopwords)
            return combined_stopwords
        else:
            st.warning(f"Failed to fetch external stopwords (Status: {response.status_code}). Using fallback list.")
            return fallback_stopwords
    except Exception as e:
        st.warning(f"Error fetching external stopwords: {e}. Using fallback list.")
        return fallback_stopwords

nepali_stopwords = get_nepali_stopwords()
st.info(f"Loaded {len(nepali_stopwords)} stopwords.")

# Default text (News Snippet)
default_text = '''
उत्तर कोरियाबाट सयौँ मानिसलाई भाग्न र दक्षिणतिर आउन सघाएको भनेर दक्षिण कोरियामा ती पादरीलाई नायक मान्दै आइएको थियो। 
एउटा अदालतले सउलस्थित आफ्नो आवासीय विद्यालयमा बसेका नाबालिगहरूको ति ६७ वर्षीय चुन कि-वनले यौनशोषण गरेको ठहर गरेको हो। 
उनले पाँच वर्ष कारागारमा बस्नुपर्ने छ। दशकौँसम्म उनलाई उद्धारक र संरक्षक मानिन्थ्यो। 
उनले उत्तर कोरियाबाट भागेका मानिसहरूलाई सघाउन गरेको कामको मानिसहरूले प्रशंसा गर्दै आएका थिए। 
उनी गत सेप्टेम्बर महिनामा सउलमा पक्राउ परेका थिए।
'''

# Text Input
st.header("Input Text (Nepali)")
text = st.text_area("Enter Nepali text to analyze:", value=default_text, height=300)

if st.button("Analyze") and text:
    st.divider()
    
    # Tokenization
    st.header("Tokenization")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentences")
        sentences = sent_tokenize(text)
        st.write(f"Number of sentences: {len(sentences)}")
        st.write(sentences)

    with col2:
        st.subheader("Words")
        words = word_tokenize(text)
        st.write(f"Number of words: {len(words)}")
        st.write(words)

    st.divider()

    # Frequency Distribution
    st.header("Frequency Distribution")
    
    fdist = FreqDist(words)
    st.subheader("Most Common Words (Raw)")
    st.write(fdist.most_common(10))

    st.subheader("Frequency Plot (Raw)")
    # Use a font that supports Devanagari for the plot if possible, 
    # but matplotlib default might not support it well without config.
    # We will try to plot, but be aware of potential boxes.
    fig, ax = plt.subplots()
    fdist.plot(10, show=False)
    st.pyplot(plt) 
    plt.clf()

    st.divider()

    # Cleaning Text
    st.header("Text Cleaning")
    
    # Simple cleaning: remove punctuation-like characters if possible, 
    # but for now just keep alphanumeric (which might exclude some nepali chars if not careful).
    # Better to just filter by length or known punctuation.
    # Let's try to keep everything that is not in a basic punctuation list.
    punctuation = "।,-?!()\"':;[]{}–"
    words_nopunc = [w for w in words if w not in punctuation]
    
    st.write(f"Words without punctuation: {len(words_nopunc)}")

    st.subheader("Frequency Plot (No Punctuation)")
    fdist_nopunc = FreqDist(words_nopunc)
    fdist_nopunc.plot(10, show=False)
    st.pyplot(plt)
    plt.clf()

    # Stopwords Removal
    st.subheader("Stopwords Removal")
    words_clean = [w for w in words_nopunc if w not in nepali_stopwords]
    st.write(f"Cleaned words (no stopwords): {len(words_clean)}")
    st.write(words_clean)

    st.divider()

    # Word Cloud
    st.header("Word Cloud")
    if words_clean:
        wordcloud_text = " ".join(words_clean)
        
        # Path to a Devanagari font
        font_path = '/System/Library/Fonts/Supplemental/DevanagariMT.ttc'
        
        # Check if font exists, otherwise fallback (might show boxes)
        if not os.path.exists(font_path):
             st.warning(f"Font not found at {font_path}. WordCloud might not render correctly.")
             font_path = None

        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            font_path=font_path,
            regexp=r"[\w']+" # Simple regex for tokenization in wordcloud
        ).generate(wordcloud_text)
        
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.warning("No words left after cleaning to generate Word Cloud.")
