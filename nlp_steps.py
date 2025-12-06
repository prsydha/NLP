import streamlit as st
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("NLP Text Analysis with NLTK")
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk_data()

st.header("input text")
text = st.text_area("enter text to analyze:", height=300)

if st.button("Analyze") and text:
    st.divider()
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
        st.write(f"Number of words:{len(words)}")
        st.write(words)

    st.divider()

    st.header("Frequency Distribution")
        
    fdist = FreqDist(words)
    st.subheader("Most common words(Raw)")
    st.write(fdist.most_common(10))

    st.subheader("Frequency Plot(Raw)")
    fig, ax = plt.subplots()
    fdist.plot(10, show=False)
    st.pyplot(plt)
    plt.clf()
    st.divider()

    #Cleaning Text

    st.header("Text Cleaning")
    words_nopunc = [w.lower() for w in words if w.isalpha()]
    st.write(f"Words without punctuation: {len(words_nopunc)}")
    # st.write(words_nopunc) #Optional: don't show all if too many

    st.subheader("Frequency Plot(No Punctuation)")
    fdist_nopunc = FreqDist(words_nopunc)
    fdist_nopunc.plot(10, show=False)
    st.pyplot(plt)
    plt.clf()

    # stop words removal

    st.subheader("Stopwords Removal")
    stop_words = set(stopwords.words('english'))
    words_clean = [w for w in words_nopunc if w not in stop_words]
    st.write(f"Cleaned words(no stopwords): {len(words_clean)}")
    st.write(words_clean)
    st.divider()

    # word cloud generation

    st.header("Word Cloud")
    if words_clean:
        wordcloud_text = " ".join(words_clean)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.warning("No words left after cleaning to generate Word Cloud.")

