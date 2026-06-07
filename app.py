import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title="News Classifier", page_icon="📰")

@st.cache_resource
def setup():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = setup()

stemmer = PorterStemmer()
LABELS = ["World", "Sports", "Business", "Tech"]

def preprocess(text):
    text = text.lower().replace('\n', ' ')
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text)
    stop = set(stopwords.words('english'))
    words = [stemmer.stem(w) for w in words if w not in stop]
    return ' '.join(words)

st.title("📰 News Classifier")
st.write("Enter a news headline or article snippet to classify it.")

text = st.text_area("News Text", height=150, placeholder="Type or paste news text here...")

if st.button("Classify", type="primary"):
    if text.strip():
        vec = vectorizer.transform([preprocess(text)])
        pred = model.predict(vec)[0]
        st.success(f"**Category: {LABELS[pred]}**")
    else:
        st.warning("Please enter some text.")