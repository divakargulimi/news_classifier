import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from datasets import load_dataset
df = load_dataset("AG_news")

import pandas as pd
train = df['train']
test = df['test']

train = pd.DataFrame(train)
test = pd.DataFrame(test)

X_train = train['text']
y_train = train['label']

X_test = test['text']
y_test = test['label']

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    return text

X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

import re
def only_alpha(text):
  return re.sub('[^a-zA-Z]', ' ', text)

X_train = X_train.apply(only_alpha)
X_test = X_test.apply(only_alpha)

stemmer = PorterStemmer()
for i in range(len(X_train.index)):
  words = nltk.word_tokenize(X_train[i])
  words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
  X_train[i] = ' '.join(words)

for i in range(len(X_test.index)):
  words = nltk.word_tokenize(X_test[i])
  words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
  X_test[i] = ' '.join(words)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000)

xtrain = tfidf.fit_transform(X_train)
xtest = tfidf.transform(X_test)

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(xtrain, y_train)

prediction = model.predict(xtest)

import joblib

# Save
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction))
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))

world_samples = [
    "The president met with foreign leaders to discuss global security issues",
    "A new peace agreement was signed between the two countries",
    "The United Nations held an emergency meeting on climate change",
    "Diplomatic relations between the nations have improved",
    "The war in the region has caused a humanitarian crisis"
]

sports_samples = [
    "The team won the championship after a thrilling final match",
    "The player scored a hat-trick in yesterday’s game",
    "The Olympics will feature new sporting events this year",
    "The coach praised the players for their performance",
    "The football match ended in a draw"
]

business_samples = [
    "The stock market saw a significant increase today",
    "The company reported higher profits this quarter",
    "Investors are concerned about rising inflation",
    "The startup raised millions in funding",
    "The CEO announced a merger with a rival firm"
]

tech_samples = [
    "The new smartphone features advanced AI capabilities",
    "Researchers developed a breakthrough in quantum computing",
    "The software update improves system performance",
    "Cybersecurity threats are increasing globally",
    "The company launched a new artificial intelligence model"
]

test_sentences = world_samples + sports_samples + business_samples + tech_samples

# Reconstruct the original list of test sentences (it might have been overwritten in memory during the error)
test_sentences_raw = world_samples + sports_samples + business_samples + tech_samples

# Re-initialize and re-fit the TFIDF vectorizer locally within this cell
# This ensures consistency, as the global 'tfidf' object might have become inconsistent.
from sklearn.feature_extraction.text import TfidfVectorizer
local_tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
local_tfidf.fit(X_train) # Fit on the original training data X_train

# Apply the same preprocessing steps as used for X_train and X_test
processed_test_sentences = []
for sentence in test_sentences_raw:
    # Clean text: lowercase and replace newlines
    cleaned_text = clean_text(sentence)
    # Keep only alphabetic characters
    alpha_only_text = only_alpha(cleaned_text)
    # Tokenize words, stem, and remove stopwords
    words = nltk.word_tokenize(alpha_only_text)
    filtered_words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    processed_test_sentences.append(' '.join(filtered_words))

# Transform the preprocessed test sentences using the locally fitted tfidf vectorizer
test_sentences = local_tfidf.transform(processed_test_sentences)

# We can also verify the number of features created:
print(f"Number of features in transformed test_sentences: {test_sentences.shape[1]}")
print(f"Number of features expected by the model: {model.n_features_in_}")
extract_label = ["politics","sports","business","tech"]
lab = model.predict(test_sentences[:])
for i in lab:
  print(extract_label[i])