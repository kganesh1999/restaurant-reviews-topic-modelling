# NLP
from nltk.tokenize.regexp import RegexpTokenizer
from subprocess import check_output
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import email, re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# Function to preprocess review column
def text_prep_func(text):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    porter = PorterStemmer()
    text = text.rstrip()
    text = re.sub(r'[^a-zA-Z]',' ',text)
    stop_free = " ".join([i for i in text.lower().split() if ((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    cleaned_text = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return cleaned_text

def get_topic_keywords_yearly(dataset, year, topic):
    inputTuple = dict()
    #Get doc_corpus
    reviews = dataset[f'reviews{year}']['prep_review'][dataset[f'reviews{year}']['cluster_label']==topic].tolist()
    cv=CountVectorizer(max_df=0.85, stop_words = set(stopwords.words('english')) )
    word_count_vector = cv.fit_transform(reviews)
    feature_names = cv.get_feature_names()
    tfidf = TfidfTransformer(smooth_idf = True, use_idf = True)
    tfidf_vector = tfidf.fit(word_count_vector)
    # Extracting keywords from each review over the topic
    for review in reviews:
        review_vector=tfidf.transform(cv.transform([review]))
        review_vector = review_vector.tocoo()
        tuples = zip(review_vector.col, review_vector.data)
        #Finding top topn words
        topn = 10
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        topn_sorted = sorted_items[:topn]
        topn_words = []
        tfidf_val = []
        for word, value in topn_sorted:
            topn_words.append(feature_names[word])
            tfidf_val.append(round(value,3))
        d = {}
        for i in range(len(topn_words)):
            d[topn_words[i]] = tfidf_val[i]
        for key, value in d.items():
            inputTuple[key] = 1+inputTuple.get(key, 0)
    # Pick top 50 keywords from overall review corpus
    inputTuple = sorted(inputTuple.items(), key=lambda x:x[1], reverse=True)[:30]
    resultDictionary = dict((x, y) for x, y in inputTuple)
    return list(resultDictionary.keys())