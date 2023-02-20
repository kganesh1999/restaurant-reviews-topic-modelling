# NLP
from nltk.tokenize.regexp import RegexpTokenizer
from subprocess import check_output
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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
