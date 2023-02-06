#Dataframe libraries
import pandas as pd
import numpy as np
import re

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')
import wordcloud
from collections import defaultdict

# NLP
from nltk.tokenize.regexp import RegexpTokenizer
from subprocess import check_output
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

#Topic modelling libraries
import gensim
from gensim import corpora, models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.ldamodel import LdaModel
from utils import text_prep_func, get_topic_keywords_yearly


class TopicModelling:
    def __init__(self, dataset, text_data_col_name, n_topics, num_words):
        self.dataset = dataset
        self.text_data_col_name = text_data_col_name
        self.n_topics = n_topics
        self.num_words = num_words
        self.viz_inputs = dict()
     
    def generate_topics(self):
        texts = []
        for review in self.dataset[self.text_data_col_name]:
            texts.append(review.split()) 
           
        #Generate corpus
        dictionary = corpora.Dictionary(texts)
        corpus_bow = [dictionary.doc2bow(text) for text in texts]
        # Get TFIDF
        tfidf = models.TfidfModel(corpus_bow)
        corpus_tfidf = tfidf[corpus_bow]

        # Create SVD object
        lda = LdaModel(corpus_tfidf, num_topics=self.n_topics, id2word=dictionary)
        result = defaultdict(list)
        for topic_num, words in lda.print_topics(num_topics=self.n_topics, num_words=self.num_words):
            temp_list = [word.replace(" ","") for word in words.split('+')]
            temp_w = []
            top_words = []
            for word in temp_list:
                proba, w = word.split("*")
                temp_w.append(w)
                top_words.append((w, proba))
            top_words = sorted(top_words, key=lambda x: x[1], reverse=True)
            for top_word in top_words:
                result[f'Topic{topic_num+1}'].append(top_word[0])
        return result

    def perform_clustering(self, N):
        #Get corpus
        wordvector = TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.4,min_df=5)
        wordvector_fit = wordvector.fit_transform(self.dataset['prep_review'])
        feature = wordvector.get_feature_names()

        #Reduce embeddings dimensionality
        svd =  TruncatedSVD(n_components = 50)
        corpus_svd = svd.fit_transform(wordvector_fit)

        #Perform clustering with corpus generated
        clf = KMeans(n_clusters=N, max_iter=50, init='k-means++', n_init=1)
        labels = clf.fit_predict(corpus_svd)
        self.dataset['cluster_label'] = labels
        self.viz_inputs['text_corpus'] = corpus_svd
        self.viz_inputs['labels'] = labels
        print('Assigned cluster labels! Print dataset variable to see labelled output...')
       
    def cluster_2d_viz(self):
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=100)
        labels = self.viz_inputs['labels']
        text_embeddings = self.viz_inputs['text_corpus']
        tsne_results = tsne.fit_transform(text_embeddings)
        tsne_result = pd.DataFrame({'tsne-2d-one': tsne_pca_results[:, 0], 'tsne-2d-two': tsne_pca_results[:, 1],'label':target})
        sns.FacetGrid(tsne_result, hue='label', height=6).map(plt.scatter, 'tsne-2d-one', 'tsne-2d-two').add_legend()
        plt.show()
        
