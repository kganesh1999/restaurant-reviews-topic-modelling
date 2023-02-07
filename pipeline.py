#Dataframe libraries
import pandas as pd
import numpy as np
import re

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('whitegrid')
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
from gensim.models import LsiModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

#Utility functions
from utils import text_prep_func


class ComputeCoherenceScores:
    def __init__(self, start, limit, step_size, base_model):
        self.start = start
        self.limit = limit
        self.step_size = step_size  
        self.base_model = base_model
        self.best_score = 0.0
        self.best_n = 0
        self.k_scores = []

    def compute_K_scores(self, dictionary, corpus, processed_texts):
        for i in range(self.start, self.limit+1, self.step_size):
            model_i = self.base_model(corpus, num_topics=i, id2word=dictionary)
            coherence_model = CoherenceModel(model=model_i, texts = processed_texts, dictionary = dictionary, coherence='c_v')
            score_i = coherence_model.get_coherence()
            self.k_scores.append(score_i)
            if score_i > self.best_score:
                self.best_score = score_i
                self.best_n = i
        return self.k_scores

    def scores_line_plot(self):
        start = self.start
        limit = self.limit+1
        step = self.step_size
        x = range(start, limit, step)
        b_score = float(self.best_score)
        b_num_topics = self.best_n
        plt.plot(x, self.k_scores)
        plt.scatter(b_num_topics, b_score)
        plt.ylabel("Coherence score")
        plt.legend(('Coherence_Values'), loc='best')
        plt.show()


class TopicModelling:
    def __init__(self, dataset, text_data_col_name, n_topics, num_words):
        self.dataset = dataset
        self.text_data_col_name = text_data_col_name
        self.n_topics = n_topics
        self.num_words = num_words
        self.topics_overview = defaultdict(list)
        self.viz_inputs = dict()
     
    def generate_topics(self):
        texts = []
        for review in self.dataset[self.text_data_col_name]:
            texts.append(review.split()) 
           
        #Generate corpus
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=10, no_above=0.2)
        corpus_bow = [dictionary.doc2bow(text) for text in texts]
        # Get TFIDF
        tfidf = models.TfidfModel(corpus_bow)
        corpus_tfidf = tfidf[corpus_bow]

        # Create SVD object
        lda = LdaModel(corpus_tfidf, num_topics=self.n_topics, id2word=dictionary)
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
                self.topics_overview[f'Topic{topic_num+1}'].append(top_word[0])
        print('Topics are generated! Call display_topics method for results...')

    def display_topics(self):
        return pd.DataFrame.from_dict(dict(self.topics_overview))

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
       
    def visualize_clusters(self):
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        labels = self.viz_inputs['labels']
        text_embeddings = self.viz_inputs['text_corpus']
        tsne_results = tsne.fit_transform(text_embeddings)
        tsne_result = pd.DataFrame({'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1],'label':labels})
        sns.FacetGrid(tsne_result, hue='label', height=6).map(plt.scatter, 'tsne-2d-one', 'tsne-2d-two').add_legend()
        plt.show()
        
