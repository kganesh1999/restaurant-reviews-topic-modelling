from req_libs import *


class TopicModelling:
    def __init__(self, dataset, topic_model=None):
        self.dataset = dataset
        self.model = topic_model
        self.corpus_vectors = None
        self.dictionary = None
        
    def fit_corpus(self, corpus_column, feature_vector_type):
        docs = []
        for doc in self.dataset[corpus_column]:
            docs.append(text_prep_func(doc).split())
        #Generate BOW corpus
        self.dictionary = corpora.Dictionary(docs)
#         self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus_vectors = [self.dictionary.doc2bow(doc) for doc in docs]
        print(len(self.dictionary))
        # Get TFIDF
        if feature_vector_type == 'tfidf':
            tfidf = models.TfidfModel(self.corpus_vectors)
            self.corpus_vectors = tfidf[self.corpus_vectors]
        
    def generate_topics(self, n_topics = 5, num_words=25):
        topics_overview = defaultdict(list)
        if self.model is None:
            # Create SVD object
            self.model = LdaModel(self.corpus_vectors, num_topics=n_topics, id2word=self.dictionary)
        n_topics = self.model.num_topics
        for topic_num, words in self.model.print_topics(num_topics=n_topics, num_words=num_words):
            temp_list = [word.replace(" ","") for word in words.split('+')]
            temp_w = []
            top_words = []
            for word in temp_list:
                proba, w = word.split("*")
                temp_w.append(w)
                top_words.append((w, proba))
            top_words = sorted(top_words, key=lambda x: x[1], reverse=True)
            for top_word in top_words:
                topics_overview[f'Topic{topic_num+1}'].append(top_word[0])
        return pd.DataFrame.from_dict(dict(topics_overview))
    
    def assign_topic(self):
        lda_corpus= self.model[self.corpus_vectors]
        all_topics = self.model.get_document_topics(self.corpus_vectors)
        all_topics_csr= gensim.matutils.corpus2csc(all_topics)
        all_topics_numpy= all_topics_csr.T.toarray()
        major_topic = [np.argmax(arr) for arr in all_topics_numpy]
        self.dataset['major_lda_topic']= major_topic
        print("Topic assigned for all documents :)")
    
    def visualize_topics(self, top_n_words):
        return pyLDAvis.gensim_models.prepare(self.model, self.corpus_vectors, self.dictionary, mds="mmds", R=top_n_words)
    
    #Inner class for Best n_topic LDA selection
    class BestTopicNumSelection:
        def __init__(self, start, limit, step_size, model_type, feature_vector_type):
            self.start = start
            self.limit = limit
            self.step_size = step_size 
            self.model_type = model_type
            self.feature_vector_type = feature_vector_type
            self.best_model = None
            self.best_score = 0.0
            self.best_n = 0
            self.k_scores = []

        def initiate_process(self, dataset, corpus_column):
            processed_docs = []
            for doc in dataset[corpus_column]:
                processed_docs.append(text_prep_func(doc).split())

            #Generate corpus
            dictionary = corpora.Dictionary(processed_docs)
            dictionary.filter_extremes(no_below=5, no_above=0.5)
            corpus_vectors = [dictionary.doc2bow(doc) for doc in processed_docs]
            # Get TFIDF
            if self.feature_vector_type == 'tfidf':
                tfidf = models.TfidfModel(corpus_vectors)
                self.corpus_vectors = tfidf[corpus_vectors]  
            return self.compute_K_scores(dictionary, corpus_vectors, processed_docs)

        def compute_K_scores(self, dictionary, corpus, processed_texts):
            k_models = []
            for i in range(self.start, self.limit+1, self.step_size):
                model_i = self.model_type(corpus, num_topics=i, id2word=dictionary)
                coherence_model = CoherenceModel(model=model_i, texts = processed_texts, dictionary = dictionary, coherence='c_v')
                score_i = coherence_model.get_coherence()
                self.k_scores.append(score_i)
                k_models.append(model_i)
                if score_i > self.best_score:
                    self.best_score = score_i
                    self.best_n = i
                    self.best_model = model_i
            return self.k_scores, k_models

        def get_optimal_selection(self):
            return self.best_model, self.best_n, self.best_score

        def scores_line_plot(self):
            start = self.start
            limit = self.limit+1
            step = self.step_size
            x_ticks = list(range(start, limit, step))
            plt.plot(range(len(x_ticks)), self.k_scores)
            plt.xticks(range(len(x_ticks)), x_ticks)
            plt.ylabel("Coherence score")
            plt.legend(('Coherence_Values'), loc='best')
            plt.show()

            
class DocumentClustering:
    def __init__(self, dataset, corpus_column, n_topics):
        self.dataset = dataset
        self.corpus_column = corpus_column
        self.n_topics = n_topics
        self.viz_inputs = dict()

    def perform_clustering(self):
        #Get corpus
        wordvector = TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.4,min_df=5)
        wordvector_fit = wordvector.fit_transform(self.dataset[self.corpus_column])
        feature = wordvector.get_feature_names()

        #Reduce embeddings dimensionality
        svd =  TruncatedSVD(n_components = 50)
        corpus_svd = svd.fit_transform(wordvector_fit)

        #Perform clustering with corpus generated
        clf = KMeans(n_clusters=self.n_topics, max_iter=50, init='k-means++', n_init=1)
        labels = clf.fit_predict(corpus_svd)
        self.dataset['cluster_label'] = labels
        self.viz_inputs['corpus_embeddings'] = corpus_svd
        self.viz_inputs['labels'] = labels
        print('Assigned cluster labels! Print dataset variable to see labelled output :)')
       
    def visualize_clusters(self):
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        labels = self.viz_inputs['labels']
        corpus_embeddings = self.viz_inputs['corpus_embeddings']
        tsne_results = tsne.fit_transform(text_embeddings)
        tsne_result = pd.DataFrame({'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1],'label':labels})
        sns.FacetGrid(tsne_result, hue='label', height=6).map(plt.scatter, 'tsne-2d-one', 'tsne-2d-two').add_legend()
        plt.show()
   
