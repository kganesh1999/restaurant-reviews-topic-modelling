from req_libs import *


class TopicModelling:
    def __init__(self, dataset, topic_model=None):
        self.dataset = dataset
        self.model = topic_model
        self.corpus_vectors = None
        self.dictionary = None
        self.docs = None
        
    def fit_corpus(self, corpus_column, feature_vector_type):
        self.docs = []
        for doc in self.dataset[corpus_column]:
            self.docs.append(text_prep_func(doc).split())
        #Generate BOW corpus
        self.dictionary = corpora.Dictionary(self.docs)
        self.dictionary.filter_extremes(no_below=5, no_above=0.7)
        self.corpus_vectors = [self.dictionary.doc2bow(doc) for doc in self.docs]
        # Get TFIDF
        if feature_vector_type == 'tfidf':
            tfidf = models.TfidfModel(self.corpus_vectors)
            self.corpus_vectors = tfidf[self.corpus_vectors]
        
    def generate_topics(self, n_topics = 5, num_words=25):
        topics_overview = defaultdict(list)
        if self.model is None:
            # Create SVD object
            self.model = LdaModel(self.corpus_vectors, num_topics=n_topics, id2word=self.dictionary, iterations=200)
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
    
    def format_topics_sentences(self):
        sent_topics_df = pd.DataFrame()
        for i, row_list in enumerate(self.model[self.corpus_vectors]):
            row = row_list[0] if self.model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            topic_num, prob_topic = row[0]
            word_prob_pairs = self.model.get_topic_terms(topic_num)            
            topic_keywords = ", ".join([self.dictionary[wordid] for wordid, prob in word_prob_pairs])
            sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prob_topic,4), topic_keywords]), ignore_index=True)
        contents = pd.Series(self.docs)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        sent_topics_df.columns = ['Dominant_topic', 'Perc_Contribution', 'Topic_keywords', 'Text']
        return sent_topics_df
    
    def assign_topic(self):
        lda_corpus= self.model[self.corpus_vectors]
        all_topics = self.model.get_document_topics(self.corpus_vectors)
        all_topics_csr= gensim.matutils.corpus2csc(all_topics)
        all_topics_numpy= all_topics_csr.T.toarray()
        major_topic = [np.argmax(arr) for arr in all_topics_numpy]
        self.dataset['LDA_topic_assigned']= major_topic
        print("Topic assigned for all documents :)")
    
    def topics_wordcloud(self, topn=10):        
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
        cloud = WordCloud(stopwords = STOPWORDS,
                          background_color='white',
                          width=2500,
                          height=1800,
                          max_words=100,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        topics = self.model.show_topics(num_words=topn, formatted=False)
        if len(topics) < 4:
            r, c = len(topics), 1
        elif len(topics) < 11:
            r, c = math.ceil(len(topics)/2), 2
        else:
            # I've chosen to have a maximum of 3 columns
            r, c = math.ceil(k/3), 3
        fig, axes = plt.subplots(r, c, figsize=(10,10), sharex=True, sharey=True)
        for i, ax in enumerate(axes.flatten()):
            if i==len(topics):
                break
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()
    
    def visualize_topics(self, top_n_words):
        return pyLDAvis.gensim_models.prepare(self.model, self.corpus_vectors, self.dictionary, mds="mmds", R=top_n_words)
    
    def tsne_clustering(self):
        topic_weights = []
        for i, row_list in enumerate(self.model[self.corpus_vectors]):
            topic_weights.append([w for i, w in row_list])
        arr = pd.DataFrame(topic_weights).fillna(0).values
        arr = arr[np.amax(arr, axis=1) > 0.35]
        topic_num = np.argmax(arr, axis=1)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(arr)
        tsne_result = pd.DataFrame({'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1],'label':topic_num})
        sns.FacetGrid(tsne_result, hue='label', height=6).map(plt.scatter, 'tsne-2d-one', 'tsne-2d-two').add_legend()
        plt.show()
 
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
            dictionary.filter_extremes(no_below=5, no_above=0.7)
            corpus_vectors = [dictionary.doc2bow(doc) for doc in processed_docs]
            # Get TFIDF
            if self.feature_vector_type == 'tfidf':
                tfidf = models.TfidfModel(corpus_vectors)
                self.corpus_vectors = tfidf[corpus_vectors]  
            return self.compute_K_scores(dictionary, corpus_vectors, processed_docs)

        def compute_K_scores(self, dictionary, corpus, processed_texts):
            k_models = []
            for i in range(self.start, self.limit+1, self.step_size):
                model_i = self.model_type(corpus, num_topics=i, id2word=dictionary, iterations = 200)
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
            
