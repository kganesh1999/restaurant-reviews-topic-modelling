# Necessary libraries for ML pipeline

# Utility functions
from collections import defaultdict
from utils import text_prep_func
from subprocess import check_output
import math
import random

# Dataframe libraries
import pandas as pd
import numpy as np
import re

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns; sns.set_style('whitegrid')

# NLP
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# Topic modelling libraries
import gensim
from gensim import corpora, models
from gensim.models import LsiModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
