
# imports
import pandas as pd
import streamlit as st
from textblob import TextBlob
from umap import UMAP
import re
import nltk
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
#streamlit imports
import streamlit_authenticator as stauth

import re 
import spacy
import streamlit as st

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from langdetect import detect
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
stop_words = set(stopwords.words('english'))
from bertopic import BERTopic
# text summarizer imports
import networkx as nx
import numpy as np
import nltk
import jaro
import openai
import utils
import yaml
from yaml.loader import SafeLoader
# Code for  project structure
from spacy import displacy


def clean_text(dataframe, col_name):

    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    word = "inversely"
    print("stemming:", stem.stem(word))
    print("lemmatization:", lem.lemmatize(word, "v"))
    

    docs = []
    for i in dataframe[col_name]:
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', i)

        # Convert to lowercase
        text = text.lower()

        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)

        # Convert to list from string
        text = text.split()

        # Stemming
        stem=PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if word not in stop_words]

        text = " ".join(text)
        # print(text)
        docs.append(text)
    # print(docs)
    return docs


# Applying function
bad_reviews_data = clean_text(bad_reviews, 'Comment')
good_reviews_data= clean_text(good_reviews, 'Comment')
# ngram
from sklearn.feature_extraction.text import TfidfVectorizer
c_vec = TfidfVectorizer(analyzer= 'word' ,stop_words= custom_stopwords, ngram_range=(2,4))
# matrix of ngrams
ngrams = c_vec.fit_transform(good_reviews_data)
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_pros = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                       ).rename(columns={0: 'frequency', 1:'Pros'})

df_pros['percentage'] = df_pros['frequency'].apply(lambda x: (x / df_pros['frequency'].sum()*100))
st.write('Top pros')
st.write(df_pros)
df_pros =df_pros.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download pros",
    data=df_pros,
    mime='text/csv',
    file_name='pros_analysis.csv')


ngrams_cons = c_vec.fit_transform(bad_reviews_data)

# count frequency of ngrams
count_values = ngrams_cons.toarray().sum(axis=0)

# list of ngrams
vocab_cons = c_vec.vocabulary_
df_ngram_cons = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab_cons.items()], reverse=True)).rename(columns={0: 'frequency', 1:'Cons'})
df_ngram_cons['percentage'] = df_ngram_cons['frequency'].apply(lambda x: (x / df_ngram_cons['frequency'].sum()))
st.write('Top cons')


st.write(df_ngram_cons)
df_ngram_cons =df_ngram_cons.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download cons",
    data=df_ngram_cons,
    mime='text/csv',
    file_name='cons.csv')

