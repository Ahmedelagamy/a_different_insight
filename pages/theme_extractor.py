
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


st.write(df_ngram_cons)
df_ngram_cons =df_ngram_cons.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download cons",
    data=df_ngram_cons,
    mime='text/csv',
    file_name='cons.csv')

"""
st.header("Dependency visualizer")
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    df = pd.DataFrame(entities, columns=('Entity', 'Label')) 
    st.write(df)
    
    
elif app_mode == "Entity Anlysis": 
    import spacy
    from spacy import displacy
    nlp = spacy.load('en_core_web_md')
    text = article
    doc = nlp(text)

else:
    #berttopic modeling section
    topic_model = BERTopic()


    topics, probs = topic_model.fit_transform(sentences*200)


    st.write(topic_model.visualize_topics())

    st.write(topic_model.get_topic_info())

