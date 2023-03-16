import pandas as pd
import streamlit as st
from textblob import TextBlob
from umap import UMAP
import re
import nltk
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

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


# constants
sw = list(set(stopwords.words('english')))
punct = [
    '!','#','$','%','&','(',')','*',
    '+',',','-','/',':',';','<','=','>','@',
    '[','\\',']','^','_','`','{','|','}','~'
]


#sentiment analysis score
def get_analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
 
# openai api
openai.api_key = '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'

#####----------START Image FUNCTIONS--------------------------------------------------------------------
def createImageFromPrompt(prompt):
    response = openai.Image.create(prompt=prompt, n=3, size="512x512")
    return response['data']

#####----------START Text FUNCTIONS-----------

# AI Engine general call
def openAIQuery(query):
	response = openai.Completion.create(
		engine="davinci-instruct-beta-v3",
		prompt=query,
		temperature=0.8,
		max_tokens=200,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0)

	if 'choices' in response:
		if len(response['choices']) > 0:
			answer = response['choices'][0]['text']
		else:
			answer = 'Opps sorry, you beat the AI this time'
	else:
		answer = 'Opps sorry, you beat the AI this time'
	return answer

# text cleaning using nltk
def clean_text(text, sw = sw, punct = punct):
    '''
    This function will clean the input text by lowering, removing certain punctuations, stopwords and 
    new line tags.
    
    params:
        text (String) : The body of text you want to clean
        sw (List) : The list of stopwords you wish to removed from the input text
        punct (List) : The slist of punctuations you wish to remove from the input text
        
    returns:
        This function will return the input text after it's cleaned (the output will be a string) and 
        a dictionary mapping of the original sentences with its index
    '''
    article = text.lower()
    
    # clean punctuations
    for pun in punct:
        article = article.replace(pun, '')
    
    article = article.replace("[^a-zA-Z]", " ").replace('\r\n', ' ').replace('\n', ' ')
    original_text_mapping = {k:v for k,v in enumerate(article.split('. '))}
    
    article = article.split(' ')
    
    # clean stopwords
    article = [x.lstrip().rstrip() for x in article if x not in sw]
    article = [x for x in article if x]
    article = ' '.join(article)

    return original_text_mapping, article


# similarity matrix
def create_similarity_matrix(sentences):
    '''
    The purpose of this function will be to create an N x N similarity matrix.
    N represents the number of sentences and the similarity of a pair of sentences
    will be determined through the Jaro-Winkler Score.
    
    params:
        sentences (List -> String) : This is a list of strings you want to create
                                     the similarity matrix with.
     
    returns:
        This function will return a square numpy matrix
    '''
    
    # identify sentence similarity matrix with Jaro Winkler score
    sentence_length = len(sentences)
    sim_mat = np.zeros((sentence_length, sentence_length))

    for i in range(sentence_length):
        for j in range(sentence_length):
            if i != j:
                similarity = jaro.jaro_winkler_metric(sentences[i], sentences[j])
                sim_mat[i][j] = similarity
    return sim_mat

# sumamry generation function
def generate_summary(ranked_sentences, N):
    '''
    This function will generate the summary given a list of ranked sentences and the
    number of sentences the user wants in their summary.
    
    params:
        ranked_sentences (List -> Tuples) : The list of ranked sentences where each
                                            element is a tuple, the first value in the
                                            tuple is the sentence, the second value is
                                            the rank
        N (Integer) : The number of sentences the user wants in the summary
        
    returns:
        This function will return a string associated to the summarized ranked_sentences
        of a book
    '''
    summary = '. '.join([sent[0] for sent in ranked_sentences[0:N]])
    return summary