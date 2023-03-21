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
#st.image('./asset/img/acronym-color.eps')


st.sidebar.title('A Different Storyteller')
st.image('logo.jfif')
st.title("42: The Knower")


try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download('stopwords')
finally:
    from nltk.corpus import stopwords


# password section
#hashed_passwords = stauth.Hasher(['abc', 'def']).generate() 
#Create login widget
#app structure
app_mode = st.sidebar.selectbox("Choose the display mode", ["Home", "Entity Anlysis", "Topic Analysis"])
# text window
article = st.text_area('Enter your Text for Analysis: ', 'What do you want analyzed')

#prevents code from running until input
if not article or article== 'What do you want analyzed':
    st.warning('no food for thought.')
    st.stop()
else:
    st.success('Thank you for the food for thought.')

#Main area 

st.header("Welcome to the storyteller hub") 

    
#Summarizer 
original_text_mapping, cleaned_book = utils.clean_text(article)
sentences = [x for x in cleaned_book.split('. ') if x not in ['', ' ', '..', '.', '...']]  
sim_mat = utils.create_similarity_matrix(sentences)
# create network
G = nx.from_numpy_matrix(sim_mat)
   # calculate page rank scores
pr_sentence_similarity = nx.pagerank(G)

ranked_sentences = [
        (original_text_mapping[sent], rank) for sent,rank in sorted(pr_sentence_similarity.items(), key=lambda item: item[1], reverse = True)
    ]


    #add user ability to edit the number of generated sentences in summary    
N = 4
summary = utils.generate_summary(ranked_sentences, N)

st.header('Highlights')

st.write(summary)

 #   dependency parser
 #   dep_svg = displacy.render(doc, style='dep', jupyter=False)
 #   st.image(dep_svg, width=400, use_column_width='never')
        # Add a section header:
    # Take the text from the input field and render the entity html.
    # Note that style="ent" indicates entities.
    #ent_html = displacy.render(doc, style='ent', jupyter=False)
    # Display the entity visualization in the browser:
    #st.markdown(ent_html, unsafe_allow_html=True)
    
    
    #st.write(displacy.serve(doc, style="ent", port= 8080))


    # sentiment analysis
    # Making result human friendly
def get_analysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

total_num_reviews= len(sentences)   
sentiment_score = []
sentiment_subjectivity=[]

analysis_df=pd.DataFrame(ranked_sentences)

for Sentence in sentences:
        testimonial = TextBlob(Sentence)
        sentiment_score.append(testimonial.sentiment.polarity)
        sentiment_subjectivity.append(testimonial.sentiment.subjectivity)

analysis_df['Sentiment'] = sentiment_score
analysis_df['Subjectivity'] = sentiment_subjectivity

analysis_df['human_sentiment'] = analysis_df['Sentiment'].apply(get_analysis)


    #Visualiing the distribution of Sentiment
pos = 0
neg = 0
for score in analysis_df['Sentiment']:
        if score > 0:
            pos += 1
        elif score < 0:
            neg += 1


values = [pos, neg]
label = ['Positive Sentences', 'Negative Sentences']

fig = plt.figure(figsize =(10, 7))
plt.pie(values, labels = label)

st.pyplot(fig)

st.write(analysis_df)

from sklearn.feature_extraction.text import TfidfVectorizer
c_vec = TfidfVectorizer(analyzer= 'word' , ngram_range=(1,2))
# matrix of ngrams 
ngrams = c_vec.fit_transform(sentences)
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
ngram_df = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                       ).rename(columns={0: 'frequency', 1:'Pros'})
st.write(ngram_df[:20])


st.header("What is this text about")



#ktrain.text.get_topic_model(sentences, n_topics=20, n_features=1000, min_df=2, max_df=0.95)
#ktrain
topic_model = BERTopic()


topics, probs = topic_model.fit_transform(sentences*200)


st.write(topic_model.visualize_topics())

st.write(topic_model.get_topic_info())

    #options = ['Tab 1', 'Tab 2'] selection = st.sidebar.selectbox('Select an option', options)
###if selection == 'Tab 1': st.subheader('Tab 1') st.write('You have selected Tab 1') elif selection == 'Tab 2': st.subheader('Tab 2') st.write('You have selected Tab 2')

#Create a sidebar with different options 
# Reading reviews


#displaying the entered text

#st.write('Your article is ', collected_text)

#text_cleaner

# generation prompt building
#submission = st.text_input('What do you want to do with your text')
#query = "extract the main topics and themes from this document in a table: {}".format(submission)
#openAIAnswerUnformatted = utils.openAIQuery(query)
#openAIAnswer = openAIAnswerUnformatted
#st.write(openAIAnswer)






uploaded_file = st.file_uploader("Choose a file")



if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.stop()

# Loading Data
# Applying language detection

text_col = data['TEXT'].astype(str)

