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
nlp = spacy.load('en_core_web_sm')
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
doc = nlp(summary)
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

st.write(vocab)
# list of ngrams
vocab_cons = c_vec.vocabulary_

# Create new dataframe from scratch
 
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
    data = pd.read_excel(uploaded_file)
else:
    st.stop()

# Loading Data
# Applying language detection

text_col = data['TEXT'].astype(str)

# Language detection
langdet = []
for i in range(len(data)):
    try:
        lang=detect(text_col[i])
    except:
        lang='no'

    langdet.append(lang)

data['detect'] = langdet
# Select language module
en_df = data[data['detect'] == 'en']


import contractions


# Sentiment Analysis
from textblob import TextBlob

#POS Tagging
import nltk
nltk.download("popular")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
stop_words = set(stopwords.words('english'))
import re




reviews = en_df['TEXT'].tolist()
sentiment_score = []
sentiment_subjectivity=[]

#Number of Negative words in a review
reviews =en_df['Comment'].tolist()
negative_count = []
for rev in reviews:
    words = rev.split()
    neg = 0
    for w in words:
        testimonial = TextBlob(w)
        score = testimonial.sentiment.polarity
        if score < 0:
            neg += 1
    negative_count.append(neg)
 
en_df['Neg_Count'] = negative_count

#Word Count
en_df['Word_Count'] = en_df['Comment'].str.split().str.len()


reviews = en_df['Comment'].str.lower().str.split()

# Get amount of unique words
en_df['Unique_words'] = reviews.apply(set).apply(len)
en_df['Unique_words'] = en_df[['Unique_words']].div(en_df.Word_Count, axis=0)
review_text = en_df['Comment']
import datetime
today = datetime.date.today ()


bad_reviews = en_df[en_df['human_sentiment'] == 'Negative']
good_reviews = en_df[en_df['human_sentiment'] == 'Positive']
st.header('Select Stop Words')

custom_stopwords = st.text_input('Enter Stopword')
custom_stopwords = custom_stopwords.split()
nltk_Stop= stopwords.words("english")
final_stop_words = nltk_Stop + custom_stopwords


# Create DataFrame
df_1 = pd.DataFrame(data)
st.write(df_1)
df_1 =df_1.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Analysis",
    data=df_1,
    mime='text/csv',
    file_name='analysis.csv')

#text cleaning function