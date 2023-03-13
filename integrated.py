# imports
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

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
article= ''
if check_password():
# Code for project structure
    #st.image('./asset/img/acronym-color.eps')
    st.title("A Different Storyteller")
    st.sidebar.title('A Different Storyteller')
    article = st.text_input('Enter your Text for Analysis: ',key =123)


if not article:
    st.warning('no food for thought.')
    st.stop()
else:
    st.success('Thank you for the food for thought.')

try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download('stopwords')
finally:
    from nltk.corpus import stopwords
    
# constants
sw = list(set(stopwords.words('english')))
punct = [
    '!','#','$','%','&','(',')','*',
    '+',',','-','/',':',';','<','=','>','@',
    '[','\\',']','^','_','`','{','|','}','~'
]
# Reading reviews

#displaying the entered text

#st.write('Your article is ', collected_text)

#text_cleaner
def get_analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
 

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
  


try:
    original_text_mapping, cleaned_book = clean_text(article)

except:
    st.write('time to get to work. enter something to analyize')

# get sentences
sentences = [x for x in cleaned_book.split('. ') if x not in ['', ' ', '..', '.', '...']]

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
  
sim_mat = create_similarity_matrix(sentences)

# create network
G = nx.from_numpy_matrix(sim_mat)

# calculate page rank scores
pr_sentence_similarity = nx.pagerank(G)

ranked_sentences = [
    (original_text_mapping[sent], rank) for sent,rank in sorted(pr_sentence_similarity.items(), key=lambda item: item[1], reverse = True)
]

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
  
N = 10
summary = generate_summary(ranked_sentences, N)

st.write(summary)

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
st.write(total_num_reviews)
sentiment_score = []
sentiment_subjectivity=[]

analysis_df=pd.DataFrame()
analysis_df['sentence']= sentences
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

st.write(analysis_df.head())


#berttopic modeling section
topic_model = BERTopic()



st.write(ranked_sentences)

topics, probs = topic_model.fit_transform(sentences*200)


st.write(topic_model.visualize_topics())

st.write(topic_model.get_topic_info())


"""**Visualizing the sentiment**"""

uploaded_file = st.file_uploader("Choose a file")



if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
else:
    st.stop()




uploaded_file = st.file_uploader("Choose a file")



if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
else:
    st.stop()


total_reviews_num = len(data)



# Loading Data
# Applying language detection

text_col = data['Comment'].astype(str)

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

POS = ['Noun_Count', 'Adj_Count', 'Verb_Count', 'Adv_Count', 'Pro_Count', 'Pre_Count', 'Con_Count', 'Art_Count', 'Nega_Count', 'Aux_Count']
array_Noun=[]
array_Adj=[]
array_Verb=[]
array_Adv=[]
array_Pro=[]
array_Con=[]
array_Art=[]
array_Nega=[]
array_Pre=[]
array_Aux=[]
Values = [array_Noun, array_Adj, array_Verb, array_Adv, array_Pro, array_Pre, array_Con, array_Art, array_Nega, array_Aux]

i = 0
for x in POS:
    data[x] = pd.Series(Values[i])
    data[x] = data[x].fillna(0)
    data[x] = data[x].astype(float)
    i += 1




#content= data['review-date'].astype('str').apply(datefinder.find_dates(data['review-date']))
#content

#content

"""**Removing Noise**"""

en_df['Comment'] = en_df['Comment'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

def fixContra(text):
    return contractions.fix(text)

en_df['Comment'] = en_df['Comment'].apply(lambda x: fixContra(x))
# \W represents Special characters
en_df['Comment'] = en_df['Comment'].str.replace('\W', ' ')
# \d represents Numeric digits
en_df['Comment'] = en_df['Comment'].str.replace('\d', ' ')
en_df['Comment'] = en_df['Comment'].str.lower()
en_df.head()





reviews = en_df['Comment'].tolist()
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

array_Noun = []
array_Adj = []
array_Verb = []
array_Adv = []
array_Pro = []
array_Pre = []
array_Con = []
array_Art = []
array_Nega = []
array_Aux = []

articles = ['a', 'an', 'the']
negations = ['no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'barely', 'scarcely']
auxilliary = ['am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'do', 'does', 'did', 'have', 'having', 'has', 'had']

for j in review_text:
    text = j
    filter=re.sub('[^\w\s]', '', text)
    conver_lower= filter.lower()
    Tinput = conver_lower.split(" ")

    for i in range(0, len(Tinput)):
        Tinput[i] = "".join(Tinput[i])
    Uniqe_word = Counter(Tinput)
    s = " ".join(Uniqe_word.keys())

    tokenized = sent_tokenize(s)

    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        wordsList = [w for w in wordsList if not w in stop_words]

        Art = 0
        Nega = 0
        Aux = 0
        for word in wordsList:
            if word in articles:
                Art += 1
            elif word in negations:
                Nega += 1
            elif word in auxilliary:
                Aux += 1

        tagged = nltk.pos_tag(wordsList)
        counts = Counter(tag for word,tag in tagged)

        N = sum([counts[i] for i in counts.keys() if 'NN' in i])
        Adj = sum([counts[i] for i in counts.keys() if 'JJ' in i])
        Verb = sum([counts[i] for i in counts.keys() if 'VB' in i])
        Adv = sum([counts[i] for i in counts.keys() if 'RB' in i])
        Pro = sum([counts[i] for i in counts.keys() if (('PRP' in i) or ('PRP$' in i) or ('WP' in i) or ('WP$' in i))])
        Pre = sum([counts[i] for i in counts.keys() if 'IN' in i])
        Con = sum([counts[i] for i in counts.keys() if 'CC' in i])

        array_Noun.append(N)
        array_Adj.append(Adj)
        array_Verb.append(Verb)
        array_Adv.append(Adv)
        array_Pro.append(Pro)
        array_Pre.append(Pre)
        array_Con.append(Con)
        array_Art.append(Art)
        array_Nega.append(Nega)
        array_Aux.append(Aux)
print('Completed')

POS = ['Noun_Count', 'Adj_Count', 'Verb_Count', 'Adv_Count', 'Pro_Count', 'Pre_Count', 'Con_Count', 'Art_Count', 'Nega_Count', 'Aux_Count']
Values = [array_Noun, array_Adj, array_Verb, array_Adv, array_Pro, array_Pre, array_Con, array_Art, array_Nega, array_Aux]
i = 0
for x in POS:
    en_df[x] = pd.Series(Values[i])
    en_df[x] = en_df[x].fillna(0)
    en_df[x] = en_df[x].astype(float)
    i += 1



en_df = en_df.assign(Authenticity = lambda x: (x.Pro_Count + x.Unique_words - x.Nega_Count) / x.Word_Count)


en_df = en_df.assign(AT = lambda x: 30 + (x.Art_Count + x.Pre_Count - x.Pro_Count - x.Aux_Count - x.Con_Count - x.Adv_Count - x.Nega_Count))

en_df.to_csv('before_labeling.csv')


#criteria for labeling
def label(Auth, At, N, Adj, V, Av, S, Sub, W):
    score = 0
    if Auth >= 0.49:
        score += 2
    if At <= 20:
        score += 1
    if (N + Adj) >= (V + Av):
        score += 1
    if -0.5 <= S <= 0.5:
        score += 1
    if Sub <= 0.5:
        score += 2
    if W > 75:
        score += 3
    if score >= 5:
        return 1
    else:
        return 0

en_df['Rev_Type'] = en_df.apply(lambda x: label(x['Authenticity'], x['AT'], x['Noun_Count'], x['Adj_Count'], x['Verb_Count'], x['Adv_Count'], x['Sentiment'], x['Subjectivity'], x['Word_Count']), axis = 1)


import datetime
today = datetime.date.today ()


bad_reviews = en_df[en_df['human_sentiment'] == 'Negative']
good_reviews = en_df[en_df['human_sentiment'] == 'Positive']
st.header('Select Stop Words')

custom_stopwords = st.text_input('Enter Stopword')
custom_stopwords = custom_stopwords.split()
nltk_Stop= stopwords.words("english")
final_stop_words = nltk_Stop + custom_stopwords
en_df['Rev_Type'].replace(1,'Suspected',inplace=True)
en_df['Rev_Type'].replace(0,'Real', inplace=True)

st.write('search terms used are:')
st.write(data['Search Name'].unique())

st.write( 'Source Analysis')
analyzed_total = len(data) - (len(en_df[en_df['Rev_Type']== 'Suspected'])+ len(en_df[en_df['Word_Count']==1]) + len(data[data['detect']!='en']))
youtube_total = len(data[(data['Data Source'] == "YouTube") & (data['detect']== 'en')]) - len(en_df[(en_df['Data Source'] == "YouTube") & (en_df['Rev_Type']== 'Suspected')])-len(data[(data['Data Source'] == "Youtube") & (en_df['Word_Count']==1)])
amazon_total = len(data[(data['Data Source'] == "Amazon") & (data['detect']== 'en')]) - len(en_df[(en_df['Data Source'] == "Amazon") & (en_df['Rev_Type']== 'Suspected')])-len(data[(data['Data Source'] == "Amazon") & (en_df['Word_Count']==1)])
google_total = len(data[(data['Data Source'] == "Google") & (data['detect']== 'en')]) - len(en_df[(en_df['Data Source'] == "Google") & (en_df['Rev_Type']== 'Suspected')])-len(data[(data['Data Source'] == "Google") & (en_df['Word_Count']==1)])
Trustpilot_total = len(data[(data['Data Source'] == "Trustpilot") & (data['detect']== 'en')]) - len(en_df[(en_df['Data Source'] == "Trustpilot") & (en_df['Rev_Type']== 'Suspected')])-len(data[(data['Data Source'] == "Trustpilot") & (en_df['Word_Count']==1)])
definitions= ['total number of analyzed reviews from said source','suspicious reviews basaed on linguistic features such as POS tagging', 'reviews that have one word only','Reviews in other languages','actually analyzed for output']

data = {"Columns":['Total Reviews', 'suspected fake reviews','One Word Reviews','non-English reviews','Total Analyzed']
    ,'definitions': definitions
    ,'Total':[len(data), len(en_df[en_df['Rev_Type']== 'Suspected']), len(en_df[en_df['Word_Count']==1]), len(data[data['detect']!='en']), analyzed_total ]
    ,'Youtube':[len(data[data['Data Source']== "YouTube"]), len(data[(data['Data Source'] == "YouTube") & (en_df['Rev_Type']== 'Suspected')]), len(en_df[(en_df['Data Source'] == "YouTube") & (en_df['Word_Count']== 1)]),len(data[(data['Data Source'] == "YouTube") & (data['detect']!= 'en')]), youtube_total],
    'Amazon':[len(data[data['Data Source']== "Amazon"]), len(data[(data['Data Source'] == "Amazon") & (en_df['Rev_Type']== 'Suspected')]), len(en_df[(en_df['Data Source'] == "Amazon") & (en_df['Word_Count']== 1)]), len(data[(data['Data Source'] == "Amazon") & (data['detect']!= 'en')]), amazon_total],
    'Google':[len(data[data['Data Source']== "Google"]),len(data[(data['Data Source'] == "Google") & (en_df['Rev_Type']== 'Suspected')]),len(en_df[(en_df['Data Source'] == "Google") & (en_df['Word_Count']== 1)]),len(data[(data['Data Source'] == "Google") & (data['detect']!= 'en')]),google_total],
        'Trust pilot':[len(data[data['Data Source']== "Trustpilot"]),len(data[(data['Data Source'] == "Trustpilot") & (en_df['Rev_Type']== 'Suspected')]),len(en_df[(en_df['Data Source'] == "Trustpilot") & (en_df['Word_Count']== 1)]),len(data[(data['Data Source'] == "Trustpilot") & (data['detect']!= 'en')]),Trustpilot_total]}


# Create DataFrame
df_1 = pd.DataFrame(data)
st.write(today)
st.write(df_1)
df_1 =df_1.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Analysis",
    data=df_1,
    mime='text/csv',
    file_name='analysis.csv')

#text cleaning function
def clean_text(dataframe, col_name):

    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    word = "inversely"
    print("stemming:", stem.stem(word))
    print("lemmatization:", lem.lemmatize(word, "v"))
    stop_words = final_stop_words

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

