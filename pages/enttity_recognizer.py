import re 
import spacy
import streamlit as st
import integrated

st.title('who did what?')
nlp = spacy.load('en_core_web_md')

text = integrated.article

doc = nlp(text)

key_themes = []

for para in doc.sents: 
    para = re.sub(r'[^\w\s]','',str(para)) 
    para_doc = nlp(para) 
    para_themes = [token.text for token in para_doc if token.pos_ == 'NOUN']
    key_themes.append(para_themes)

st.text_area(key_themes)

#Output: [['data'], ['power', 'machine', 'learning'], ['potential', 'artificial', 'intelligence']]
x
for sent in doc.sents:
    for token in sent:
        # Check if token is a verb 
        if token.pos_ == "VERB":
            subject = [t for t in token.ancestors if t.dep_ == "nsubj"]
            obj = [t for t in token.ancestors if t.dep_ == "dobj"]
            print("Subject:", subject[0].text, "Verb:", token.text, "Object:", obj[0].text)

# extracting all person names 
