import openai
import streamlit as st

import textract

st.title("your writer")
st.sidebar.title('unleash your brand')
# Preprepared pipeline



demand = st.text_input('what is your question')


# Set your OpenAI API key
openai.api_key = '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'


prompt ='answer this question: {}'.format(demand)

# Text engine
response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=.7, max_tokens=1000)
# Text engine
# Allow the user to enter a prefix
st.header("""Article""")

st.write(response.choices[0].text)
