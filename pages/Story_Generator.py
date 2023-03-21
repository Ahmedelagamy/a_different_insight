import openai
import streamlit as st


st.image('./logo.jfif')
st.title("42: The StoryTeller")
st.sidebar.title('Tell Your Side')
# Preprepared pipeline
# generation specific form
with st.form("my_form"):
        name_option = st.text_input("What is the name of your company?")
        industry = st.text_input("Enter the field of the industry?")
        competitive_edge = st.text_input("what stands out about your company?")
        submitted = st.form_submit_button("Submit")




demand = st.text_input('what do you need?')

#prevents code from running until input
if not demand or demand== 'what do you need?':
    st.warning('no food for thought.')
    st.stop()
else:
    st.success('Thank you for the food for thought.')



# Set your OpenAI API key
openai.api_key = st.secrets['AI_KEY']


prompt ='write a {} for a company named.{} that works in {} and focuses on the company\'s strengths that include {}: '.format(demand, name_option , industry, competitive_edge)

# Text engine
response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=.5, max_tokens=1000)
# Text engine
# Allow the user to enter a prefix
st.header("""your 42 Consultation is:""")

st.write(response.choices[0].text)
