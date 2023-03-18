import openai
import streamlit as st


st.title("42: The Story Teller")
st.sidebar.title('Tell Your Side')
# Preprepared pipeline
# generation specific form
with st.form("my_form"):
        name_option = st.text_input("What is the name of your company")
        industry = st.text_input("Enter the field of the industry")
        submitted = st.form_submit_button("Submit")




demand = st.text_input('what do you need?')

#prevents code from running until input
if not demand or demand== 'what do you need?':
    st.warning('no food for thought.')
    st.stop()
else:
    st.success('Thank you for the food for thought.')



# Set your OpenAI API key
openai.api_key ='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'


prompt ='write a {} for a company named.{} that works in {}: '.format(demand, name_option , industry)

# Text engine
response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=.7, max_tokens=1000)
# Text engine
# Allow the user to enter a prefix
st.header("""Brand Generator""")

st.write(response.choices[0].text)
