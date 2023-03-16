import openai
import streamlit as st


st.title("42: The Story Teller")
st.sidebar.title('Tell Your Side')
# Preprepared pipeline



demand = st.text_input('what do you need?')

#prevents code from running until input
if not demand or demand== 'What do you want analyzed':
    st.warning('no food for thought.')
    st.stop()
else:
    st.success('Thank you for the food for thought.')



# Set your OpenAI API key
openai.api_key ='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'


prompt ='answer this question: {}'.format(demand)

# Text engine
response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=.7, max_tokens=1000)
# Text engine
# Allow the user to enter a prefix
st.header("""Brand Generator""")

st.write(response.choices[0].text)

# generation specific form
with st.form("my_form"):
        name_option = st.selectbox("Select Name", ['A', 'B', 'C'])
        gender_option = st.text_input("Enter the field of the industry")
        date_option = st.text_input("what is your purpose")
        submitted = st.form_submit_button("Submit")

