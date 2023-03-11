import streamlit as st

def main(): 
    st.title('Login Page') 
    username = st.text_input('Username') 
    password = st.text_input('Password', type='password') 
    remember_me = st.checkbox('Remember me') 
    btn = st.button('Login') 
    if btn and username == 'admin' and password == 'password': 
        st.write('Login successful!') 
    else: 
        st.write('Login failed, please check your credentials')

if name == 'main': main()