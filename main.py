import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain
import base64
import time

st.markdown("""# Welcome to <span style="color:red">redBus</span> chatbot🤖""", unsafe_allow_html=True)
st.markdown(" ")

file_ = open("redBus_Gif.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
unsafe_allow_html=True,
)

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
question = st.text_input("Enter your Query on redBus: ")

if question:

    placeholder=st.empty()
    #st.write('Data is migrating to MongoDB🚀...')
    placeholder.write('Thinking...')
    time.sleep(10)
    chain= get_qa_chain()
    response = chain(question)
    # placeholder.empty()
    st.header("Your Solution: ")
    st.write(response["result"])
