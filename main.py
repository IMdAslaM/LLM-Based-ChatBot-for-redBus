import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain


st.markdown("""# Welcome to <span style="color:red">redBus</span> chatbot🤖""", unsafe_allow_html=True)

question = st.text_input("Enter your Query on redBus: ")

if question:
    chain= get_qa_chain()
    response = chain(question)

    st.header("Your Solution: ")
    st.write(response["result"])