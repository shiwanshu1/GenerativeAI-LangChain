import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain
st.title("Hi! How can i help you ❓❓")  #we can add emoji using keyword (windows+.)
btn = st.button("Create knowledgebase")
if btn:
    pass
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer: ")
    st.write(response["result"])
