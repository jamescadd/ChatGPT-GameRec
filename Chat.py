import streamlit as st
import openai
from gamerec.chatConversation import ChatConversation 
from gamerec.vectorStoreFaiss import VectorStoreFaiss

st.set_page_config(page_title="Chat",page_icon="👾")
st.title("# 🕹️ Game recommendations")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if 'chatConversation' not in st.session_state:
    vectorStoreFaiss = VectorStoreFaiss("./faiss_index")
    retriever = vectorStoreFaiss.get_retriever()
    st.session_state.chatConversation = ChatConversation(retriever)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.chatConversation.ask_question_with_context(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
