import streamlit as st
import openai

st.set_page_config(page_title="Chat",page_icon="👾")
st.title("# 🕹️ Game recommendations")

openai.api_key = st.secrets["OPENAI_API_KEY"]




if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question [q or quit to exit]"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
