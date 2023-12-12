import os
import uuid

from audio_recorder_streamlit import audio_recorder
import azure.cognitiveservices.speech as speechsdk
import openai
import streamlit as st

from gamerec.chatConversation import ChatConversation
from gamerec.vectorStoreFaiss import VectorStoreFaiss

TMP_AUDIO_FILENAME = "TTS.wav"

st.set_page_config(page_title="Chat",page_icon="👾")
st.title("# 🕹️ Game recommendations")

openai.api_key = st.secrets["OPENAI_API_KEY"]


def clear_audio():
    if st.session_state.audio is not None:
        st.session_state.audio = None


def recognize_speech(filename):
    """
    Starts speech recognition, and returns after a single utterance is recognized. The end of a
    single utterance is determined by listening for silence at the end or until a maximum of 15
    seconds of audio is processed.  The task returns the recognition text as result.
    Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    shot recognition like command or query.
    For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    """

    audio_config = speechsdk.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,
                                                   audio_config=audio_config)
    result = speech_recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # may want to first check if 'transcription' key is in session state, but might not want to or need to,
        # depending on how the app evolves
        if 'transcription' not in st.session_state:
            st.session_state['transcription'] = result.text
        else:
            st.session_state.transcription = result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        st.write("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        st.write("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            st.write("Error details: {}".format(cancellation_details.error_details))


if 'chatConversation' not in st.session_state:
    vectorStoreFaiss = VectorStoreFaiss("./faiss_index")
    retriever = vectorStoreFaiss.get_retriever()
    st.session_state.chatConversation = ChatConversation(retriever)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.audio is not None:
    with open(TMP_AUDIO_FILENAME, mode="wb") as f:
        f.write(st.session_state.audio)
    recognize_speech(TMP_AUDIO_FILENAME)
    st.session_state.recorder_key = str(uuid.uuid4())
    st.write(f'"{st.session_state.transcription}"')
    clear_audio()

if prompt := st.chat_input("Ask a question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.chatConversation.ask_question_with_context(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
