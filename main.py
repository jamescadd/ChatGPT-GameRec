# https://www.pragnakalp.com/automate-youtube-video-transcription-python/

import argparse
import json
import os
from random import random

from alive_progress import alive_bar
from googleapiclient.discovery import build
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled as TranscriptsDisabledError, NoTranscriptFound

from gamerec.video import Video


def get_channel_videos(config):

    assert type(config.get('channel_id')) in [str, list]

    youtube = build('youtube', 'v3', developerKey=config.get('api_key'))

    # get playlist id(s)
    channel_ids = config.get('channel_id')
    if isinstance(channel_ids, str):
        channel_ids = [channel_ids]

    videos = []

    for channel_id in channel_ids:
        res = youtube.channels().list(id=channel_id,
                                      part='contentDetails').execute()
        playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        next_page_token = None
        loop = True

        while loop:
            res = youtube.playlistItems().list(playlistId=playlist_id,
                                               part='snippet',
                                               maxResults=50,
                                               pageToken=next_page_token).execute()
            videos += res['items']
            next_page_token = res.get('nextPageToken')

            if next_page_token is None:
                loop = False

    return videos


def get_captions_from_videos(videos, channel_id):
    video_ids = [video['snippet']['resourceId']['videoId'] for video in videos]  # list of all video_id of channel

    print(f"Extracting {len(video_ids)} transcripts from channel ID {channel_id}")

    with alive_bar(len(video_ids)) as bar:
        output_json = []

        for video_id in video_ids:
            try:
                responses = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                url = f"https://www.youtube.com/watch?v={video_id}"
                captions = [response['text'] for response in responses]
                output_json.append({"url": url, "captions": captions})
            except (TranscriptsDisabledError, NoTranscriptFound) as e:
                print(e)
            finally:
                bar()

    return output_json


def get_transcript_summary(transcript):
    """
    Get an LLM-generated summary given a video transcript
    """
    chat_ = ChatOpenAI()

    messages = [
        SystemMessage(content="You are an assistant that provides summaries of video game reviews given a transcript "
                              "of a video review for a particular video game."),
        HumanMessage(content=f"Video game review transcript: {transcript}")
    ]

    ai_message = chat_(messages)

    return ai_message.content


def summary_demo():
    with open(args.transcripts_file) as i:
        transcripts = json.load(i)

    videos = []

    for transcript in transcripts:
        videos.append(Video(transcript['url'], transcript['captions']))

    output = get_transcript_summary(videos[int(random() * len(videos))].get_transcription())
    print(f"LLM-generated summary:\n\n{output}")


def extract_transcripts(config):
    """
    Extract transcripts and save to JSON file
    """
    videos = get_channel_videos(config.get('youtube'))
    output_json = get_captions_from_videos(videos, config.get('youtube').get('channel_id'))

    with open(args.transcripts_file, 'w') as o:
        json.dump(output_json, o, indent=2)


def create_and_save_faiss_embeddings():
    embeddings = OpenAIEmbeddings(chunk_size=25)

    loader = JSONLoader(
        file_path=args.transcripts_file,
        text_content=False,
        jq_schema='.[].captions')

    print("Creating and saving FAISS embeddings...")

    with alive_bar(1) as bar:

        pages = loader.load_and_split()

        # Use LangChain to create the embeddings
        db = FAISS.from_documents(documents=pages, embedding=embeddings)

        # save the embeddings into FAISS vector store
        db.save_local("faiss_index")

        bar()


def chat():
    assert os.path.exists('faiss_index')

    def ask_question_with_context(qa_, question, chat_history_):
        result = qa_.invoke({"question": question})
        print(result)
        print("answer:", result["answer"])

    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings(chunk_size=25)

    # Initialize gpt-35-turbo and our embedding model
    # load the faiss vector store we saved into memory
    print("loading FAISS embeddings...")
    vector_store = FAISS.load_local("./faiss_index", embeddings)
    print("Done loading embeddings....")
    # use the faiss vector store we saved to search the local document
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    condense_question_prompt = PromptTemplate.from_template(
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone"
        "\nquestion, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\n"
        "Standalone question:")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key='question',
        output_key='answer',
        return_messages=True,
    )

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                               retriever=retriever,
                                               return_source_documents=True,
                                               memory=memory,
                                               condense_question_prompt=condense_question_prompt,
                                               verbose=True)

    chat_history = ''
    continue_chat = True
    while continue_chat:
        query = input('you ("q" or "quit" to quit): ')
        if query.lower() in ['q', 'quit']:
            continue_chat = False
        if continue_chat:
            ask_question_with_context(qa, query, chat_history)


def main():

    with open(args.config) as f:
        config = json.load(f)

    # set OPENAI_API_KEY env variable for LangChain
    os.environ["OPENAI_API_KEY"] = config.get("openai").get("api_key")

    if args.mode == "extract-transcripts":
        extract_transcripts(config)

    if args.mode == 'summary-demo':
        assert os.path.exists(args.transcripts_file)
        summary_demo()

    if args.mode == 'create-embeddings':
        assert os.path.exists(args.transcripts_file)
        create_and_save_faiss_embeddings()

    if args.mode == "chat-demo":
        chat()

    if args.mode == 'end-to-end':
        extract_transcripts(config)
        create_and_save_faiss_embeddings()
        chat()

    if 'OPENAI_API_KEY' in os.environ:
        os.environ.pop('OPENAI_API_KEY')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGPT GameRec')
    parser.add_argument('-c', '--config', metavar='config', type=str,
                        help='JSON config file, defaults to config.json',
                        default='config.json')
    parser.add_argument('-tf', '--transcripts-file', metavar='transcripts_file', type=str,
                        help='JSON file containing video transcripts, defaults to transcripts.json',
                        default='transcripts.json')
    parser.add_argument('-m', '--mode',
                        choices=['extract-transcripts',
                                 'create-embeddings',
                                 'summary-demo',
                                 'chat-demo',
                                 'end-to-end'],
                        help='One of "extract-transcripts", "create-embeddings", "summary-demo", "chat-demo", or'
                             '"end-to-end".  See README for more info.',
                        default='chat-demo'
                        )
    args = parser.parse_args()
    main()
