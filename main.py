# https://www.pragnakalp.com/automate-youtube-video-transcription-python/

import ast
import argparse
# import getpass
import itertools
import json
import os
from random import random

from alive_progress import alive_bar
from googleapiclient.discovery import build
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled as TranscriptsDisabledError, NoTranscriptFound

from gamerec.dataStore import DataStoreType, createDataStore
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


def get_captions_from_videos(videos):
    # video_ids = [video['snippet']['resourceId']['videoId'] for video in videos]  # list of all video_id of channel

    print(f"Extracting {len(videos)} transcripts")

    with alive_bar(len(videos)) as bar:
        output_json = []

        for video in videos:
            video_id = video['snippet']['resourceId']['videoId']
            url = f"https://www.youtube.com/watch?v={video_id}"
            captions = ""
            
            try:
                responses = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                captions = [response['text'] for response in responses]
            except (TranscriptsDisabledError, NoTranscriptFound) as e:
                print(e)
            finally:
                # Always save video metadata to prevent re-attempting to pull transcripts for videos without them
                output_json.append({"channelId": video['snippet']['channelId'], "channelTitle": video['snippet']['channelTitle'], "videoId": video_id, "publishedAt": video['snippet']['publishedAt'], "url": url, "captions": captions})
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


def extract_transcripts(config, data_store):
    """
    Extract transcripts and save to JSON file
    """
    new_videos = get_channel_videos(config.get('youtube'))

    current_videos = data_store.read_transcripts()
    current_video_ids = [video['videoId'] for video in current_videos]
    new_video_ids = [video['snippet']['resourceId']['videoId'] for video in new_videos]
    delta_video_ids = set(new_video_ids).difference(set(current_video_ids))
    delta_videos = [delta_video for delta_video in new_videos if delta_video['snippet']['resourceId']['videoId'] in delta_video_ids]

    output_json = get_captions_from_videos(delta_videos)

    #with open(args.transcripts_file, 'w') as o:
    #    json.dump(output_json, o, indent=2)
    data_store.write_transcripts(current_videos + output_json)
    return output_json


def create_and_save_faiss_embeddings(delta_videos = []):
    embeddings = OpenAIEmbeddings(chunk_size=25)
    # TODO: Get a delta between videos that have an embedding saved and videos in transcriptions.json that have not yet been embedded. Below will automatically re-embed all saved documents which could be unintentionally expensive.
    with alive_bar(1) as bar:
        if (len(delta_videos) == 0):
            loader = JSONLoader(
                file_path=args.transcripts_file,
                text_content=False,
                jq_schema='.[].captions')
            print("Creating and saving FAISS embeddings...")

            pages = loader.load_and_split()

            # Use LangChain to create the embeddings
            db = FAISS.from_documents(documents=pages, embedding=embeddings)

            # save the embeddings into FAISS vector store
            db.save_local("faiss_index")
        else:
            text_splitter = RecursiveCharacterTextSplitter() # chunk_size=25, Using same as default for JSONLoader.load_and_split
            # delta_captions = [ast.literal_eval(doc['captions']) for doc in delta_videos] 
            # captions_map = map(ast.literal_eval, [doc['captions'] for doc in delta_videos])
            captions = [doc['captions'] for doc in delta_videos]
            cleaned_captions = []
            for caption in captions:
                if isinstance(caption, str) and caption.startswith('[') and caption.endswith(']'):
                    continue
                else:
                    try:
                        cleaned_captions.append(ast.literal_eval(caption))
                    except Exception:
                        cleaned_captions.append(caption)
            delta_captions = list(itertools.chain.from_iterable(cleaned_captions))
            delta_documents = text_splitter.create_documents(delta_captions) # TODO: Probably need to replicate the jq_schema='.[].captions' when loading from file

            if (os.path.exists('faiss_index')):
                vector_store = FAISS.load_local("./faiss_index", embeddings)
                vector_store.add_documents(delta_documents)
            else:
                vector_store = FAISS.from_documents(documents=delta_documents, embedding=embeddings)
            
            vector_store.save_local("faiss_index")

        bar()


def chat():
    assert os.path.exists('faiss_index')

    def ask_question_with_context(qa_, question, chat_history_):
        result = qa_({"question": question})
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

    # Setup LangSmith
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

    data_store = createDataStore(args.data_store, config, transcripts_file = args.transcripts_file)

    if args.mode == "extract-transcripts":
        extract_transcripts(config, data_store)

    if args.mode == 'summary-demo':
        assert os.path.exists(args.transcripts_file)
        summary_demo()

    if args.mode == 'create-embeddings':
        assert os.path.exists(args.transcripts_file)
        create_and_save_faiss_embeddings()

    if args.mode == 'create-update-transcripts-embeddings':
        delta_videos = extract_transcripts(config, data_store)
        create_and_save_faiss_embeddings(delta_videos)

    if args.mode == "chat-demo":
        chat()

    if args.mode == 'end-to-end':
        extract_transcripts(config, data_store)
        create_and_save_faiss_embeddings()
        chat()

    if 'OPENAI_API_KEY' in os.environ:
        os.environ.pop('OPENAI_API_KEY')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGPT GameRec')
    parser.add_argument('-c', '--config', metavar='config', type=str,
                        help='JSON config file, defaults to config.json',
                        default='config.json')
    parser.add_argument('-ds', '--data-store', metavar='data_store_type', type=DataStoreType,
                        help='Data store for transcripts and embeddings, defaults to FILESYSTEM',
                        default=DataStoreType.FILESYSTEM)
    parser.add_argument('-tf', '--transcripts-file', metavar='transcripts_file', type=str,
                        help='JSON file containing video transcripts, defaults to transcripts.json',
                        default='transcripts.json')
    parser.add_argument('-m', '--mode',
                        choices=['extract-transcripts',
                                 'create-embeddings',
                                 'create-update-transcripts-embeddings'
                                 'summary-demo',
                                 'chat-demo',
                                 'end-to-end'],
                        help='One of "extract-transcripts", "create-embeddings", "create-update-transcripts-embeddings", "summary-demo", "chat-demo", or'
                             '"end-to-end".  See README for more info.',
                        default='create-update-transcripts-embeddings'
                        )
    args = parser.parse_args()
    main()
