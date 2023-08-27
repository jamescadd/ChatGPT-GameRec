# https://www.pragnakalp.com/automate-youtube-video-transcription-python/

import argparse
import json
import os
from random import random

from alive_progress import alive_bar
from apiclient.discovery import build
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.vectorstores import FAISS
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled as TranscriptsDisabledError

from video import Video


def get_channel_videos(config):

    youtube = build('youtube', 'v3', developerKey=config.get('api_key'))

    # get Uploads playlist id
    res = youtube.channels().list(id=config.get('channel_id'),
                                  part='contentDetails').execute()
    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    videos = []
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
            except TranscriptsDisabledError as e:
                print(e)
            finally:
                bar()

    return output_json


def get_transcript_summary(transcript):
    """
    Get an LLM-generated summary given a video transcript
    """
    chat = ChatOpenAI()

    messages = [
        SystemMessage(content="You are an assistant that provides summaries of video game reviews given a transcript "
                              "of a video review for a particular video game."),
        HumanMessage(content=f"Video game review transcript: {transcript}")
    ]

    ai_message = chat(messages)

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
    embeddings = OpenAIEmbeddings(chunk_size=1)

    loader = JSONLoader(
        file_path=args.transcripts_file,
        text_content=False,
        jq_schema='.[].captions')

    pages = loader.load_and_split()

    # Use LangChain to create the embeddings
    db = FAISS.from_documents(documents=pages, embedding=embeddings)

    # save the embeddings into FAISS vector store
    db.save_local("faiss_index")


def main():

    with open(args.config) as f:
        config = json.load(f)

    # set OPENAI_API_KEY env variable for LangChain
    os.environ["OPENAI_API_KEY"] = config.get("openai").get("api_key")

    assert args.load_transcripts != args.extract_transcripts, "Must either load or extract transcripts, but not both"

    if args.extract_transcripts:
        extract_transcripts(config)

    if args.load_transcripts:
        if args.demo:
            summary_demo()
        else:
            create_and_save_faiss_embeddings()

    if args.chat:
        print("Not yet implemented")

    if 'OPENAI_API_KEY' in os.environ:
        os.environ.pop('OPENAI_API_KEY')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGPT GameRec')
    parser.add_argument('-c', '--config', metavar='config', type=str,
                        help='JSON config file', default='config.json')
    parser.add_argument('-lt', '--load-transcripts', type=bool, metavar='load_transcripts',
                        help='whether to load previously-extracted transcripts from JSON file',
                        default=True)
    parser.add_argument('-et', '--extract-transcripts', metavar='extract_transcripts', type=bool,
                        help='whether to extract transcripts for videos from channel ID given in config',
                        default=False)
    parser.add_argument('-tf', '--transcripts-file', metavar='transcripts_file', type=str,
                        help='JSON file containing video transcripts', default='transcripts.json')
    parser.add_argument('-d', '--demo', action='store_true',
                        help='Simple demo mode: output an LLM-generated summary of one random video', default=False)
    parser.add_argument('-ch', '--chat', action='store_true',
                        help='Chat demo mode: chat interactively with transcript demo store', default=False)
    args = parser.parse_args()
    main()
