# https://www.pragnakalp.com/automate-youtube-video-transcription-python/

import argparse
import json
import os

from alive_progress import alive_bar
from apiclient.discovery import build
import langchain
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled as TranscriptsDisabledError
 

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


def main(args_):

    with open(args_.config) as f:
        config = json.load(f)

    # set OPENAI_API_KEY env variable for langchain
    os.putenv("OPENAI_API_KEY", config.get("openai").get("api_key"))

    videos = get_channel_videos(config.get('youtube'))
    video_ids = [video['snippet']['resourceId']['videoId'] for video in videos]  # list of all video_id of channel

    print(f"Extracting {len(video_ids)} transcripts from channel ID {config.get('youtube').get('channel_id')}")

    with alive_bar(len(video_ids)) as bar, open(args.output_transcripts, 'w') as o:
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

        json.dump(output_json, o, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGPT GameRec')
    parser.add_argument('-c', '--config', metavar='config', type=str,
                        help='JSON config file', default='config.json')
    parser.add_argument('-ot', '--output-transcripts', metavar='output_transcripts', type=str,
                        help='JSON output file for video transcripts', default='output_transcripts.json')
    args = parser.parse_args()
    main(args)
