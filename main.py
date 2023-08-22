# https://www.pragnakalp.com/automate-youtube-video-transcription-python/

import argparse
import json
import os

from apiclient.discovery import build
import langchain
import openai
from youtube_transcript_api import YouTubeTranscriptApi
 

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

    videos = get_channel_videos(config['youtube'])
    video_ids = []  # list of all video_id of channel

    for video in videos:
        video_ids.append(video['snippet']['resourceId']['videoId'])

    for video_id in video_ids:
        try:
            responses = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            print(f"\nVideo: https://www.youtube.com/watch?v={video_id}\n\nCaptions:")
            for response in responses:
                text = response['text']
                print(text)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGPT GameRec')
    parser.add_argument('-c', '--config', metavar='config', type=str,
                        help='JSON config file', default='config.json')
    args = parser.parse_args()
    main(args)
