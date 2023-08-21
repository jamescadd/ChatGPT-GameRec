# https://www.pragnakalp.com/automate-youtube-video-transcription-python/

import json

from apiclient.discovery import build
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


def main():
    with open("config.json") as f:
        config = json.load(f)

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
    main()
