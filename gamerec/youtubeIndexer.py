import os
from alive_progress import alive_bar
from faiss import index_factory
from googleapiclient.discovery import build
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from youtube_transcript_api._errors import TranscriptsDisabled as TranscriptsDisabledError, NoTranscriptFound

class YoutubeIndexer(object):
    collection_name = "gamerec_youtube"
    namespace = f"faiss/{collection_name}"
    db_url = "sqlite:///record_manager_cache.sql"

    def __init__(self, config):
        self.config = config
        self.embedding = OpenAIEmbeddings()
        self.record_manager = SQLRecordManager(self.namespace, db_url=self.db_url)
        self.record_manager.create_schema()
        self.youtube = build('youtube', 'v3', developerKey=config.get("youtube").get('api_key'))

    def update(self):
        """Update the faiss vector store with 'incremental' deletion mode (adds new and replaces mutated transcripts)"""  
        channel_ids = self.config.get('youtube').get('channel_id')
        videos = self.get_channel_videos(channel_ids)
        video_ids = [video['snippet']['resourceId']['videoId'] for video in videos]  # list of all video_id of channel

        if os.path.exists("./faiss_youtube"):
            self.vector_store = FAISS.load_local("./faiss_youtube", self.embedding)
        else:
            dimensions: int = len(self.embedding.embed_query("dummy"))
            index_flat = index_factory(dimensions, "Flat")
            self.vector_store = FAISS(self.embedding, index=index_flat, docstore=InMemoryDocstore(), index_to_docstore_id={})
        
        with alive_bar(len(new_video_ids)) as bar:
            for video_id in video_ids: # TODO: Remove video_ids that the record_manager shows as already being indexed, or tell the record_manager not to worry about mutations
                try:
                    loader = YoutubeLoader(video_id, continue_on_failure=True)
                    index(loader, self.record_manager, self.vector_store, cleanup="incremental", source_id_key="source")
                except (TranscriptsDisabledError, NoTranscriptFound) as e:
                    print(e)
                finally:
                    bar()

        self.vector_store.save_local("./faiss_youtube")

    def get_channel_videos(self, channel_ids):
        videos = []

        for channel_id in channel_ids:
            res = self.youtube.channels().list(id=channel_id,
                                          part='contentDetails').execute()
            playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            next_page_token = None
            loop = True

            while loop:
                res = self.youtube.playlistItems().list(playlistId=playlist_id,
                                                   part='snippet',
                                                   maxResults=50,
                                                   pageToken=next_page_token).execute()
                videos += res['items']
                next_page_token = res.get('nextPageToken')

                if next_page_token is None:
                    loop = False

        return videos
