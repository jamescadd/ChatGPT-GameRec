import json
import os

class FileDataStore(object):
    def __init__(self, config, transcripts_file):
        self.config = config
        self.transcripts_file = transcripts_file

    def writeTranscripts(self, transcripts_json):
        with open(self.transcripts_file, 'w') as o:
            json.dump(transcripts_json, o, indent=2)

    def readTranscripts(self):
        if os.path.isfile(self.transcripts_file):
            with open(self.transcripts_file, "r") as f:
                data = json.load(f)
                return data;