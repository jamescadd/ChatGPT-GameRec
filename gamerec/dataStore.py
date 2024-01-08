from enum import Enum

from gamerec.fileDataStore import FileDataStore

DataStoreType = Enum('DataStoreType', ['FILESYSTEM', 'MONGODB'])

def createDataStore(dataStoreType, config, **kwargs):
    if dataStoreType == DataStoreType.FILESYSTEM:
        return FileDataStore(config, kwargs["transcripts_file"])
    elif dataStoreType == DataStoreType.MONGODB:
        raise NotImplementedError("todo - Mongo")
    else:
        raise ValueError("Invalid type")