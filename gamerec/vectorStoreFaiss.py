import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreFaiss(object):

    def __init__(self, folder_path: str):
        assert os.path.exists(folder_path)
        self.folder_path = folder_path

    def get_retriever(self):
        embeddings = OpenAIEmbeddings(chunk_size=25)

        # Initialize gpt-35-turbo and our embedding model
        # load the faiss vector store we saved into memory
        print("loading FAISS embeddings...")
        
        vector_store = FAISS.load_local(self.folder_path,
                                        embeddings,
                                        allow_dangerous_deserialization=True)
        print("Done loading embeddings....")
        # use the faiss vector store we saved to search the local document
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        return retriever