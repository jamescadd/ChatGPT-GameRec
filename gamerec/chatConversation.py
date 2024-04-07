from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_core.vectorstores import VectorStoreRetriever

class ChatConversation(object):

    def __init__(self, retriever: VectorStoreRetriever):
        self.llm = ChatOpenAI()

        self.condense_question_prompt = PromptTemplate.from_template(
            "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone"
            "\nquestion, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\n"
            "Standalone question:")

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key='question',
            output_key='answer',
            return_messages=True,
        )
    
        self.qa = ConversationalRetrievalChain.from_llm(llm=self.llm,
                                                   retriever=retriever,
                                                   return_source_documents=True,
                                                   memory=self.memory,
                                                   condense_question_prompt=self.condense_question_prompt,
                                                   verbose=True)
    
        self.chat_history = ''

    def ask_question_with_context(self, question) -> str:
        result = self.qa({"question": question})
        print(result)
        print("answer:", result["answer"])
        return result["answer"]