"""rag implementation"""
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import BaseMessage, AIMessage 

"""memory implementation"""
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

"""know tools"""
import random
import string 
from operator import itemgetter
import os 
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm=ChatGroq(temperature=0,
            model_name="llama3-8b-8192",
            api_key=os.getenv("GROQ_API_KEY"))


prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a document assistant use this context :{context} to answer the question"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

def loadfile(filePath): 
    """
    this function will return a list of documents(pages)

    Args: 
        filePath(string): the full name of the file with its root

    Returns: 
        docs(list): list of Documents(langchain type) Dcument{"metadata":{} "page_content":{}}
    """
    loader=PyPDFLoader(filePath)
    docs=loader.load()
    return docs 

def splitContent(filePath): 
    docs=loadfile(filePath=filePath)

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000,
        chunk_overlap=200 
    )

    all_splits = text_splitter.split_documents(docs)

    return all_splits 

def embedContent(filePath):
    """
    this function will create a retriever and return it

    Args: 
        all_splits(list): list of splitted documents

    Returns: 
        retriever(VecotreStoreRetriever): retriever object that i can use for similarity searching  
    """
    all_splits=splitContent(filePath=filePath)
    vectore_store=FAISS.from_documents(documents=all_splits, embedding=embeddings)
    retriever=vectore_store.as_retriever(search_type="mmr", search_kwargs={"k":2})
    return retriever 



def getSessionId():
    """
    generates a random session id using random caracters and random numbers

    Returns:
        sessionId(str): combination of 12 random charactes and numbres  
    """
    chars= string.ascii_letters + string.digits
    sessionId= ''.join(random.choice(chars) for i in range(12))
    return sessionId 

def getConfig():
    """
    genreates a random configuration for the session

    Retruns:
        config(dict): configuration used by the conversationalRagChain
    """
    sessionId=getSessionId()
    config={"configurable":{"session_id":sessionId}}
    return config


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


store={}
def getSessionById(sessionId:str)->InMemoryHistory:
    """
    uses the store global variable to load the conversationHistory or create it 
    Args:
        sessionId(str): string combination representing the session id 
    Returns: 
        store[sessionId](InMemoryHistory): the instance of the chat history
    """
    if sessionId not in store:
        store[sessionId]=InMemoryHistory()
    return store[sessionId]

def getConversationalRagChain(filePath) : 
    """
    creates and returns a conversational instance for a specefic file 
    
    Args : 
        filePath(str): file path

    Returns: 
        conversationalRagChain(RunnableWithMessageHistory): instance of the RunnableWithMessageHistory class with a wrapped a memoryLess rag chain 

    """

    retriever=(itemgetter("question") | embedContent(filePath=filePath))

    chain=( {"context":retriever, "history":itemgetter("history"),"question":RunnablePassthrough()} |prompt| llm| StrOutputParser())

    chain_with_history = RunnableWithMessageHistory(
        chain,
        # Uses the get_by_session_id function defined in the example
        # above.
        getSessionById,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    return chain_with_history 


def makeQuery(query:str) -> dict:
    return {"question":query} 


def chat(conversationalRagChain:RunnableWithMessageHistory, config:dict ,query:str)->str:
    """multiple invokations in the same conversation store 
    
    Args:
        conversationalRagChain: wrapped rag chain with a conversation memory
        config: dict with a unique session id  
        query: dict the user query with a question key mark
    
    Returns:
        the response
    """
    return conversationalRagChain.invoke(makeQuery(query=query),config)



#######Testing Section################
# history=getSessionHistory("1234")
# history.add_messages(AIMessage(content="this is an ai message"))
# history.add_messages(AIMessage(content="this is another meassage from the ai"))
# print(store)

# conversation=getConversationalRagChain("etc/document.pdf")
# result=chat(conversation,getConfig(),"what is the closest person to a superior man")
# print(result)
# result=converstaion.invoke(
#     {"question":"this is a quesion"},
#     config=getConfig()
#     )
# print(result)

# config=getConfig()
# conversation = getConversationalRagChain("etc/document.pdf")

# for chunk in conversation.stream({"question":"what is a superior man"},config=config):
#     print(chunk,end="",flush=True)
# print("")
# for chunk in conversation.stream({"question":"what have asked you before this and what did you respond and why "},config=config):
#     print(chunk,end="",flush=True)
# print("")
# for chunk in conversation.stream({"question":"in that reagards what does it mean to be free as a man "},config=config):
#     print(chunk,end="",flush=True)
# print("")
# for chunk in conversation.stream({"question":"based on youre previous repond what is a high value women"},config=config):
#     print(chunk,end="",flush=True)