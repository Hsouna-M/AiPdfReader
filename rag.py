#used this to load and extract the content 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os 
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm=ChatGroq(temperature=0,
            model_name="llama3-8b-8192",
            api_key=os.getenv("GROQ_API_KEY"))


template="""you are a document assistant , use the context provided in addition to youre knowledge to better answer the user questions , use simple language and always say thank you for asking in the begining of each answer 

context:{context}

question:{question} 

answer:  
"""

prompt=PromptTemplate.from_template(template)

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

def splitContent(docs): 
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000,
        chunk_overlap=200 
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits 

#embed and sotre
def embedContent(all_splits):
    """
    this function will create a retriever and return it
    Args: 
        all_splits(list): list of splitted documents
    Returns: 
        retriever(VecotreStoreRetriever): retriever object that i can use for similarity searching  
    """
    vectore_store=FAISS.from_documents(documents=all_splits, embedding=embeddings)
    retriever=vectore_store.as_retriever(search_type="mmr", search_kwargs={"k":2})
    return retriever 

def ragPipe(filename , query) : 
    listdocs=loadfile(filename) 
    docs_splits=splitContent(listdocs)
    print(docs_splits)

    retriever=embedContent(docs_splits)

    rag_chain=(
    {"context":retriever, "question": RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
    )
    
    return rag_chain.invoke(query)