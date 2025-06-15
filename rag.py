'''this file will contain all the necessary presteps for the rag pipline '''
#content extraction (using fitz from the PyMuPDF)

# import os 
# from dotenv import load_dotenv 
# load_dotenv() #this will load the enviroment variables from the .env file

#used this to load and extract the content 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def loadfile(filePath): 
    content=""
    with open(filePath) as document_obj: 
        reader_obj=PdfReader(filePath)
        for page in reader_obj.pages :
            content+=page.extract_text() or ""
    return content

def splitContent(content): 
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(content)
    return text_chunks

#embed and sotre
def embedContent(chunks):
    storePath="vectorDB" 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local("vectorDB",
                                embeddings,
                                allow_dangerous_deserialization=True)
        db.add_texts(chunks)
    except:
        print("creating a new vector data Base")
        db=FAISS.from_texts(chunks,embeddings)
        db.save_local("vectorDB")

#testing section 
chunks=splitContent(loadfile("etc/document2.pdf"))
embedContent(chunks=chunks)