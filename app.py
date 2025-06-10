# File: AiProject/app.py
from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
#from gtts import gTTS
import os

app = Flask(__name__)

# Global variables to hold the state
conversation_chain = None
pdf_text_content = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    global conversation_chain
    global pdf_text_content
    answer = None
    pdf_ready = (conversation_chain is not None)

    if request.method == 'POST':
        if 'pdf_file' in request.files and request.files['pdf_file'].filename != '':
            pdf_file = request.files['pdf_file']

            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            pdf_text_content = text

            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000,
                chunk_overlap=200, length_function=len
            )
            text_chunks = text_splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

            llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            answer = "PDF processed successfully! You can now ask a question or generate audio."
            pdf_ready = True

        user_question = request.form.get('question')
        if user_question and conversation_chain:
            response = conversation_chain({'question': user_question, 'chat_history': []})
            answer = response['answer']

    return render_template('index.html', answer=answer, pdf_ready=pdf_ready)

#@app.route('/synthesize', methods=['POST'])
#def synthesize():
#    global pdf_text_content
#    if pdf_text_content:
#        try:
#            tts = gTTS(text=pdf_text_content, lang='en', slow=False)
#            audio_file_path = os.path.join('static', 'output.mp3')
#            tts.save(audio_file_path)
#            return send_file(audio_file_path, as_attachment=False)
#        except Exception as e:
#            return str(e), 500
#    return "No PDF content available to synthesize.", 400

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)