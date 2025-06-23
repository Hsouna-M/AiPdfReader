import os #will read variables from the enviroment
from dotenv import load_dotenv 

# File: AiProject/app.py
from flask import Flask, render_template, request ,redirect
import rag


load_dotenv() #this will load the enviroment variables from the .env file

app = Flask(__name__)

CONFIG=rag.getConfig()
UPLOAD_FOLDER="./etc"
app.config['upload_folder']=UPLOAD_FOLDER
conversation=None

@app.route("/", methods=["GET", "POST"])
#upload the pdf file page
def index():
    if request.method=="POST" : 
        try :   
            if 'pdf_file' in request.files and request.files['pdf_file']!="":
                fileStorageObject=request.files["pdf_file"]
                save_path = os.path.join(app.config['upload_folder'], fileStorageObject.filename)
                fileStorageObject.save(dst=save_path)
                conversation = rag.getConversationalRagChain(filePath=save_path)
                return redirect('/chat')
        except :
            return render_template('error.html')
    #if GET we will render the index template
    return render_template('index.html')


@app.route("/chat", methods=["GET","POST"])
#here will be going the rag pt
# rocess and the conversation
def hello():
    return render_template('chat.html')


if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
