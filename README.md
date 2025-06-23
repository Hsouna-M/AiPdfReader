### Required Libarys to install 
- Flask
- PyPDF2 or PyMuPDF
- langchain 
- langchain_community
- faiss-cpu (for vector storing) or chromadb (high level easier option )
- transformers
- sentence-transformers
- datasets
- langchain_together or langchain_groq 
- gTTS

1. activate a virtual environment `python3 -m venv venvName`
2. install necassary libarys 
```shell
pip install Flask langchain faiss-cpu transformers....
```
3. activate the virtual environment `source venvName/bin/activate`
4. run the project `flask run`

*i might do this project in a docker container it seems the suitable for this case*

