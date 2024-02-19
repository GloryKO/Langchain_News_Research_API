from flask import Flask,request, jsonify
import pickle
import time
import os
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

app =  Flask(__name__)

llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.9, max_tokens=500)
file_path = "faiss_store_openai.pkl"

@app.route('/process_urls',methods=['POST']) #define route for processing the urls
def process_urls():
    data = request.json.get('urls',[])
    #load the data (the urls)
    loader = UnstructuredURLLoader(urls=data)
    data = loader.load()

    #create a text splitter object 
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],
        chunk_size=1000)
    #split the data and save in docs
    docs = text_splitter.split_documents(data)

    #create embeddings using openAI embeddings and save in FAISS 
    embeddings = OpenAIEmbeddings()
    vectorestore_openai = FAISS.from_documents(docs,embeddings)
    time.sleep(2)

    #save FAISS index into a pickle file

    with open(file_path,"wb") as f :
        pickle.dump(vectorestore_openai,f)
    
    return jsonify({"message":"URLs processed successfully"})

@app.route('/ask_questions',methods=['POST'])
def ask_questions():
    question = request.json.get('question','')
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorestore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorestore.as_retriever())
            result = chain({"question":question},return_only_outputs=True)

            response = {
                "answer":result["answer"],
                "source":result.get("sources",[])
            }
    else:
        return jsonify({"error":"FAISS index not found"})

if __name__ == "__main__":
    app.run(debug=True) 