import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4"),
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

qa_chain = None

@app.route("/", methods=["GET", "POST"])
def index():
    global qa_chain
    resposta = ""
    if request.method == "POST":
        if 'pdf' in request.files:
            file = request.files['pdf']
            if file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                qa_chain = process_pdf(file_path)
                resposta = "PDF carregado com sucesso. Pode agora fazer perguntas."
        elif 'pergunta' in request.form and qa_chain:
            pergunta = request.form["pergunta"]
            resposta = qa_chain.run(pergunta)
    return render_template("index.html", resposta=resposta)

if __name__ == "__main__":
    app.run(debug=True)
