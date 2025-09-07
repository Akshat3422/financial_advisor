from fastapi import FastAPI, HTTPException, UploadFile, File, Query
import os
import requests
from helpers.document_loader import document_loader
from helpers.document_splitter import chunk_splitter
from helpers.prompt import prompt
from helpers.format_document import format_docs
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# ----------------- Initialize -----------------
app = FastAPI()
DOCS_DIR = os.path.join(os.getcwd(), "documents")
os.makedirs(DOCS_DIR, exist_ok=True)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ['GOOGLE_API_KEY']
)

# Global Pinecone index
INDEX_NAME = "langchain"
index = None  # Will be initialized after loading documents

# RunnableChain placeholder
chain = None


# ----------------- Utility Functions -----------------
def update_index():
    """Load documents, split, and update Pinecone index."""
    global index, chain
    documents = document_loader(DOCS_DIR)
    if not documents:
        return False
    splitted_doc = chunk_splitter(documents)
    index = PineconeVectorStore.from_documents(
        documents=splitted_doc,
        embedding=embeddings_model,
        index_name=INDEX_NAME
    )
    # Build RunnableChain
    parallel_chain = RunnableParallel({
        'context': RunnableLambda(retrive_query) | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    chain = parallel_chain | prompt | model | parser
    print("Pinecone index and chain updated successfully")
    return True


def retrive_query(inputs, k=2):
    """Retrieval function for RunnableLambda."""
    query = inputs["question"]
    return index.similarity_search(query=query, k=k)


# ----------------- API Endpoints -----------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Fetcher API. Use /fetch_document or /upload_document"}


@app.post("/fetch_document/")
def fetch_document(url: str = Query(..., description="URL of the document to download")):
    try:
        file_name = url.split("/")[-1] or "downloaded_file.pdf"
        file_path = os.path.join(DOCS_DIR, file_name)

        # Download file
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Update Pinecone index after fetching
        update_index()
        return {"message": "File saved and indexed successfully", "file_path": file_path}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")


@app.post("/upload_document/")
async def upload_document(upload_file: UploadFile = File(...)):
    file_path = os.path.join(DOCS_DIR, upload_file.filename)

    if os.path.exists(file_path):
        return {"message": "File already exists", "file_path": file_path}

    with open(file_path, "wb") as f:
        f.write(await upload_file.read())

    update_index()
    return {"message": "File uploaded and indexed successfully", "file_path": file_path}



@app.get("/submit/")
def submit_question(question: str = Query(..., description="Your question for the documents")):
    if not chain:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload or fetch a document first.")
    try:
        result = chain.invoke({"question": question})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



