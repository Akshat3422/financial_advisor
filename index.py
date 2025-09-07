from dotenv import load_dotenv
import os   
from pinecone import Pinecone
from helpers.document_loader import document_loader
from helpers.document_splitter import chunk_splitter
from helpers.prompt import prompt
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from helpers.format_document import format_docs
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import requests


load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")


URL="https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2023-24/ar/annual-report-2023-2024.pdf"
filename=os.path.join(os.getcwd(),"documents","tcs.pdf")
response=requests.get(URL,stream=True)
if response.status_code==200:
    with open(filename,"wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
        print(f"File saved successfully as {filename}")


# Loading the documents
documents=document_loader(r"C:\Users\user\Desktop\Financial_Analyzer\documents")

# Splitting the documents into chunks
splitted_doc=chunk_splitter(documents)

# Using embeddings  
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ['GOOGLE_API_KEY'])
print("Embeddings created successfully }")

# Using Pinecone as vector store

# Import the Pinecone library
# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create a dense index with integrated embedding
index_name = "langchain"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )


index=PineconeVectorStore.from_documents(documents=splitted_doc,embedding=embeddings,index_name=index_name)

# Function to create retrival query
def retrive_query(query,k=2):
    matching_result=index.similarity_search(query=query,k=k)
    return matching_result



# Retrieval
parallel_chain = RunnableParallel({
    'context': RunnableLambda(retrive_query) | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser=StrOutputParser()
chain= parallel_chain | prompt | model | parser

question="How resilient was TCS’ performance compared to the previous year’s growth momentum?"
print(chain.invoke(question))