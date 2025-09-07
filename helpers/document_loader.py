from langchain.document_loaders import PyPDFDirectoryLoader
def document_loader(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    return documents