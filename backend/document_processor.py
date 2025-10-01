from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_and_chunk_documents(self):
        """Load all documents from data directory and split into chunks"""
        documents = []
        
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue
                
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks