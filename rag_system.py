# rag_system.py (Updated for UI Refinements and Ensemble Retriever for Multiple Chroma Collections)

import os
import tempfile
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class RAGSystem:
    def __init__(self):
        """Initializes the CORE components of the RAG system."""
        print("Initializing RAG System...")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.8)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # --- MODIFICATION START ---
        # Load the main persistent vector stores (local_docs and web_docs)
        # It's crucial to specify the collection_name when loading
        print("Loading 'local_docs' Chroma collection...")
        self.local_db = Chroma(
            persist_directory="db", 
            embedding_function=self.embeddings, 
            collection_name="local_docs"
        )
        print("Loading 'web_docs' Chroma collection...")
        self.web_db = Chroma(
            persist_directory="db", 
            embedding_function=self.embeddings, 
            collection_name="web_docs"
        )
        
        # Set up retrievers for each persistent DB
        self.local_retriever = self.local_db.as_retriever(search_kwargs={"k": 7})
        self.web_retriever = self.web_db.as_retriever(search_kwargs={"k": 7})

        # Combine them into an EnsembleRetriever as the main retriever
        print("Setting up EnsembleRetriever for main RAG...")
        self.main_retriever = EnsembleRetriever(
            retrievers=[self.local_retriever, self.web_retriever],
            weights=[0.6, 0.4] # Adjust weights as desired (e.g., give more weight to local docs)
        )
        # --- MODIFICATION END ---

        # Set up the conversational chain with memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key='answer'
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.main_retriever, # Start with the combined main retriever
            memory=self.memory,
            return_source_documents=True # Keep this for source retrieval
        )
        print("RAG System initialized.")

    def load_documents_to_session_retriever(self, file_content_base64: str):
        """
        Loads document content from base64 string into a temporary in-memory Chroma DB
        and returns a retriever for it.
        """
        print("Loading documents to session retriever...")
        
        # Create a temporary file to store the decoded content
        # Ensure the temp file has a .pdf extension for PyPDFium2Loader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(base64.b64decode(file_content_base64))
            tmp_file_path = temp_file.name

        docs = []
        try:
            print(f"Loading temporary file: {tmp_file_path}")
            loader = PyPDFium2Loader(tmp_file_path)
            docs = loader.load()
        finally:
            os.remove(tmp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        temp_db = Chroma.from_documents(chunks, self.embeddings)
        temp_retriever = temp_db.as_retriever(search_kwargs={"k": 3})
        
        print("Temporary file retriever created.")
        return temp_retriever

    def update_retriever(self, session_retriever=None):
        """Updates the chain's retriever to be an ensemble if a session retriever exists."""
        if session_retriever:
            print("Updating to Ensemble Retriever")
            final_retriever = EnsembleRetriever(
                retrievers=[self.main_retriever, session_retriever],
                weights=[0.6, 0.4]
            )
        else:
            print("Updating to Main Retriever")
            final_retriever = self.main_retriever
        
        self.chain.retriever = final_retriever

    def query(self, user_question: str):
        """Queries the RAG system using its currently configured retriever."""
        return self.chain.invoke({"question": user_question})

    def clear_memory(self):
        """Clears the internal conversational memory."""
        self.memory.clear()
        print("Conversational memory cleared.")
