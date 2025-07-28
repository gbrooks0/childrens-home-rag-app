# rag_system.py (Definitive, Complete, and Final Version - Enhanced for Expert Feedback and Analysis & SQLite fix)

import os
import tempfile
import base64
import sys # Import sys

# --- IMPORTANT: SQLite3 Fix for ChromaDB in Deployment Environments ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END SQLite3 Fix ---

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage

class RAGSystem:
    # The advanced prompt template for high-quality, formatted answers
    ADVANCED_PROMPT_TEMPLATE = """---
**ROLE AND GOAL**
You are a highly experienced and professional consultant specializing in the operation and strategic planning of children's homes, deeply knowledgeable in Ofsted regulations and best practices. Your goal is to provide clear, actionable, and expert advice, proactively anticipating user needs and offering comprehensive, analytical feedback that informs strategic decision-making, based *only* on the provided context.

---
**CONTEXT**
{context}

---
**OUTPUT RULES**
1.  **Comprehensive Synthesis and Cross-Referencing:** For questions about standards, policies, or detailed procedures, provide a complete answer by synthesizing information from multiple relevant sections of the context. Identify and explain connections or interdependencies between different standards or pieces of information.
2.  **Actionable Advice and Practical Implications:** Translate factual information into concrete, actionable recommendations or insights. For each key standard or topic, suggest practical steps a children's home can take to comply with or excel in that area, and explain the practical implications for daily operations.
3.  **Deeper Analysis and "Why/How":** Go beyond stating facts. Where applicable, explain the underlying principles or the 'why' behind a regulation, and offer insights into effective implementation strategies or 'how-to' guidance.
4.  **Nuance and Contextual Considerations:** If the context provides for variations or nuances (e.g., for specific types of children's homes or children with particular needs), include these considerations in your analysis to provide a well-rounded perspective.
5.  **Identify Potential Challenges or Areas of Focus:** Based on the provided context, identify any common challenges in meeting certain standards or key areas that often require particular attention during inspections or operations.
6.  **Stick to the Context:** Base your entire answer strictly on the CONTEXT provided above. Do not use outside knowledge.
7.  **Handle Missing Information:** If the context does not contain the information needed to answer the question, you MUST explicitly state: "Based on the provided context, I cannot answer this question." Do not try to guess or infer information that isn't present.

---
**FORMATTING RULES**
- Use Markdown for clear formatting (e.g., **bold** for emphasis, bullet points for lists, numbered lists for ordered items).
- **For Lists of Standards/Criteria:** Use a numbered list for primary standards, and nested bullet points (`*` or `-`) for elaborations under each standard.
- **IMPORTANT:** When the user's question implies a comparison between two or more items, policies, or concepts, you MUST format your core answer as a Markdown table.

---
**QUESTION**
{question}

---
**YOUR EXPERT RESPONSE:**"""

    def __init__(self):
        """Initializes the RAG system using a FAISS index with similarity search."""
        print("Initializing RAG System with FAISS...")
        # Temperature is set to 0.7 as a good balance of creative and factual
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        db_path = "faiss_index"
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{db_path}'. "
                "Please run the final ingest.py script to create it."
            )
        
        db = FAISS.load_local(
            db_path, self.embeddings, allow_dangerous_deserialization=True
        )
        
        # A simple, powerful similarity search retrieving a good number of chunks
        self.main_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 12} 
        )
        
        self.session_retriever = None
        print("RAG System Ready.")

    def process_uploaded_file(self, uploaded_file_bytes: bytes):
        """Processes a user-uploaded file in-memory and sets the session retriever."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file_bytes)
                tmp_file_path = tmp_file.name
            loader = PyPDFium2Loader(tmp_file_path)
            docs = loader.load()
        finally:
            os.remove(tmp_file_path)

        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        temp_db = Chroma.from_documents(chunks, self.embeddings)
        self.session_retriever = temp_db.as_retriever(search_kwargs={"k": 3})
        print("Temporary session retriever created.")

    def get_current_retriever(self):
        """Returns the correct retriever (ensemble or main) for the current session."""
        if self.session_retriever:
            print("Using Ensemble Retriever (FAISS + session file)")
            return EnsembleRetriever(
                retrievers=[self.main_retriever, self.session_retriever],
                weights=[0.7, 0.3]
            )
        else:
            print("Using Main FAISS Retriever")
            return self.main_retriever

    def clear_session(self):
        """Clears the session-specific retriever for uploaded files."""
        self.session_retriever = None
        print("Session retriever cleared.")

    def query(self, user_question: str, context_text: str, source_docs: list, image_bytes=None):
        """
        Queries the LLM with the provided context and passes through the source documents.
        """
        full_prompt_text = self.ADVANCED_PROMPT_TEMPLATE.format(
            context=context_text, 
            question=user_question
        )
        
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            message = HumanMessage(
                content=[
                    {"type": "text", "text": full_prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            )
            response = self.llm.invoke([message])
        else:
            response = self.llm.invoke(full_prompt_text)
            
        return {"answer": response.content, "source_documents": source_docs}
