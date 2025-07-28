# rag_system.py (Definitive, Complete, and Final Version - Enhanced for Expert Feedback and Analysis & SQLite fix & Diagnostic Prints)

import os
import tempfile
import base64
import sys # Import sys

# --- IMPORTANT: SQLite3 Fix for ChromaDB in Deployment Environments ---
# This ensures that chromadb uses a compatible SQLite version if the system's is outdated.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("DEBUG: pysqlite3 imported and set as default sqlite3 module.")
except ImportError:
    print("DEBUG: pysqlite3 not found, falling back to system sqlite3.")
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
        print("DEBUG: Initializing RAG System...")

        # Check for API Key explicitly
        api_key_status = "Not Set"
        if "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"]:
            api_key_status = "Found and Non-Empty"
        print(f"DEBUG: GOOGLE_API_KEY status: {api_key_status}")

        print("DEBUG: Attempting to initialize LLM (Gemini 1.5 Pro) and Embeddings (text-embedding-004)...")
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            print("DEBUG: LLM and Embeddings initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize LLM or Embeddings. This often means GOOGLE_API_KEY is missing or invalid. Details: {e}")
            raise

        db_path = "faiss_index"
        print(f"DEBUG: Checking for FAISS index at: '{db_path}'...")

        if not os.path.exists(db_path):
            print(f"ERROR: FAISS index directory NOT FOUND at '{db_path}'. This is a critical error.")
            raise FileNotFoundError(
                f"FAISS index not found at '{db_path}'. "
                "Please ensure you have run ingest.py locally to create it, and then committed and pushed the 'faiss_index' directory to your GitHub repository."
            )

        print(f"DEBUG: FAISS index directory found. Attempting to load from '{db_path}'...")
        try:
            db = FAISS.load_local(
                db_path, self.embeddings, allow_dangerous_deserialization=True
            )
            print("DEBUG: FAISS index loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load FAISS index from '{db_path}'. Details: {e}")
            raise

        # A simple, powerful similarity search retrieving a good number of chunks
        self.main_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 12} 
        )
        print("DEBUG: Main retriever initialized.")

        self.session_retriever = None
        print("DEBUG: RAG System initialization complete.")

    def process_uploaded_file(self, uploaded_file_bytes: bytes):
        """Processes a user-uploaded file in-memory and sets the session retriever."""
        print("DEBUG: Processing uploaded file...")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file_bytes)
                tmp_file_path = tmp_file.name
            print(f"DEBUG: Temporary file created at {tmp_file_path}")
            loader = PyPDFium2Loader(tmp_file_path)
            docs = loader.load()
            print(f"DEBUG: Loaded {len(docs)} documents from temporary file.")
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                print(f"DEBUG: Temporary file {tmp_file_path} removed.")

        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        print(f"DEBUG: Split uploaded document into {len(chunks)} chunks.")
        temp_db = Chroma.from_documents(chunks, self.embeddings)
        self.session_retriever = temp_db.as_retriever(search_kwargs={"k": 3})
        print("DEBUG: Temporary session retriever created.")

    def get_current_retriever(self):
        """Returns the correct retriever (ensemble or main) for the current session."""
        if self.session_retriever:
            print("DEBUG: Using Ensemble Retriever (FAISS + session file)")
            return EnsembleRetriever(
                retrievers=[self.main_retriever, self.session_retriever],
                weights=[0.7, 0.3]
            )
        else:
            print("DEBUG: Using Main FAISS Retriever")
            return self.main_retriever

    def clear_session(self):
        """Clears the session-specific retriever for uploaded files."""
        self.session_retriever = None
        print("DEBUG: Session retriever cleared.")

    def query(self, user_question: str, context_text: str, source_docs: list, image_bytes=None):
        """
        Queries the LLM with the provided context and passes through the source documents.
        """
        print("DEBUG: Starting query process...")
        full_prompt_text = self.ADVANCED_PROMPT_TEMPLATE.format(
            context=context_text, 
            question=user_question
        )

        if image_bytes:
            print("DEBUG: Image bytes provided, preparing for multimodal query.")
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            message = HumanMessage(
                content=[
                    {"type": "text", "text": full_prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            )
            response = self.llm.invoke([message])
        else:
            print("DEBUG: No image bytes provided, performing text-only query.")
            response = self.llm.invoke(full_prompt_text)

        print("DEBUG: LLM response received.")
        return {"answer": response.content, "source_documents": source_docs}
